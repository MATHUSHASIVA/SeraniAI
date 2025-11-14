"""
Task handling operations for MainAgent.
Contains logic for task creation, updates, queries, and conflict resolution.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

from .response_formatter import ResponseFormatter
from .clarification_handler import ClarificationHandler


class TaskHandlers:
    """Handles all task-related operations."""
    
    def __init__(self, task_agent, db_manager, llm):
        self.task_agent = task_agent
        self.db_manager = db_manager
        self.llm = llm
    
    def check_multiple_tasks(self, message: str) -> bool:
        """Detect if message contains multiple tasks."""
        msg_lower = message.lower()
        indicators = [' and ', ' & ', ' plus ']
        time_words = ['at', 'pm', 'am', 'o\'clock']
        
        if any(ind in msg_lower for ind in indicators):
            time_count = sum(msg_lower.count(word) for word in time_words)
            return time_count >= 2
        return False
    
    def handle_multiple_tasks(self, user_id: int, message: str, context: str, 
                             username: str, context_agent, conversation_buffer: List) -> str:
        """
        Comprehensive handler for messages containing multiple tasks.
        Detects, splits, and creates multiple tasks from a single message.
        """
        try:
            # Import needed for LLM prompts
            from .prompts import PromptTemplates
            from langchain_core.messages import HumanMessage, SystemMessage
            
            # Split message into task segments using LLM
            system_prompt = PromptTemplates.build_multiple_tasks_split_prompt()
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Split tasks: {message}")
            ])
            
            # Parse LLM response
            content = response.content.strip().replace('```json', '').replace('```', '').strip()
            tasks = json.loads(content)
            
            created_tasks = []
            for task_data in tasks:
                task_text = task_data.get('task_text', '')
                if task_text:
                    # Parse each task intent
                    task_intent = self.task_agent.parse_task_intent(task_text, user_id, context)
                    
                    # Create task if valid
                    if task_intent.get("is_task_request") and task_intent.get("task_title"):
                        success, msg, task_id, conflicts = self.task_agent.create_task_from_intent(
                            user_id, task_intent, context
                        )
                        
                        if success:
                            created_tasks.append(task_intent.get("task_title"))
            
            # Store conversation summary if tasks were created
            if created_tasks:
                self.store_conversation_summary(user_id, username, context_agent, conversation_buffer)
                
                task_list = ", ".join(created_tasks)
                return f"Perfect! I've added {len(created_tasks)} tasks: {task_list} ✅"
            else:
                return "I found multiple tasks but need more details. Could you add them one at a time? 🤔"
                
        except Exception as e:
            return "I see you mentioned multiple tasks. Let me add them one by one - what's the first one? 😊"
    
    def find_recent_task(self, tasks: List[Dict]) -> Optional[Dict]:
        """Find the most recently mentioned or created task."""
        if not tasks:
            return None
        
        now = datetime.now()
        sorted_tasks = sorted(
            tasks, 
            key=lambda t: t.get('updated_at', t.get('created_at', '')), 
            reverse=True
        )
        
        # Check for very recent activity (within 5 minutes)
        for task in sorted_tasks:
            try:
                time_str = task.get('updated_at') or task.get('created_at')
                if time_str:
                    task_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    if now - task_time < timedelta(minutes=5):
                        return task
            except Exception:
                continue
        
        # Return most recent overall
        return sorted_tasks[0] if sorted_tasks else None
    
    def filter_tasks_by_timeframe(self, all_tasks: List[Dict], message: str, 
                                  today: datetime.date) -> Tuple[List[Dict], str]:
        """Filter tasks based on timeframe mentioned in message."""
        message_lower = message.lower()
        tomorrow = today + timedelta(days=1)
        
        if "today" in message_lower:
            filtered = [t for t in all_tasks if t.get('due_date') == today.strftime('%Y-%m-%d')]
            return filtered, "today"
        elif "tomorrow" in message_lower:
            filtered = [t for t in all_tasks if t.get('due_date') == tomorrow.strftime('%Y-%m-%d')]
            return filtered, "tomorrow"
        elif "this week" in message_lower or "week" in message_lower:
            week_end = today + timedelta(days=7)
            filtered = [t for t in all_tasks if t.get('due_date') and 
                       today.strftime('%Y-%m-%d') <= t.get('due_date') <= week_end.strftime('%Y-%m-%d')]
            return filtered, "this week"
        
        return all_tasks, "all"
    
    def reschedule_old_task(self, conflicting_task: Dict, pending_task: Dict, 
                           message: str, context: str, user_id: int, username: str,
                           context_agent, conversation_buffer: List) -> str:
        """Reschedule the old conflicting task."""
        new_timing = self.task_agent.parse_task_intent(message, user_id, context)
        
        if not (new_timing.get("due_date") or new_timing.get("due_time")):
            return "I didn't catch the new time. Could you specify when to reschedule? (e.g., 7 AM)"
        
        old_task_id = conflicting_task.get("id")
        if not old_task_id:
            return "I couldn't identify which existing task to reschedule. Could you name it?"
        
        update_params = {}
        if new_timing.get("due_date"):
            update_params["due_date"] = new_timing["due_date"]
        if new_timing.get("due_time"):
            update_params["due_time"] = new_timing["due_time"]
        
        # Update old task
        self.db_manager.update_task(old_task_id, **update_params)
        
        # Create new pending task at original time
        pending_task["is_task_request"] = True
        pending_task["confidence"] = 1.0
        success, msg, task_id, _ = self.task_agent.create_task_from_intent(user_id, pending_task, context)
        
        if success:
            # Store conversation summary immediately
            self.store_conversation_summary(user_id, username, context_agent, conversation_buffer)
            
            # OLD task was rescheduled, NEW task kept its original time
            old_title = conflicting_task.get("title")
            new_title = pending_task.get("task_title")
            old_new_time = update_params.get("due_time", conflicting_task.get("due_time"))
            new_orig_time = pending_task["due_time"]
            
            return ResponseFormatter.format_reschedule_confirmation(
                old_title, new_title, old_new_time, new_orig_time
            )
        else:
            return f"I rescheduled {conflicting_task.get('title')}, but had trouble creating the new task. {msg}"
    
    def reschedule_new_task(self, pending_task: Dict, message: str, context: str, 
                           user_id: int, username: str, context_agent, conversation_buffer: List) -> str:
        """Reschedule the new pending task to a different time."""
        new_timing = self.task_agent.parse_task_intent(message, user_id, context)
        
        if not (new_timing.get("due_date") and new_timing.get("due_time")):
            return "I didn't catch the new time. Could you specify when? (e.g., 6 PM)"
        
        pending_task["due_date"] = new_timing["due_date"]
        pending_task["due_time"] = new_timing["due_time"]
        pending_task["is_task_request"] = True
        pending_task["confidence"] = 1.0
        
        success, result_message, task_id, new_conflicts = self.task_agent.create_task_from_intent(
            user_id, pending_task, context
        )
        
        if success:
            # Store conversation summary immediately
            self.store_conversation_summary(user_id, username, context_agent, conversation_buffer)
            
            # NEW task was rescheduled to avoid conflict
            task_title = pending_task.get("task_title")
            date_obj = datetime.strptime(pending_task["due_date"], "%Y-%m-%d")
            day_name = date_obj.strftime("%A")
            time_12h = datetime.strptime(pending_task["due_time"], "%H:%M").strftime("%I:%M %p").lstrip("0")
            
            return f"Perfect! ✅ {task_title} scheduled for {day_name} at {time_12h}."
        else:
            return f"Hmm, had trouble with that. {result_message} 🤔"
    
    def store_conversation_summary(self, user_id: int, username: str, context_agent, 
                                   conversation_buffer: List, llm=None):
        """
        Consolidated helper to store conversation summary.
        Generates LLM-based summary and stores in ChromaDB.
        """
        if not conversation_buffer:
            return
            
        try:
            from .prompts import PromptTemplates
            from langchain_core.messages import HumanMessage, SystemMessage
            
            # Use provided LLM or fallback to self.llm
            llm_instance = llm if llm else self.llm
            
            # Build conversation text
            lines = []
            for exchange in conversation_buffer:
                lines.append(f"User: {exchange['user']}")
                lines.append(f"Assistant: {exchange['assistant']}")
            conversation_text = "\n".join(lines)
            
            # Generate summary
            system_prompt = PromptTemplates.build_conversation_summary_prompt(username)
            
            response = llm_instance.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Summarize this conversation:\n\n{conversation_text}")
            ])
            
            summary = response.content.strip()
            if summary:
                context_agent.store_conversation_summary(
                    user_id=user_id,
                    summary=summary,
                    start_time=conversation_buffer[0]["timestamp"],
                    end_time=conversation_buffer[-1]["timestamp"],
                    conversation_metadata={
                        "type": "task_conversation",
                        "message_count": len(conversation_buffer)
                    }
                )
                
                # Clear buffer after storing
                conversation_buffer.clear()
                
        except Exception:
            pass
