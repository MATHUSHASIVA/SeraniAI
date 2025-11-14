"""
Handles all clarification-related logic for task creation and updates.
Manages conversation state during multi-turn clarification flows.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re


class ClarificationHandler:
    """Manages clarification flows for task creation and updates."""
    
    @staticmethod
    def check_missing_task_info(task_intent: Dict) -> List[str]:
        """Check what information is missing from task intent."""
        missing_info = []
        if not task_intent.get("due_date") or not task_intent.get("due_time"):
            missing_info.append("due date and time")
        return missing_info
    
    @staticmethod
    def parse_relative_reminder(pending_task: Dict, message: str) -> bool:
        """Parse relative reminder time (e.g., '30 minutes before')."""
        before_match = re.search(r'(\d+)\s*(minute|hour|min|hr)s?\s*before', message.lower()) or \
                      re.search(r'before\s+(\d+)\s*(minute|hour|min|hr)s?', message.lower())
        
        if before_match and pending_task.get("due_date") and pending_task.get("due_time"):
            amount = int(before_match.group(1))
            unit = before_match.group(2)
            
            due_datetime = datetime.strptime(
                f"{pending_task['due_date']} {pending_task['due_time']}", 
                "%Y-%m-%d %H:%M"
            )
            
            if 'hour' in unit or 'hr' in unit:
                reminder_datetime = due_datetime - timedelta(hours=amount)
            else:
                reminder_datetime = due_datetime - timedelta(minutes=amount)
            
            pending_task["reminder_date"] = reminder_datetime.strftime("%Y-%m-%d")
            pending_task["reminder_time"] = reminder_datetime.strftime("%H:%M")
            return True
        
        # Check for simple yes/default
        if message.strip().lower() in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
            if pending_task.get("due_date") and pending_task.get("due_time"):
                due_datetime = datetime.strptime(
                    f"{pending_task['due_date']} {pending_task['due_time']}", 
                    "%Y-%m-%d %H:%M"
                )
                reminder_datetime = due_datetime - timedelta(minutes=30)
                pending_task["reminder_date"] = reminder_datetime.strftime("%Y-%m-%d")
                pending_task["reminder_time"] = reminder_datetime.strftime("%H:%M")
                return True
        
        return False
    
    @staticmethod
    def parse_absolute_reminder(pending_task: Dict, timing_intent: Dict, message: str) -> bool:
        """Parse absolute reminder time."""
        if timing_intent.get("due_date") and timing_intent.get("due_time"):
            pending_task["reminder_date"] = timing_intent["due_date"]
            pending_task["reminder_time"] = timing_intent["due_time"]
            return True
        elif timing_intent.get("reminder_date") and timing_intent.get("reminder_time"):
            pending_task["reminder_date"] = timing_intent["reminder_date"]
            pending_task["reminder_time"] = timing_intent["reminder_time"]
            return True
        return False
    
    @staticmethod
    def handle_due_datetime_clarification(pending_task: Dict, timing_intent: Dict) -> tuple:
        """
        Handle clarification for due date/time.
        Returns (success: bool, response: str, needs_reminder: bool)
        """
        if timing_intent.get("due_date") and timing_intent.get("due_time"):
            pending_task["due_date"] = timing_intent["due_date"]
            pending_task["due_time"] = timing_intent["due_time"]
            return True, "Got it! I've added it to your calendar. Do you need a reminder before you leave?", True
        else:
            return False, "Hmm, I didn't catch the date and time. Could you tell me when? 🤔", False
    
    @staticmethod
    def handle_reminder_clarification(pending_task: Dict, timing_intent: Dict, message: str) -> tuple:
        """
        Handle clarification for reminder.
        Returns (success: bool, needs_finalization: bool)
        """
        if "no" in message.lower() and "yes" not in message.lower():
            pending_task["reminder_date"] = None
            pending_task["reminder_time"] = None
            return True, True
        
        # Try to parse relative time
        reminder_set = ClarificationHandler.parse_relative_reminder(pending_task, message)
        
        if not reminder_set:
            # Try absolute time
            reminder_set = ClarificationHandler.parse_absolute_reminder(pending_task, timing_intent, message)
        
        return reminder_set, reminder_set
    
    @staticmethod
    def determine_conflict_resolution_target(conflicting_task: Dict, pending_task: Dict, 
                                            message: str) -> str:
        """
        Determine which task to reschedule based on user message.
        Returns 'old', 'new', or 'ambiguous'
        """
        msg_lower = message.lower()
        conflict_title = (conflicting_task.get('title') or '').lower() if conflicting_task else ''
        pending_title = (pending_task.get('task_title') or '').lower()
        
        # Check if user explicitly mentions the OLD conflicting task
        mentions_old_task = conflict_title and conflict_title in msg_lower
        
        # Check if user explicitly mentions the NEW pending task
        mentions_new_task = pending_title and pending_title in msg_lower
        
        # Determine which task to reschedule based on explicit mentions
        if mentions_new_task and not mentions_old_task:
            return 'new'
        elif mentions_old_task and not mentions_new_task:
            return 'old'
        else:
            # Ambiguous: check for time-related keywords to reschedule NEW task (default behavior)
            new_task_keywords = ["schedule", "then", "at", "to", "change", "move", "shift"]
            if any(word in msg_lower for word in new_task_keywords):
                return 'new'
        
        return 'ambiguous'
    
    @staticmethod
    def request_task_clarification(task_intent: Dict, missing_info: List[str], 
                                   message: str, llm, conversation_state: Dict) -> str:
        """
        Request clarification for missing task information.
        Updates conversation state and generates natural clarification question.
        """
        from .prompts import PromptTemplates
        from langchain_core.messages import HumanMessage, SystemMessage
        
        conversation_state["awaiting_clarification"] = True
        conversation_state["pending_task"] = task_intent
        conversation_state["original_message"] = message
        conversation_state["clarification_type"] = "due_datetime" if "due date and time" in missing_info else "reminder_datetime"
        
        # Generate natural clarification question
        clarification_prompt = PromptTemplates.build_clarification_prompt(task_intent, missing_info)
        
        response = llm.invoke([
            SystemMessage(content=clarification_prompt),
            HumanMessage(content=f"Generate question for: {missing_info[0]}")
        ])
        
        return response.content.strip()
    
    @staticmethod
    def finalize_clarified_task(user_id: int, pending_task: Dict, username: str, 
                               task_agent, conversation_state: Dict) -> tuple:
        """
        Finalize and create the clarified task.
        Returns (success: bool, message: str)
        """
        pending_task["is_task_request"] = True
        pending_task["confidence"] = 1.0
        
        success, result_message, task_id, conflicts = task_agent.create_task_from_intent(
            user_id, pending_task, ""
        )
        
        # Clear clarification state
        ClarificationHandler.clear_clarification_state(conversation_state)
        
        if success:
            if pending_task.get("reminder_date") and pending_task.get("reminder_time"):
                reminder_time_obj = datetime.strptime(pending_task['reminder_time'], "%H:%M")
                reminder_time_12h = reminder_time_obj.strftime("%I:%M %p").lstrip("0")
                return True, f"Perfect! I'll remind you at {reminder_time_12h} ✅"
            else:
                return True, "Perfect! Task added successfully ✅"
        else:
            return False, f"Hmm, had trouble with that. {result_message} 🤔"
    
    @staticmethod
    def clear_clarification_state(conversation_state: Dict):
        """Clear all clarification-related state."""
        conversation_state["awaiting_clarification"] = False
        conversation_state["clarification_type"] = None
        conversation_state["pending_task"] = None
        conversation_state["original_message"] = None
        conversation_state["initial_message_causing_clarification"] = None
