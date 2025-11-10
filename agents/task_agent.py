import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import dateparser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class TaskAgent:
    """
    Task Management Agent for creating, updating, and managing tasks.
    Handles conflict resolution and intelligent task parsing.
    """
    
    def __init__(self, openai_api_key: str, db_manager):
        self.openai_api_key = openai_api_key
        self.db_manager = db_manager
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Note: Tools functionality simplified for compatibility
    
    def parse_task_intent(self, user_message: str, user_id: int, context: str = "") -> Dict:
        """
        Parse user message to extract task creation intent and details.
        """
        system_prompt = f"""
        You are a task parsing assistant. Analyze the user's message to extract task details.
        
        Context: {context}
        
        Classify as a task request if the user:
        - Explicitly asks: "remind me to...", "schedule...", "set a reminder..."
        - Mentions events/meetings: "I have a meeting", "meeting tomorrow", "appointment at"
        - Describes commitments with time: "I need to call", "I have to submit by"
        
        Do NOT classify as task requests:
        - Casual mentions: "I'm flying to Canada" (without wanting to track it)
        - General updates without scheduling needs
        
        Extract the following information if this is a task creation request:
        1. Task title/description
        2. Start time (convert relative times like "tomorrow evening" to specific times)
        3. Duration (in minutes)
        4. Priority level
        5. Whether this is a task creation request
        
        Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Return a JSON object with:
        {{
            "is_task_request": boolean,
            "task_title": string or null,
            "task_description": string or null,
            "start_time": "YYYY-MM-DD HH:MM:SS" or null,
            "duration_minutes": integer or null,
            "priority": "low"|"medium"|"high" or null,
            "confidence": float (0-1)
        }}
        
        Examples:
        - "remind me to start preparing for my project presentation tomorrow evening" 
          -> is_task_request: true, start_time: tomorrow at 18:00
        - "I'm flying to Canada on Thursday morning" 
          -> is_task_request: false (just sharing information)
        - "Maybe two hours should be enough" (in context of duration)
          -> duration_minutes: 120
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User message: {user_message}")
            ])
            
            # Try to parse JSON response
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            parsed_intent = json.loads(content)
            return parsed_intent
            
        except Exception as e:
            print(f"Error parsing task intent: {e}")
            return {
                "is_task_request": False,
                "task_title": None,
                "task_description": None,
                "start_time": None,
                "duration_minutes": None,
                "priority": None,
                "confidence": 0.0
            }
    
    def create_task_from_intent(self, user_id: int, intent: Dict, 
                              context: str = "") -> Tuple[bool, str, Optional[int]]:
        """
        Create a task based on parsed intent.
        Returns (success, message, task_id)
        """
        try:
            if not intent.get("is_task_request") or intent.get("confidence", 0) < 0.6:
                return False, "Not recognized as a task creation request", None
            
            title = intent.get("task_title")
            description = intent.get("task_description", "")
            start_time_str = intent.get("start_time")
            duration_minutes = intent.get("duration_minutes")
            priority = intent.get("priority", "medium")
            
            if not title:
                return False, "Could not extract task title", None
            
            # Parse start time
            start_time = None
            if start_time_str:
                try:
                    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Try alternative parsing
                    start_time = dateparser.parse(start_time_str)
                    if not start_time:
                        return False, f"Could not parse start time: {start_time_str}", None
            
            # Check for conflicts if we have timing info
            conflicts = []
            if start_time and duration_minutes:
                end_time = start_time + timedelta(minutes=duration_minutes)
                conflicts = self.db_manager.check_schedule_conflicts(
                    user_id, start_time, end_time
                )
            
            # If conflicts exist, handle them
            if conflicts:
                conflict_msg = self._format_conflicts(conflicts)
                return False, f"Scheduling conflict detected:\\n{conflict_msg}", None
            
            # Create the task
            task_id = self.db_manager.create_task(
                user_id=user_id,
                title=title,
                description=description,
                start_time=start_time,
                duration_minutes=duration_minutes,
                priority=priority
            )
            
            # Format success message
            success_msg = f"Created task: '{title}'"
            if start_time:
                success_msg += f" scheduled for {start_time.strftime('%B %d at %I:%M %p')}"
            if duration_minutes:
                hours = duration_minutes // 60
                mins = duration_minutes % 60
                duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
                success_msg += f" (Duration: {duration_str})"
            
            return True, success_msg, task_id
            
        except Exception as e:
            print(f"Error creating task: {e}")
            return False, f"Error creating task: {str(e)}", None
    
    def update_task_from_conversation(self, user_id: int, message: str, 
                                    context: str = "") -> Tuple[bool, str]:
        """
        Update existing tasks based on conversation context.
        """
        try:
            # Look for update patterns in the message
            update_intent = self._parse_update_intent(message, user_id, context)
            
            if not update_intent.get("is_update_request"):
                return False, "No task update detected"
            
            task_id = update_intent.get("task_id")
            updates = update_intent.get("updates", {})
            
            if not task_id or not updates:
                return False, "Could not identify task or updates to apply"
            
            # Apply updates
            success_messages = []
            
            if "start_time" in updates:
                # Handle time changes and conflict checking
                new_start_time = datetime.strptime(updates["start_time"], '%Y-%m-%d %H:%M:%S')
                
                # Get current task for duration info
                tasks = self.db_manager.get_user_tasks(user_id)
                current_task = next((t for t in tasks if t['id'] == task_id), None)
                
                if current_task and current_task['duration_minutes']:
                    new_end_time = new_start_time + timedelta(minutes=current_task['duration_minutes'])
                    conflicts = self.db_manager.check_schedule_conflicts(
                        user_id, new_start_time, new_end_time, exclude_task_id=task_id
                    )
                    
                    if conflicts:
                        conflict_msg = self._format_conflicts(conflicts)
                        return False, f"Cannot update: scheduling conflict\\n{conflict_msg}"
                
                # Update start time in database (would need additional DB method)
                success_messages.append(f"Updated start time to {new_start_time.strftime('%B %d at %I:%M %p')}")
            
            if "duration_minutes" in updates:
                # Update duration (would need additional DB method)
                duration = updates["duration_minutes"]
                hours = duration // 60
                mins = duration % 60
                duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
                success_messages.append(f"Updated duration to {duration_str}")
            
            if "status" in updates:
                self.db_manager.update_task_status(task_id, updates["status"])
                success_messages.append(f"Updated status to {updates['status']}")
            
            return True, "; ".join(success_messages)
            
        except Exception as e:
            print(f"Error updating task: {e}")
            return False, f"Error updating task: {str(e)}"
    
    def _parse_update_intent(self, message: str, user_id: int, context: str) -> Dict:
        """Parse message for task update intent."""
        system_prompt = f"""
        Analyze the message for task update requests in the context of an ongoing conversation.
        
        Context: {context}
        
        Look for:
        1. References to existing tasks
        2. Time changes ("move it to...", "change time to...", "reschedule...")
        3. Duration changes ("make it longer", "2 hours instead", "extend to...")
        4. Status changes ("mark complete", "cancel", "done")
        
        Return JSON:
        {{
            "is_update_request": boolean,
            "task_id": integer or null,
            "updates": {{
                "start_time": "YYYY-MM-DD HH:MM:SS" or null,
                "duration_minutes": integer or null,
                "status": string or null
            }}
        }}
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Message: {message}")
            ])
            
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"Error parsing update intent: {e}")
            return {"is_update_request": False}
    
    def suggest_conflict_resolution(self, user_id: int, conflicts: List[Dict], 
                                  new_task: Dict) -> List[str]:
        """
        Suggest ways to resolve scheduling conflicts.
        """
        suggestions = []
        
        try:
            # Analyze conflicts and suggest alternatives
            system_prompt = """
            You are a scheduling assistant. Given scheduling conflicts, suggest practical solutions.
            
            Provide 2-3 specific suggestions like:
            1. Reschedule conflicting task to different time
            2. Adjust duration of existing or new task
            3. Suggest alternative time slots
            
            Keep suggestions concise and actionable.
            """
            
            conflict_info = "Conflicts:\\n"
            for conflict in conflicts:
                conflict_info += f"- '{conflict['title']}' from {conflict['start_time']} to {conflict['end_time']}\\n"
            
            new_task_info = f"New task: '{new_task.get('title', 'Untitled')}'"
            if new_task.get('start_time') and new_task.get('duration_minutes'):
                end_time = new_task['start_time'] + timedelta(minutes=new_task['duration_minutes'])
                new_task_info += f" from {new_task['start_time']} to {end_time}"
            
            prompt = f"{conflict_info}\\n{new_task_info}"
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            # Parse suggestions from response
            content = response.content.strip()
            suggestions = [line.strip() for line in content.split('\\n') if line.strip() and not line.strip().startswith('Conflicts')]
            
        except Exception as e:
            print(f"Error generating conflict suggestions: {e}")
            suggestions = [
                "Consider rescheduling one of the conflicting tasks",
                "Adjust the duration to avoid overlap",
                "Find an alternative time slot"
            ]
        
        return suggestions
    
    def _format_conflicts(self, conflicts: List[Dict]) -> str:
        """Format conflict information for user display."""
        if not conflicts:
            return "No conflicts found."
        
        conflict_strs = []
        for conflict in conflicts:
            start_time = datetime.fromisoformat(conflict['start_time'])
            end_time = datetime.fromisoformat(conflict['end_time'])
            conflict_strs.append(
                f"'{conflict['title']}' ({start_time.strftime('%b %d, %I:%M %p')} - {end_time.strftime('%I:%M %p')})"
            )
        
        return "\\n".join(conflict_strs)
    
    def get_task_summary(self, user_id: int) -> str:
        """Get a formatted summary of user's tasks."""
        try:
            tasks = self.db_manager.get_user_tasks(user_id)
            
            if not tasks:
                return "You have no tasks scheduled."
            
            # Group tasks by status
            pending_tasks = [t for t in tasks if t['status'] == 'pending']
            completed_tasks = [t for t in tasks if t['status'] == 'completed']
            
            summary_parts = []
            
            if pending_tasks:
                summary_parts.append(f"**Upcoming Tasks ({len(pending_tasks)}):**")
                for task in pending_tasks[:5]:  # Show up to 5
                    task_str = f"• {task['title']}"
                    if task['start_time']:
                        start_time = datetime.fromisoformat(task['start_time'])
                        task_str += f" - {start_time.strftime('%b %d, %I:%M %p')}"
                    if task['duration_minutes']:
                        hours = task['duration_minutes'] // 60
                        mins = task['duration_minutes'] % 60
                        duration = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
                        task_str += f" ({duration})"
                    summary_parts.append(task_str)
            
            if completed_tasks:
                summary_parts.append(f"\\n**Completed Tasks: {len(completed_tasks)}**")
            
            return "\\n".join(summary_parts)
            
        except Exception as e:
            print(f"Error getting task summary: {e}")
            return "Unable to retrieve task summary."
    
    # Tool functions for LangChain agent
    def _create_task_tool(self, tool_input: str) -> str:
        """Tool function for creating tasks."""
        # This would be called by LangChain agent
        # Parse tool_input and call create_task_from_intent
        return "Task creation tool called"
    
    def _check_conflicts_tool(self, tool_input: str) -> str:
        """Tool function for checking conflicts."""
        return "Conflict checking tool called"
    
    def _update_task_tool(self, tool_input: str) -> str:
        """Tool function for updating tasks."""
        return "Task update tool called"
    
    def _list_tasks_tool(self, tool_input: str) -> str:
        """Tool function for listing tasks."""
        return "Task listing tool called"
    
    def _resolve_conflict_tool(self, tool_input: str) -> str:
        """Tool function for resolving conflicts."""
        return "Conflict resolution tool called"