import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import TaskPrompts, clean_json_response


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
    
    def parse_task_intent(self, user_message: str, user_id: int, context: str = "") -> Dict:
        """
        Parse user message to extract task creation intent and details.
        """
        system_prompt = TaskPrompts.build_task_parsing_prompt(context)
        user_content = f"User message: {user_message}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            
            content = clean_json_response(response.content)
            return json.loads(content)
            
        except Exception as e:
            return self._get_empty_task_intent()
    
    def _get_empty_task_intent(self) -> Dict:
        """Return empty task intent structure."""
        return {
            "is_task_request": False,
            "task_title": None,
            "description": None,
            "due_date": None,
            "due_time": None,
            "reminder_date": None,
            "reminder_time": None,
            "confidence": 0.0
        }
    
    def create_task_from_intent(self, user_id: int, intent: Dict, 
                              context: str = "") -> Tuple[bool, str, Optional[int], Optional[List[Dict]]]:
        """
        Create a task based on parsed intent.
        Returns (success, message, task_id, conflicts)
        """
        try:
            if not self._is_valid_task_intent(intent):
                return False, "Not recognized as a task creation request", None, None
            
            title = intent.get("task_title")
            if not title:
                return False, "Could not extract task title", None, None
            
            # Check for conflicts if timing info exists
            conflicts = self._check_for_conflicts(user_id, intent)
            if conflicts:
                return False, "conflict", None, conflicts
            
            # Create the task
            task_id = self._create_task_in_db(user_id, intent)
            return True, "Task created successfully", task_id, None
            
        except Exception as e:
            return False, f"Error creating task: {str(e)}", None, None
    
    def _is_valid_task_intent(self, intent: Dict) -> bool:
        """Check if intent is valid for task creation."""
        return intent.get("is_task_request") and intent.get("confidence", 0) >= 0.6
    
    def _check_for_conflicts(self, user_id: int, intent: Dict) -> List[Dict]:
        """Check for scheduling conflicts."""
        due_date = intent.get("due_date")
        due_time = intent.get("due_time")
        
        if due_date and due_time:
            return self.db_manager.check_schedule_conflicts(user_id, due_date, due_time)
        return []
    
    def _create_task_in_db(self, user_id: int, intent: Dict) -> int:
        """Create task in database from intent."""
        return self.db_manager.create_task(
            user_id=user_id,
            title=intent.get("task_title"),
            description=intent.get("description"),
            due_date=intent.get("due_date"),
            due_time=intent.get("due_time"),
            reminder_date=intent.get("reminder_date"),
            reminder_time=intent.get("reminder_time")
        )
    
    def update_task_from_conversation(self, user_id: int, message: str, context: str = "", 
                                     recent_task_hint: Dict = None) -> Tuple[bool, str, Optional[Dict]]:
        """
        Update existing task based on conversation.
        Returns (success, message, updated_task)
        """
        try:
            update_intent = self._parse_update_intent(user_id, message, context, recent_task_hint)
            
            if not update_intent.get("is_update_request"):
                return False, "Not an update request", None
            
            # Find the task to update
            matching_task = self._find_task_to_update(user_id, update_intent, recent_task_hint)
            if not matching_task:
                return False, "Could not find the task to update", None
            
            # Calculate and apply updates
            self._apply_task_updates(matching_task, update_intent)
            
            # Return updated task info
            updated_task = matching_task.copy()
            if update_intent.get("new_reminder_date"):
                updated_task['reminder_date'] = update_intent["new_reminder_date"]
            if update_intent.get("new_reminder_time"):
                updated_task['reminder_time'] = update_intent["new_reminder_time"]
            
            return True, f"Updated {matching_task['title']}", updated_task
            
        except Exception as e:
            return False, f"Error: {str(e)}", None
    
    def _parse_update_intent(self, user_id: int, message: str, context: str, 
                            recent_task_hint: Dict = None) -> Dict:
        """Parse update intent from user message."""
        recent_task_info = ""
        if recent_task_hint:
            recent_task_info = f"""
            Recent task just created:
            - Title: {recent_task_hint.get('title')}
            - Due: {recent_task_hint.get('due_date')} at {recent_task_hint.get('due_time')}
            """
        
        system_prompt = TaskPrompts.build_task_update_prompt(context, recent_task_info)
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User message: {message}")
        ])
        
        content = clean_json_response(response.content)
        return json.loads(content)
    
    def _find_task_to_update(self, user_id: int, update_intent: Dict, 
                            recent_task_hint: Dict = None) -> Optional[Dict]:
        """Find the task to update based on intent."""
        tasks = self.db_manager.get_user_tasks(user_id)
        task_identifier = update_intent.get("task_identifier", "").lower()
        
        # Priority 1: Explicit task identifier
        if task_identifier:
            for task in tasks:
                if (task_identifier in task['title'].lower() or 
                    (task.get('description') and task_identifier in task.get('description', '').lower())):
                    return task
        
        # Priority 2: Recent task hint
        if recent_task_hint:
            return recent_task_hint
        
        return None
    
    def _apply_task_updates(self, task: Dict, update_intent: Dict):
        """Apply updates to the task."""
        # Calculate reminder time if offset is provided
        reminder_date = update_intent.get("new_reminder_date")
        reminder_time = update_intent.get("new_reminder_time")
        reminder_offset = update_intent.get("reminder_offset_minutes")
        
        if reminder_offset and task.get('due_date') and task.get('due_time'):
            due_datetime = datetime.strptime(
                f"{task['due_date']} {task['due_time']}", 
                "%Y-%m-%d %H:%M"
            )
            reminder_datetime = due_datetime - timedelta(minutes=reminder_offset)
            reminder_date = reminder_datetime.strftime("%Y-%m-%d")
            reminder_time = reminder_datetime.strftime("%H:%M")
        
        # Update the task in database
        self.db_manager.update_task(
            task_id=task['id'],
            due_date=update_intent.get("new_due_date"),
            due_time=update_intent.get("new_due_time"),
            reminder_date=reminder_date,
            reminder_time=reminder_time
        )
    
    def delete_task(self, task_id: int) -> Tuple[bool, str]:
        """Delete a task."""
        try:
            self.db_manager.delete_task(task_id)
            return True, "Task deleted successfully"
        except Exception as e:
            return False, f"Error deleting task: {str(e)}"
    
    def get_task_summary(self, user_id: int) -> str:
        """Get a formatted summary of user's tasks."""
        try:
            tasks = self.db_manager.get_user_tasks(user_id)
            
            if not tasks:
                return "You have no tasks scheduled."
            
            # Count tasks by status
            pending_tasks = [t for t in tasks if t.get('status') == 'pending']
            completed_tasks = [t for t in tasks if t.get('status') == 'completed']
            
            summary_parts = []
            summary_parts.append(f"**Your Tasks ({len(tasks)} total - {len(pending_tasks)} pending, {len(completed_tasks)} completed):**")
            
            # Show pending tasks first
            if pending_tasks:
                summary_parts.append("\\n**Pending:**")
                for task in pending_tasks[:10]:  # Show up to 10
                    task_str = f"• {task['title']}"
                    if task['due_date'] and task['due_time']:
                        task_str += f" - Due: {task['due_date']} at {task['due_time']}"
                    if task['reminder_date'] and task['reminder_time']:
                        task_str += f" (Reminder: {task['reminder_date']} at {task['reminder_time']})"
                    summary_parts.append(task_str)
            
            return "\\n".join(summary_parts)
            
        except Exception as e:
            return "Unable to retrieve task summary."