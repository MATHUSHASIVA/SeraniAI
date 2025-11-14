"""
Response formatting and message generation utilities.
Handles formatting of task confirmations, query results, and user-facing messages.
"""

from datetime import datetime
from typing import Dict, List


class ResponseFormatter:
    """Formats responses for various task operations and queries."""
    
    @staticmethod
    def format_task_confirmation(task_intent: Dict) -> str:
        """Format task creation confirmation message."""
        task_title = task_intent.get("task_title")
        due_date = task_intent.get("due_date")
        due_time = task_intent.get("due_time")
        reminder_date = task_intent.get("reminder_date")
        reminder_time = task_intent.get("reminder_time")
        
        confirmation = f"Got it! I've added your {task_title.lower()} "
        
        if due_date and due_time:
            date_obj = datetime.strptime(due_date, "%Y-%m-%d")
            day_name = date_obj.strftime("%A")
            time_12h = datetime.strptime(due_time, "%H:%M").strftime("%I:%M %p").lstrip("0")
            confirmation += f"for {day_name} at {time_12h}. "
        
        if reminder_date and reminder_time:
            confirmation += "Reminder set! "
            
        confirmation += "You're all set! ✅"
        return confirmation
    
    @staticmethod
    def format_conflict_message(conflict_task: Dict, username: str) -> str:
        """Format conflict resolution message."""
        return f"By the way, {username} — looks like you've got {conflict_task['title']} scheduled at {conflict_task['due_time']}. Want me to handle that overlap?"
    
    @staticmethod
    def format_empty_task_response(time_frame: str, username: str) -> str:
        """Format response when no tasks are found."""
        responses = {
            "today": f"You're all clear for today, {username}! No tasks scheduled 😊",
            "tomorrow": f"Nothing on your schedule for tomorrow, {username}! 📅",
        }
        return responses.get(time_frame, f"No tasks found for that time frame, {username}! 👍")
    
    @staticmethod
    def build_task_summary(tasks: List[Dict], today: datetime.date) -> str:
        """Build a formatted summary of tasks."""
        task_list = []
        for task in tasks[:10]:  # Limit to 10 tasks
            task_str = f"• {task['title']}"
            if task.get('due_date') and task.get('due_time'):
                due_date = datetime.strptime(task['due_date'], '%Y-%m-%d').date()
                if due_date == today:
                    task_str += f" at {task['due_time']}"
                else:
                    day_name = due_date.strftime('%A')
                    task_str += f" on {day_name} at {task['due_time']}"
            if task.get('reminder_date') and task.get('reminder_time'):
                task_str += " (Reminder set)"
            task_list.append(task_str)
        
        return "\n".join(task_list)
    
    @staticmethod
    def format_reschedule_confirmation(old_task_title: str, new_task_title: str, 
                                      old_new_time: str, new_orig_time: str) -> str:
        """Format rescheduling confirmation message."""
        old_new_time_obj = datetime.strptime(old_new_time, "%H:%M")
        old_new_time_12h = old_new_time_obj.strftime("%I:%M %p").lstrip("0")
        new_orig_time_obj = datetime.strptime(new_orig_time, "%H:%M")
        new_orig_time_12h = new_orig_time_obj.strftime("%I:%M %p").lstrip("0")
        return f"Perfect! ✅ I've rescheduled {old_task_title} to {old_new_time_12h} and kept {new_task_title} at {new_orig_time_12h}."
    
    @staticmethod
    def format_task_list_prompt(task_names: List[str]) -> str:
        """Format a list of task names for user selection."""
        return f"I have these tasks: {', '.join(task_names)}. Which one would you like to update? 🤔"
    
    @staticmethod
    def handle_task_creation_failure(task_intent: Dict, conflicts: List[Dict], 
                                    message: str, username: str, conversation_state: Dict) -> str:
        """
        Handle task creation failure, including conflicts.
        Updates conversation state if needed.
        """
        if conflicts:
            conflict_task = conflicts[0]
            conversation_state["pending_task"] = task_intent
            conversation_state["awaiting_clarification"] = True
            conversation_state["clarification_type"] = "conflict_resolution"
            conversation_state["conflicting_task"] = conflict_task
            conversation_state["original_message"] = message
            conversation_state["initial_message_causing_clarification"] = message
            return ResponseFormatter.format_conflict_message(conflict_task, username)
        else:
            return "Hmm, had trouble with that. Could you try again? 🤔"
