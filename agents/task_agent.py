import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import dateparser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken

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
        
        # Token counter for debugging
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Note: Tools functionality simplified for compatibility
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return 0
    
    def _print_llm_input(self, call_name: str, system_prompt: str, user_message: str):
        """Print LLM input details for debugging (production mode: minimal output)."""
        # Production mode: Only show call name and token count
        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(user_message)
        total_tokens = system_tokens + user_tokens
        
        print(f"📋 {call_name} | Tokens: {total_tokens} (system: {system_tokens}, user: {user_tokens})")
        
        # Uncomment below for detailed debugging:
        # print("\n" + "="*80)
        # print(f"📋 LLM CALL: {call_name}")
        # print("="*80)
        # print(f"\n📝 SYSTEM PROMPT ({len(system_prompt)} chars):")
        # print("-" * 80)
        # print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
        # print(f"\n💬 USER MESSAGE ({len(user_message)} chars):")
        # print("-" * 80)
        # print(user_message)
        # print("="*80 + "\n")
    
    def parse_task_intent(self, user_message: str, user_id: int, context: str = "") -> Dict:
        """
        Parse user message to extract task creation intent and details.
        """
        system_prompt = f"""
        You are a task parsing assistant. Analyze the user's message to extract task details.
        
        Context: {context}
        Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Extract the following information:
        1. Task title (brief, e.g., "Meeting", "Dentist Appointment", "Cricket Practice")
        2. Description (details like "work meeting", "online", "with team", "business", etc.)
        3. Due date (YYYY-MM-DD format)
        4. Due time (HH:MM format, 24-hour)
        5. Reminder date (YYYY-MM-DD format)
        6. Reminder time (HH:MM format, 24-hour)
        
        Return a JSON object with:
        {{
            "is_task_request": boolean,
            "task_title": string or null,
            "description": string or null,
            "due_date": "YYYY-MM-DD" or null,
            "due_time": "HH:MM" or null,
            "reminder_date": "YYYY-MM-DD" or null,
            "reminder_time": "HH:MM" or null,
            "confidence": float (0-1)
        }}
        
        Examples:
        - "meeting tomorrow at 2 PM" 
          -> task_title: "Meeting", due_date: tomorrow, due_time: "14:00"
        - "work related, online meeting"
          -> description: "Work meeting, online"
        """
        
        user_content = f"User message: {user_message}"
        
        # Print LLM input for debugging
        self._print_llm_input("Task Intent Parsing", system_prompt, user_content)
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
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
            print(f"DEBUG TaskAgent: Intent received: {intent}")
            print(f"DEBUG TaskAgent: is_task_request={intent.get('is_task_request')}, confidence={intent.get('confidence')}")
            
            if not intent.get("is_task_request") or intent.get("confidence", 0) < 0.6:
                print(f"DEBUG TaskAgent: REJECTED - Not task request or low confidence")
                return False, "Not recognized as a task creation request", None, None
            
            title = intent.get("task_title")
            description = intent.get("description")
            due_date = intent.get("due_date")
            due_time = intent.get("due_time")
            reminder_date = intent.get("reminder_date")
            reminder_time = intent.get("reminder_time")
            
            print(f"DEBUG TaskAgent: Extracted - title={title}, due_date={due_date}, due_time={due_time}")
            print(f"DEBUG TaskAgent: Reminder - date={reminder_date}, time={reminder_time}")
            
            if not title:
                print(f"DEBUG TaskAgent: REJECTED - No title")
                return False, "Could not extract task title", None, None
            
            # Check for conflicts if we have timing info
            conflicts = []
            if due_date and due_time:
                conflicts = self.db_manager.check_schedule_conflicts(
                    user_id, due_date, due_time
                )
                print(f"DEBUG TaskAgent: Conflicts found: {len(conflicts)}")
            
            # If conflicts exist, return them for handling
            if conflicts:
                conflict_msg = self._format_conflicts(conflicts)
                return False, f"conflict", None, conflicts
            
            # Create the task
            print(f"DEBUG TaskAgent: Creating task in database...")
            task_id = self.db_manager.create_task(
                user_id=user_id,
                title=title,
                description=description,
                due_date=due_date,
                due_time=due_time,
                reminder_date=reminder_date,
                reminder_time=reminder_time
            )
            
            print(f"DEBUG TaskAgent: Task created successfully with ID: {task_id}")
            return True, "Task created successfully", task_id, None
            
        except Exception as e:
            print(f"Error creating task: {e}")
            return False, f"Error creating task: {str(e)}", None, None
    
    def update_task_from_conversation(self, user_id: int, message: str, context: str = "", recent_task_hint: Dict = None) -> Tuple[bool, str, Optional[Dict]]:
        """
        Update existing task based on conversation.
        Returns (success, message, updated_task)
        """
        try:
            # Parse update intent
            recent_task_info = ""
            if recent_task_hint:
                recent_task_info = f"""
                Recent task just created:
                - Title: {recent_task_hint.get('title')}
                - Due: {recent_task_hint.get('due_date')} at {recent_task_hint.get('due_time')}
                """
            
            system_prompt = f"""
            Analyze if the user wants to update an existing task.
            
            Context: {context}
            {recent_task_info}
            Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Look for:
            - Adding/changing reminder: "remind me X minutes before", "set a reminder", "change the reminder to"
            - Rescheduling: "move to", "shift to", "change to", "reschedule"
            - Which task to update (by title or description match)
            - New time/date
            - Reminder time (parse "30 minutes before" as relative to due time)
            
            IMPORTANT:
            - If the user says "the reminder" or "the appointment" without specifying which one, 
              they are likely referring to the most recently discussed task.
            - If a recent task is provided above, prefer it as the target unless the user explicitly names a different task.
            - Look for task identifiers in context like "dentist", "meeting", "appointment", etc.
            
            Return JSON:
            {{
                "is_update_request": boolean,
                "task_identifier": string or null (title/description to match, or null to use recent task),
                "new_due_date": "YYYY-MM-DD" or null,
                "new_due_time": "HH:MM" or null,
                "new_reminder_date": "YYYY-MM-DD" or null,
                "new_reminder_time": "HH:MM" or null,
                "reminder_offset_minutes": number or null (for "X minutes before")
            }}
            
            Examples:
            - "shift cricket practice to 3:30 PM" -> task_identifier: "cricket practice", new_due_time: "15:30"
            - "remind me 30 minutes before appointment" -> task_identifier: "appointment", reminder_offset_minutes: 30
            - "change the reminder to November 13 at 10 AM" (with recent task context) -> task_identifier: null, new_reminder_date: "2025-11-13", new_reminder_time: "10:00"
            """
            
            user_content = f"User message: {message}"
            
            # Print LLM input for debugging
            self._print_llm_input("Task Update Parsing", system_prompt, user_content)
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            update_intent = json.loads(content)
            
            if not update_intent.get("is_update_request"):
                return False, "Not an update request", None
            
            # Find the task
            matching_task = None
            tasks = self.db_manager.get_user_tasks(user_id)
            task_identifier = update_intent.get("task_identifier", "").lower()
            
            # Priority 1: If user explicitly mentioned a task by name, find it
            if task_identifier:
                print(f"DEBUG TaskAgent: Looking for task with identifier: '{task_identifier}'")
                # Search through all tasks for explicit mention
                for task in tasks:
                    if (task_identifier in task['title'].lower() or 
                        (task.get('description') and task_identifier in task.get('description', '').lower())):
                        matching_task = task
                        print(f"DEBUG TaskAgent: Found matching task by identifier: {task['title']}")
                        break
            
            # Priority 2: If no explicit match and we have a recent task hint, use it
            if not matching_task and recent_task_hint:
                matching_task = recent_task_hint
                print(f"DEBUG TaskAgent: Using recent task hint: {recent_task_hint['title']}")
            
            if not matching_task:
                return False, "Could not find the task to update", None
            
            # Calculate reminder time if offset is provided
            reminder_date = update_intent.get("new_reminder_date")
            reminder_time = update_intent.get("new_reminder_time")
            reminder_offset = update_intent.get("reminder_offset_minutes")
            
            if reminder_offset and matching_task.get('due_date') and matching_task.get('due_time'):
                due_datetime = datetime.strptime(
                    f"{matching_task['due_date']} {matching_task['due_time']}", 
                    "%Y-%m-%d %H:%M"
                )
                reminder_datetime = due_datetime - timedelta(minutes=reminder_offset)
                reminder_date = reminder_datetime.strftime("%Y-%m-%d")
                reminder_time = reminder_datetime.strftime("%H:%M")
                print(f"DEBUG: Calculated reminder: {reminder_date} {reminder_time} ({reminder_offset} mins before)")
            
            # Update the task
            self.db_manager.update_task(
                task_id=matching_task['id'],
                due_date=update_intent.get("new_due_date"),
                due_time=update_intent.get("new_due_time"),
                reminder_date=reminder_date,
                reminder_time=reminder_time
            )
            
            # Fetch updated task
            updated_task = matching_task.copy()
            if reminder_date:
                updated_task['reminder_date'] = reminder_date
            if reminder_time:
                updated_task['reminder_time'] = reminder_time
            
            return True, f"Updated {matching_task['title']}", updated_task
            
        except Exception as e:
            print(f"Error updating task: {e}")
            return False, f"Error: {str(e)}", None
    
    def delete_task(self, task_id: int) -> Tuple[bool, str]:
        """Delete a task."""
        try:
            self.db_manager.delete_task(task_id)
            return True, "Task deleted successfully"
        except Exception as e:
            print(f"Error deleting task: {e}")
            return False, f"Error deleting task: {str(e)}"
    
    def _format_conflicts(self, conflicts: List[Dict]) -> str:
        """Format conflict information for user display."""
        if not conflicts:
            return "No conflicts found."
        
        conflict_strs = []
        for conflict in conflicts:
            conflict_strs.append(
                f"'{conflict['title']}' on {conflict['due_date']} at {conflict['due_time']}"
            )
        
        return "\\n".join(conflict_strs)
    
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
            print(f"Error getting task summary: {e}")
            return "Unable to retrieve task summary."