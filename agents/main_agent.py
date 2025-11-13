from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
import traceback
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .base_agent import BaseAgent
from .context_agent import ContextAgent
from .task_agent import TaskAgent

class MainAgent(BaseAgent):
    """
    Main conversational agent that routes messages to specialized agents
    and maintains conversation flow.
    """
    
    def __init__(self, openai_api_key: str, db_manager, context_agent: ContextAgent, task_agent: TaskAgent):
        super().__init__()  # Initialize BaseAgent
        
        self.openai_api_key = openai_api_key
        self.db_manager = db_manager
        self.context_agent = context_agent
        self.task_agent = task_agent
        
        # Initialize the main LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.7  # Slightly higher for more natural conversation
        )
        
        # System persona
        self.system_persona = """
        You are Serani, a smart personal assistant. Be:
        
        **Tone:** Casual, warm, efficient. Use emojis naturally (👍😄💪✅🎯).
        
        **Style:**
        - Short responses (1-2 sentences max)
        - Conversational, not robotic
        - Proactive: spot conflicts, suggest improvements
        - Remember context from earlier in chat
        
        **Communication:**
        - Confirm tasks: "Got it!" or "All set ✅"
        - Ask only essential questions
        - End with encouraging phrases occasionally
        - Use user's name naturally
        - Use the user's name when appropriate
        - End messages with encouraging or supportive tone
        
        **Current Focus:**
        As specified by the user, you are currently focusing on task management only.
        You do not handle reminders or scheduling at this time.
        """
        
        # State tracking
        self.conversation_state = {
            "awaiting_clarification": False,
            "pending_task": None,
            "last_intent": None
        }
    
    def process_message(self, user_id: int, message: str, username: str = "User") -> str:
        """
        Main message processing pipeline.
        """
        try:
            # Get conversation context from ChromaDB
            try:
                context_prompt = self.context_agent.build_context_prompt(
                    user_id, message, []
                )
            except Exception as ctx_error:
                print(f"Error building context: {ctx_error}")
                context_prompt = ""
            
            # Check if we're awaiting a clarification response FIRST
            # But NOT if this is the initial message that caused the clarification
            if self.conversation_state.get("awaiting_clarification") and not self.conversation_state.get("initial_message_causing_clarification") == message:
                print("DEBUG: Handling clarification response")
                return self._handle_clarification_response(user_id, message, context_prompt, username)
            
            # Determine intent and route to appropriate handler
            try:
                intent = self._analyze_intent(message, context_prompt, username)
            except Exception as intent_error:
                print(f"Error analyzing intent: {intent_error}")
                import traceback
                traceback.print_exc()
                raise
            
            # Process based on intent
            try:
                response = self._handle_intent(user_id, message, intent, context_prompt, username)
            except Exception as handle_error:
                print(f"Error handling intent: {handle_error}")
                import traceback
                traceback.print_exc()
                raise
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            return f"I'm sorry, I encountered an error processing your message: {str(e)}"
    
    def _analyze_intent(self, message: str, context: str, username: str) -> Dict:
        """
        Analyze user message to determine intent and required actions.
        """
        system_prompt = f"""
        {self.system_persona}
        
        Analyze the user's message to determine their intent. Consider the conversation context.
        
        Context: {context}
        
        Intent types:
        1. "task_creation" - NEW meetings, appointments with full details (what, when)
        2. "task_query" - "What's my schedule?", "Show tasks"
        3. "task_update" - Modify/add reminder/reschedule/cancel existing task
        4. "general_chat" - Greetings, questions, casual chat
        5. "clarification_response" - Answering Serani's question
        
        IMPORTANT Rules for task_creation:
        - NEW appointment/meeting/call WITH specific time/date AND task name → task_creation
        - "I have a [task name] on [date] at [time]" → task_creation (even with "remind me")
        - If user provides BOTH task title AND scheduling time → task_creation
        
        IMPORTANT Rules for task_update:
        - "Remind me before [existing task name]" (no new task details) → task_update
        - "Move/shift/reschedule/change [existing task]" → task_update
        - "Set a reminder for my [existing task]" → task_update
        - Refers to existing task WITHOUT providing new task details → task_update
        
        Key Distinction:
        - Has NEW task name + date/time = task_creation (even if mentions reminder)
        - Only refers to existing task = task_update
        
        Context Analysis:
        - If context shows a task was JUST created, and user mentions "reminder" or "the appointment" 
          WITHOUT providing a new task name → task_update
        - If user provides a NEW task name and time, even if they also mention reminder → task_creation
        
        Return JSON:
        {{
            "intent": string,
            "confidence": float,
            "requires_task_agent": boolean,
            "needs_clarification": boolean,
            "clarification_type": string or null,
            "emotional_context": string or null
        }}
        """
        
        user_content = f"User ({username}): {message}"
        
        # Print LLM input for debugging
        self._print_llm_input("Intent Analysis", system_prompt, user_content)
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            
            content = response.content.strip()
            
            # Clean up JSON formatting
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse JSON
            intent = json.loads(content)
            return intent
            
        except json.JSONDecodeError as e:
            print(f"Error analyzing intent (JSON parsing failed): {e}")
            print(f"Raw response: {response.content[:200]}")
            
            # Fallback: Try to detect intent from keywords
            msg_lower = message.lower()
            if any(word in msg_lower for word in ["show", "what", "list", "display", "my tasks", "schedule", "what's"]):
                return {
                    "intent": "task_query",
                    "confidence": 0.8,
                    "requires_task_agent": True,
                    "needs_clarification": False,
                    "clarification_type": None,
                    "emotional_context": None
                }
            
            # Default fallback
            return {
                "intent": "general_chat",
                "confidence": 0.5,
                "requires_task_agent": False,
                "needs_clarification": False,
                "clarification_type": None,
                "emotional_context": None
            }
        except Exception as e:
            print(f"Error analyzing intent: {e}")
            return {
                "intent": "general_chat",
                "confidence": 0.5,
                "requires_task_agent": False,
                "needs_clarification": False,
                "clarification_type": None,
                "emotional_context": None
            }
    
    def _handle_intent(self, user_id: int, message: str, intent: Dict, 
                      context: str, username: str) -> str:
        """
        Route message to appropriate handler based on intent.
        """
        intent_type = intent.get("intent", "general_chat")
        
        # Debug: Log intent detection
        print(f"DEBUG: Detected intent '{intent_type}' with confidence {intent.get('confidence', 0)}")
        print(f"DEBUG: Full intent: {intent}")
        
        if intent_type == "task_creation":
            return self._handle_task_creation(user_id, message, context, username)
        
        elif intent_type == "task_query":
            return self._handle_task_query(user_id, message, context, username)
        
        elif intent_type == "task_update":
            return self._handle_task_update(user_id, message, context, username)
        
        elif intent_type == "clarification_response":
            return self._handle_clarification_response(user_id, message, context, username)
        
        else:  # general_chat
            return self._handle_general_conversation(user_id, message, intent, context, username)
    
    def _handle_task_creation(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle task creation requests."""
        try:
            # Parse task intent
            task_intent = self.task_agent.parse_task_intent(message, user_id, context)
            
            # Check if task title exists
            if not task_intent.get("task_title"):
                return f"What task would you like me to track? 🤔"
            
            task_title = task_intent.get("task_title")
            description = task_intent.get("description")
            due_date = task_intent.get("due_date")
            due_time = task_intent.get("due_time")
            reminder_date = task_intent.get("reminder_date")
            reminder_time = task_intent.get("reminder_time")
            
            # Check what information is missing
            missing_info = []
            if not due_date or not due_time:
                missing_info.append("due date and time")
            # Note: Reminder is optional, so we don't require it
            
            # If ESSENTIAL information is missing, ask for clarification
            if missing_info:
                self.conversation_state["awaiting_clarification"] = True
                self.conversation_state["pending_task"] = task_intent
                self.conversation_state["original_message"] = message  # Store original message for context
                
                # Determine what to ask for
                if "due date and time" in missing_info:
                    self.conversation_state["clarification_type"] = "due_datetime"
                else:
                    self.conversation_state["clarification_type"] = "reminder_datetime"
                
                # Generate natural clarification question using LLM
                # Build context about what we already know
                known_info = []
                if due_date and due_time:
                    known_info.append(f"scheduled for {due_date} at {due_time}")
                
                clarification_prompt = f"""
                {self.system_persona}
                
                The user wants to create a task: "{task_title}"
                {f"Description: {description}" if description else ""}
                {f"Already scheduled: {', '.join(known_info)}" if known_info else ""}
                
                Missing information: {missing_info[0]}
                
                Ask the user naturally for the missing information ONLY. Keep it SHORT (1 sentence).
                Be conversational and friendly with emoji 😊
                
                Examples based on what's missing:
                - If missing "due date and time": "Got it! When would you like to do this?"
                - If missing "reminder date and time" (but have due date/time): "Perfect! Should I set a reminder for you?"
                - If missing "reminder date and time" (but have due date/time): "All set! Do you need a reminder before your appointment?"
                """
                
                user_content = f"Generate question for: {missing_info[0]}"
                
                # Print LLM input for debugging
                self._print_llm_input("Clarification Question", clarification_prompt, user_content)
                
                response = self.llm.invoke([
                    SystemMessage(content=clarification_prompt),
                    HumanMessage(content=user_content)
                ])
                
                return response.content.strip()
            
            # All info present, create task
            success, message_result, task_id, conflicts = self.task_agent.create_task_from_intent(
                user_id, task_intent, context
            )
            
            if success:
                # Store task as conversation summary in ChromaDB
                try:
                    task_summary = f"Created task '{task_title}'"
                    if description:
                        task_summary += f": {description}"
                    if due_date and due_time:
                        task_summary += f" scheduled for {due_date} at {due_time}"
                    if reminder_date and reminder_time:
                        task_summary += f" with reminder at {reminder_time}"
                    
                    # Store using the same method as conversations (simple metadata)
                    self.context_agent.store_conversation_summary(
                        user_id=user_id,
                        summary=task_summary,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        conversation_metadata={}
                    )
                except Exception as storage_err:
                    print(f"⚠️ Failed to store task in ChromaDB: {storage_err}")
                
                # Create a specific confirmation message
                confirmation = f"Got it! I've added your {task_title.lower()} "
                if due_date and due_time:
                    # Format the date nicely
                    date_obj = datetime.strptime(due_date, "%Y-%m-%d")
                    day_name = date_obj.strftime("%A")  # e.g., "Monday"
                    time_12h = datetime.strptime(due_time, "%H:%M").strftime("%I:%M %p").lstrip("0")
                    confirmation += f"for {day_name} at {time_12h}. "
                
                if reminder_date and reminder_time:
                    confirmation += "Reminder set! "
                    
                confirmation += "You're all set! ✅"
                return confirmation
            else:
                # Handle conflicts
                if conflicts:
                    conflict_task = conflicts[0]
                    self.conversation_state["pending_task"] = task_intent
                    self.conversation_state["awaiting_clarification"] = True
                    self.conversation_state["clarification_type"] = "conflict_resolution"
                    self.conversation_state["conflicting_task"] = conflict_task
                    self.conversation_state["original_message"] = message  # Store original message for context
                    self.conversation_state["initial_message_causing_clarification"] = message  # Keep for compatibility
                    return f"By the way, {username} — looks like you've got {conflict_task['title']} scheduled at {conflict_task['due_time']}. Want me to handle that overlap?"
                else:
                    return f"Hmm, had trouble with that. {message_result} 🤔"
        
        except Exception as e:
            print(f"Error handling task creation: {e}")
            import traceback
            traceback.print_exc()
            return "I'm having trouble creating that task. Could you tell me more?"
    
    def _handle_clarification_response(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle responses to clarifying questions."""
        try:
            if not self.conversation_state.get("awaiting_clarification"):
                return self._handle_general_conversation(user_id, message, {}, context, username)
            
            pending_task = self.conversation_state.get("pending_task", {})
            clarification_type = self.conversation_state.get("clarification_type")
            
            # Combine original message with clarification for better context
            original_message = self.conversation_state.get("original_message", "")
            combined_message = f"{original_message}. {message}" if original_message else message
            
            # Parse the clarification response with full context
            timing_intent = self.task_agent.parse_task_intent(combined_message, user_id, context)
            
            if clarification_type == "description":
                # User provided task type/description
                pending_task["description"] = message
                pending_task["is_task_request"] = True
                pending_task["confidence"] = 1.0
                
                # Ask about timing next
                self.conversation_state["clarification_type"] = "due_datetime"
                
                # Check if description mentions "online"
                if "online" in message.lower():
                    return f"Nice! Online ones are always convenient 😄\nI'll set a reminder for you. How long do you expect it to last?"
                else:
                    return f"When do you plan to start working on it — today or later this week?"
            
            elif clarification_type == "due_datetime":
                # User is providing timing
                if timing_intent.get("due_date") and timing_intent.get("due_time"):
                    pending_task["due_date"] = timing_intent["due_date"]
                    pending_task["due_time"] = timing_intent["due_time"]
                    
                    # Now ask for reminder
                    self.conversation_state["clarification_type"] = "reminder_datetime"
                    return f"Got it! I've added it to your calendar. Do you need a reminder before you leave?"
                else:
                    return "Hmm, I didn't catch the date and time. Could you tell me when? 🤔"
            
            elif clarification_type == "reminder_datetime":
                # User is responding about reminder
                if "no" in message.lower() and "yes" not in message.lower():
                    # Skip reminder
                    pending_task["reminder_date"] = None
                    pending_task["reminder_time"] = None
                else:
                    # Parse relative time like "30 minutes before" or "before 30 minutes"
                    # Check for both word order patterns
                    before_match = re.search(r'(\d+)\s*(minute|hour|min|hr)s?\s*before', message.lower()) or \
                                  re.search(r'before\s+(\d+)\s*(minute|hour|min|hr)s?', message.lower())
                    
                    if before_match and pending_task.get("due_date") and pending_task.get("due_time"):
                        amount = int(before_match.group(1))
                        unit = before_match.group(2)
                        
                        # Calculate reminder time
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
                        print(f"DEBUG: Set reminder to {pending_task['reminder_date']} at {pending_task['reminder_time']}")
                    else:
                        # Try to parse as absolute time
                        if timing_intent.get("due_date") and timing_intent.get("due_time"):
                            pending_task["reminder_date"] = timing_intent["due_date"]
                            pending_task["reminder_time"] = timing_intent["due_time"]
                            print(f"DEBUG: Set reminder to {pending_task['reminder_date']} at {pending_task['reminder_time']}")
                        elif timing_intent.get("reminder_date") and timing_intent.get("reminder_time"):
                            pending_task["reminder_date"] = timing_intent["reminder_date"]
                            pending_task["reminder_time"] = timing_intent["reminder_time"]
                            print(f"DEBUG: Set reminder to {pending_task['reminder_date']} at {pending_task['reminder_time']}")
                        else:
                            # Check if user just said "yes" - use default 30 minutes before
                            if message.strip().lower() in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
                                if pending_task.get("due_date") and pending_task.get("due_time"):
                                    due_datetime = datetime.strptime(
                                        f"{pending_task['due_date']} {pending_task['due_time']}", 
                                        "%Y-%m-%d %H:%M"
                                    )
                                    reminder_datetime = due_datetime - timedelta(minutes=30)
                                    pending_task["reminder_date"] = reminder_datetime.strftime("%Y-%m-%d")
                                    pending_task["reminder_time"] = reminder_datetime.strftime("%H:%M")
                                    print(f"DEBUG: Using default 30 min reminder: {pending_task['reminder_date']} at {pending_task['reminder_time']}")
                                else:
                                    return "When should I remind you? (e.g., 30 minutes before, or a specific time)"
                            else:
                                return "When should I remind you? (e.g., 30 minutes before, or a specific time)"
            
            elif clarification_type == "conflict_resolution":
                # User is responding to conflict
                conflicting_task = self.conversation_state.get("conflicting_task")

                msg_lower = message.lower()
                # Prefer rescheduling the OLD (conflicting) task if user uses reschedule keywords
                # or explicitly mentions the conflicting task title/name.
                conflict_title = (conflicting_task.get('title') or '').lower() if conflicting_task else ''

                wants_to_reschedule_old = any(word in msg_lower for word in ["move", "shift", "reschedule", "change"]) or (conflict_title and conflict_title in msg_lower)

                if wants_to_reschedule_old:
                    # Parse the new time for the OLD conflicting task
                    new_timing = self.task_agent.parse_task_intent(message, user_id, context)

                    if new_timing.get("due_date") or new_timing.get("due_time"):
                        # Update the conflicting task with new time
                        old_task_id = conflicting_task.get("id") if conflicting_task else None
                        update_params = {}

                        if new_timing.get("due_date"):
                            update_params["due_date"] = new_timing["due_date"]
                        if new_timing.get("due_time"):
                            update_params["due_time"] = new_timing["due_time"]

                        if old_task_id:
                            # Update the old task in database
                            self.db_manager.update_task(old_task_id, **update_params)

                            # Now create the NEW pending task at its ORIGINAL time
                            pending_task["is_task_request"] = True
                            pending_task["confidence"] = 1.0

                            success2, msg2, task_id2, _ = self.task_agent.create_task_from_intent(
                                user_id, pending_task, context
                            )

                            # Clear state
                            self.conversation_state["awaiting_clarification"] = False
                            self.conversation_state["clarification_type"] = None
                            self.conversation_state["original_message"] = None
                            self.conversation_state["pending_task"] = None
                            self.conversation_state["conflicting_task"] = None

                            if success2:
                                old_task_title = conflicting_task.get("title")
                                new_task_title = pending_task.get("task_title")

                                # Format times nicely
                                new_time_obj = datetime.strptime(update_params.get("due_time", conflicting_task.get("due_time")), "%H:%M")
                                new_time_12h = new_time_obj.strftime("%I:%M %p").lstrip("0")

                                orig_time_obj = datetime.strptime(pending_task["due_time"], "%H:%M")
                                orig_time_12h = orig_time_obj.strftime("%I:%M %p").lstrip("0")

                                return f"Perfect! ✅ I've rescheduled {old_task_title} to {new_time_12h} and added {new_task_title} at {orig_time_12h}."
                            else:
                                return f"I rescheduled {conflicting_task.get('title')}, but had trouble creating the new task. {msg2}"
                        else:
                            return "I couldn't identify which existing task to reschedule. Could you name it?"
                    else:
                        return "I didn't catch the new time. Could you specify when to reschedule? (e.g., 7 AM)"

                # If user intends to schedule the NEW task at a different time, handle that next
                new_task_keywords = ["schedule", "then", "at", "evening", "morning", "afternoon"]
                if any(word in msg_lower for word in new_task_keywords):
                    # Parse new time for the PENDING task from the message
                    new_timing = self.task_agent.parse_task_intent(message, user_id, context)

                    if new_timing.get("due_date") and new_timing.get("due_time"):
                        # Update the pending task with new time
                        pending_task["due_date"] = new_timing["due_date"]
                        pending_task["due_time"] = new_timing["due_time"]

                        # Now try to create the task with new time
                        pending_task["is_task_request"] = True
                        pending_task["confidence"] = 1.0

                        success, result_message, task_id, new_conflicts = self.task_agent.create_task_from_intent(
                            user_id, pending_task, context
                        )

                        # Clear state
                        self.conversation_state["awaiting_clarification"] = False
                        self.conversation_state["clarification_type"] = None
                        self.conversation_state["original_message"] = None
                        self.conversation_state["pending_task"] = None
                        self.conversation_state["conflicting_task"] = None

                        if success:
                            task_title = pending_task.get("task_title")
                            # Format the date and time nicely
                            date_obj = datetime.strptime(pending_task["due_date"], "%Y-%m-%d")
                            day_name = date_obj.strftime("%A")
                            time_12h = datetime.strptime(pending_task["due_time"], "%H:%M").strftime("%I:%M %p").lstrip("0")
                            return f"Perfect! ✅ {task_title} scheduled for {day_name} at {time_12h}."
                        else:
                            return f"Hmm, had trouble with that. {result_message} 🤔"
                    else:
                        return "I didn't catch the new time. Could you specify when? (e.g., 6 PM)"

                # Default: ask for clarification
                return "Would you like to reschedule one of the tasks? Just let me know the new time."
            
            # Try to create the complete task
            print(f"DEBUG: Creating task with pending_task: {pending_task}")
            pending_task["is_task_request"] = True
            pending_task["confidence"] = 1.0
            
            success, result_message, task_id, conflicts = self.task_agent.create_task_from_intent(
                user_id, pending_task, context
            )
            
            print(f"DEBUG: Task creation result - success: {success}, message: {result_message}, task_id: {task_id}")
            
            # Clear clarification state
            self.conversation_state["awaiting_clarification"] = False
            self.conversation_state["clarification_type"] = None
            self.conversation_state["pending_task"] = None
            self.conversation_state["original_message"] = None
            self.conversation_state["initial_message_causing_clarification"] = None
            
            if success:
                if pending_task.get("reminder_date") and pending_task.get("reminder_time"):
                    # Format time nicely (12-hour format)
                    reminder_time_obj = datetime.strptime(pending_task['reminder_time'], "%H:%M")
                    reminder_time_12h = reminder_time_obj.strftime("%I:%M %p").lstrip("0")
                    return f"Perfect! I'll remind you at {reminder_time_12h} ✅"
                else:
                    return f"Perfect! Task added successfully ✅"
            else:
                return f"Hmm, had trouble with that. {result_message} 🤔"
            
        except Exception as e:
            print(f"Error handling clarification: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, I didn't quite get that. Could you clarify? 🤔"
    
    def _handle_task_query(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle queries about existing tasks."""
        try:
            # Get all tasks
            all_tasks = self.db_manager.get_user_tasks(user_id)
            
            # Parse the query to understand what time frame they're asking about
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            
            # Filter tasks based on query
            filtered_tasks = all_tasks
            time_frame = "all"
            
            message_lower = message.lower()
            if "today" in message_lower:
                filtered_tasks = [t for t in all_tasks if t.get('due_date') == today.strftime('%Y-%m-%d')]
                time_frame = "today"
            elif "tomorrow" in message_lower:
                filtered_tasks = [t for t in all_tasks if t.get('due_date') == tomorrow.strftime('%Y-%m-%d')]
                time_frame = "tomorrow"
            elif "this week" in message_lower or "week" in message_lower:
                week_end = today + timedelta(days=7)
                filtered_tasks = [t for t in all_tasks if t.get('due_date') and today.strftime('%Y-%m-%d') <= t.get('due_date') <= week_end.strftime('%Y-%m-%d')]
                time_frame = "this week"
            
            # Build task summary for the filtered tasks
            if not filtered_tasks:
                if time_frame == "today":
                    return f"You're all clear for today, {username}! No tasks scheduled 😊"
                elif time_frame == "tomorrow":
                    return f"Nothing on your schedule for tomorrow, {username}! 📅"
                else:
                    return f"No tasks found for that time frame, {username}! 👍"
            
            # Create task summary
            task_list = []
            for task in filtered_tasks[:10]:  # Limit to 10 tasks
                task_str = f"• {task['title']}"
                if task.get('due_date') and task.get('due_time'):
                    # Format date nicely
                    due_date = datetime.strptime(task['due_date'], '%Y-%m-%d').date()
                    if due_date == today:
                        task_str += f" at {task['due_time']}"
                    else:
                        day_name = due_date.strftime('%A')
                        task_str += f" on {day_name} at {task['due_time']}"
                if task.get('reminder_date') and task.get('reminder_time'):
                    task_str += f" (Reminder set)"
                task_list.append(task_str)
            
            task_summary = "\n".join(task_list)
            
            # Generate contextual response
            system_prompt = f"""
            {self.system_persona}
            
            {username} asked about their tasks for {time_frame}. 
            Current date: {today.strftime('%Y-%m-%d (%A)')}
            
            Tasks found:
            {task_summary}
            
            Give a quick, accurate summary with emoji! 
            If it's for TODAY, say "today". If it's for a future date, mention the day clearly.
            Keep it SHORT and friendly 😊
            """
            
            user_content = f"User query: {message}"
            
            # Print LLM input for debugging
            self._print_llm_input("Task Query Response", system_prompt, user_content)
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error handling task query: {e}")
            import traceback
            traceback.print_exc()
            return "Let me check your tasks for you..."
    
    def _handle_task_update(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle task modification requests."""
        try:
            # Get all user tasks
            tasks = self.db_manager.get_user_tasks(user_id)
            
            # Find the most recently mentioned or created task
            # Priority: 1) Recently updated (within 5 mins), 2) Recently created (within 5 mins), 3) Most recent overall
            recent_task = None
            now = datetime.now()
            
            if tasks:
                # Sort tasks by updated_at or created_at (most recent first)
                sorted_tasks = sorted(
                    tasks, 
                    key=lambda t: t.get('updated_at', t.get('created_at', '')), 
                    reverse=True
                )
                
                # Check for very recent activity (within 5 minutes)
                for task in sorted_tasks:
                    try:
                        # Check updated_at first, then created_at
                        time_str = task.get('updated_at') or task.get('created_at')
                        if time_str:
                            task_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                            if now - task_time < timedelta(minutes=5):
                                recent_task = task
                                print(f"DEBUG: Found recent task: {task['title']} (last activity: {time_str})")
                                break
                    except Exception as e:
                        print(f"DEBUG: Error parsing task time: {e}")
                        continue
                
                # If no recent task found, use the first task from sorted list
                if not recent_task and sorted_tasks:
                    recent_task = sorted_tasks[0]
                    print(f"DEBUG: Using most recent task overall: {recent_task['title']}")
            
            # Try to update the task, passing recent_task as a hint
            print(f"DEBUG: Attempting task update with message: {message}")
            success, update_message, updated_task = self.task_agent.update_task_from_conversation(
                user_id, message, context, recent_task_hint=recent_task
            )
            
            if success:
                # Generate appropriate response based on what was updated
                if 'reminder' in message.lower():
                    return f"Done! I've updated the reminder for your {updated_task['title'].lower()} ✅"
                else:
                    return f"All done ✅ I've updated your {updated_task['title'].lower()}."
            else:
                # Check if this might actually be a new task creation that was misclassified
                # Look for indicators like specific time/date and task name
                if any(indicator in message.lower() for indicator in ['i have', 'client call', 'meeting', 'appointment']) and \
                   any(time_word in message.lower() for time_word in ['at', 'on', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
                    print("DEBUG: This looks like a new task creation, not an update. Redirecting...")
                    return self._handle_task_creation(user_id, message, context, username)
                
                # If failed and we have context about tasks, help the user
                if tasks and len(tasks) > 1:
                    task_names = [t['title'] for t in tasks[:3]]
                    return f"I have these tasks: {', '.join(task_names)}. Which one would you like to update? 🤔"
                else:
                    return f"Which task would you like to update? 🤔"
                
        except Exception as e:
            print(f"Error handling task update: {e}")
            import traceback
            traceback.print_exc()
            return "Which task should I update? 🤔"
    
    def _handle_general_conversation(self, user_id: int, message: str, intent: Dict, 
                                   context: str, username: str) -> str:
        """Handle general conversation and provide contextual responses."""
        try:
            # Check for emotional context
            emotional_context = intent.get("emotional_context", "")
            
            system_prompt = f"""
            {self.system_persona}
            
            Chat naturally with {username}. Keep it SHORT (1-2 sentences). Use emojis 😊
            Emotional vibe: {emotional_context}
            Context: {context}
            
            If it sounds task-related but vague, ask ONE simple question.
            """
            
            user_content = f"{username}: {message}"
            
            # Print LLM input for debugging
            self._print_llm_input("General Conversation", system_prompt, user_content)
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error handling general conversation: {e}")
            return f"Thanks for sharing, {username}! How can I help you stay organized today?"
    
    def reset_conversation_state(self):

        """Reset conversation state for new session."""
        self.conversation_state = {
            "awaiting_clarification": False,
            "pending_task": None,
            "last_intent": None,
            "initial_message_causing_clarification": None
        }