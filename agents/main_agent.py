from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .context_agent import ContextAgent
from .task_agent import TaskAgent
from .prompts import PromptTemplates, clean_json_response
from .response_formatter import ResponseFormatter
from .clarification_handler import ClarificationHandler
from .task_handlers import TaskHandlers


class MainAgent:
    """
    Main conversational agent that routes messages to specialized agents
    and maintains conversation flow.
    """
    
    def __init__(self, openai_api_key: str, db_manager, context_agent: ContextAgent, task_agent: TaskAgent):
        self.openai_api_key = openai_api_key
        self.db_manager = db_manager
        self.context_agent = context_agent
        self.task_agent = task_agent
        
        # Initialize the main LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Initialize task handlers
        self.task_handlers = TaskHandlers(task_agent, db_manager, self.llm)
        
        # State tracking
        self.conversation_state = {
            "awaiting_clarification": False,
            "pending_task": None,
            "last_intent": None
        }
        
        # Conversation buffer for context tracking (stores summaries on task creation)
        self.conversation_buffer = []
    
    def process_message(self, user_id: int, message: str, username: str = "User") -> str:
        """
        Main message processing pipeline.
        """
        try:
            # Get conversation context from ChromaDB
            context_prompt = self._get_context(user_id, message)
            
            # Check if we're awaiting a clarification response
            if self._is_awaiting_clarification(message):
                response = self._handle_clarification_response(user_id, message, context_prompt, username)
            else:
                # Determine intent and route to appropriate handler
                intent = self._analyze_intent(message, context_prompt, username)
                response = self._handle_intent(user_id, message, intent, context_prompt, username)
            
            # Track conversation for periodic summarization
            self._track_conversation(user_id, message, response, username)
            
            return response
            
        except Exception as e:
            return "I'm sorry, I encountered an error. Could you try rephrasing that?"
    
    def _get_context(self, user_id: int, message: str) -> str:
        """Get conversation context from ChromaDB."""
        try:
            return self.context_agent.build_context_prompt(user_id, message, [])
        except Exception as e:
            return ""
    
    def _is_awaiting_clarification(self, message: str) -> bool:
        """Check if we're awaiting a clarification response."""
        awaiting = self.conversation_state.get("awaiting_clarification", False)
        initial_msg = self.conversation_state.get("initial_message_causing_clarification")
        return awaiting and initial_msg != message
    
    def _analyze_intent(self, message: str, context: str, username: str) -> Dict:
        """
        Analyze user message to determine intent and required actions.
        """
        system_prompt = PromptTemplates.build_intent_analysis_prompt(context)
        user_content = f"User ({username}): {message}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            
            content = clean_json_response(response.content)
            intent = json.loads(content)
            return intent
            
        except json.JSONDecodeError as e:
            return self._fallback_intent_detection(message)
        except Exception as e:
            return self._get_default_intent()
    
    def _fallback_intent_detection(self, message: str) -> Dict:
        """Fallback intent detection using keywords."""
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
        return self._get_default_intent()
    
    def _get_default_intent(self) -> Dict:
        """Get default intent structure."""
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
        
        intent_handlers = {
            "task_creation": self._handle_task_creation,
            "task_query": self._handle_task_query,
            "task_update": self._handle_task_update,
            "clarification_response": self._handle_clarification_response
        }
        
        handler = intent_handlers.get(intent_type, self._handle_general_conversation)
        return handler(user_id, message, context, username) if intent_type != "general_chat" else \
               self._handle_general_conversation(user_id, message, intent, context, username)
    
    def _handle_task_creation(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle task creation requests."""
        try:
            # Check for multiple tasks in one message
            if self.task_handlers.check_multiple_tasks(message):
                return self.task_handlers.handle_multiple_tasks(
                    user_id, message, context, username, 
                    self.context_agent, self.conversation_buffer
                )
            
            # Parse task intent
            task_intent = self.task_agent.parse_task_intent(message, user_id, context)
            
            if not task_intent.get("task_title"):
                return "What task would you like me to track? 🤔"
            
            # Check for missing information
            missing_info = ClarificationHandler.check_missing_task_info(task_intent)
            
            # If essential information is missing, ask for clarification
            if missing_info:
                return ClarificationHandler.request_task_clarification(
                    task_intent, missing_info, message, self.llm, self.conversation_state
                )
            
            # All info present, create task
            return self._create_task_with_intent(user_id, task_intent, context, message, username)
        
        except Exception as e:
            return "I'm having trouble creating that task. Could you tell me more?"
    

    
    def _create_task_with_intent(self, user_id: int, task_intent: Dict, context: str, 
                                 message: str, username: str) -> str:
        """Create task with the given intent."""
        success, message_result, task_id, conflicts = self.task_agent.create_task_from_intent(
            user_id, task_intent, context
        )
        
        if success:
            # Store conversation summary immediately when task is created
            self._store_conversation_summary_on_task_creation(user_id, username)
            return ResponseFormatter.format_task_confirmation(task_intent)
        else:
            return ResponseFormatter.handle_task_creation_failure(
                task_intent, conflicts, message, username, self.conversation_state
            )
    

    
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
            
            # Parse the clarification response
            timing_intent = self.task_agent.parse_task_intent(combined_message, user_id, context)
            
            # Handle different clarification types
            if clarification_type == "due_datetime":
                success, response, needs_reminder = ClarificationHandler.handle_due_datetime_clarification(
                    pending_task, timing_intent
                )
                if needs_reminder:
                    self.conversation_state["clarification_type"] = "reminder_datetime"
                return response
            
            elif clarification_type == "reminder_datetime":
                reminder_set, needs_finalization = ClarificationHandler.handle_reminder_clarification(
                    pending_task, timing_intent, message
                )
                
                if not reminder_set:
                    return "When should I remind you? (e.g., 30 minutes before, or a specific time)"
                
                if needs_finalization:
                    success, msg = ClarificationHandler.finalize_clarified_task(
                        user_id, pending_task, username, self.task_agent, self.conversation_state
                    )
                    if success:
                        self._store_conversation_summary_on_task_creation(user_id, username)
                    return msg
                
                success, msg = ClarificationHandler.finalize_clarified_task(
                    user_id, pending_task, username, self.task_agent, self.conversation_state
                )
                if success:
                    self._store_conversation_summary_on_task_creation(user_id, username)
                return msg
            
            elif clarification_type == "conflict_resolution":
                return self._handle_conflict_resolution(pending_task, message, context, user_id, username)
            
            # Try to create the complete task
            success, msg = ClarificationHandler.finalize_clarified_task(
                user_id, pending_task, username, self.task_agent, self.conversation_state
            )
            if success:
                self._store_conversation_summary_on_task_creation(user_id, username)
            return msg
            
        except Exception as e:
            return "Sorry, I didn't quite get that. Could you clarify? 🤔"
    
    def _handle_conflict_resolution(self, pending_task: Dict, message: str, context: str, 
                                   user_id: int, username: str) -> str:
        """Handle conflict resolution during task creation."""
        conflicting_task = self.conversation_state.get("conflicting_task")
        
        # Determine which task to reschedule
        target = ClarificationHandler.determine_conflict_resolution_target(
            conflicting_task, pending_task, message
        )
        
        if target == 'new':
            return self._reschedule_new_task(pending_task, message, context, user_id, username)
        elif target == 'old':
            return self._reschedule_old_task(conflicting_task, pending_task, message, context, user_id, username)
        else:
            return "Would you like to reschedule one of the tasks? Just let me know the new time."
    
    def _reschedule_old_task(self, conflicting_task: Dict, pending_task: Dict, 
                            message: str, context: str, user_id: int, username: str) -> str:
        """Reschedule the old conflicting task using TaskHandlers."""
        result = self.task_handlers.reschedule_old_task(
            conflicting_task, pending_task, message, context, user_id, username,
            self.context_agent, self.conversation_buffer
        )
        
        ClarificationHandler.clear_clarification_state(self.conversation_state)
        self.conversation_state.pop("conflicting_task", None)
        
        return result
    
    def _reschedule_new_task(self, pending_task: Dict, message: str, context: str, user_id: int, username: str) -> str:
        """Reschedule the new pending task using TaskHandlers."""
        result = self.task_handlers.reschedule_new_task(
            pending_task, message, context, user_id, username,
            self.context_agent, self.conversation_buffer
        )
        
        ClarificationHandler.clear_clarification_state(self.conversation_state)
        self.conversation_state.pop("conflicting_task", None)
        
        return result
    
    def _handle_task_query(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle queries about existing tasks."""
        try:
            all_tasks = self.db_manager.get_user_tasks(user_id)
            today = datetime.now().date()
            
            # Filter tasks based on query
            filtered_tasks, time_frame = self.task_handlers.filter_tasks_by_timeframe(all_tasks, message, today)
            
            # Handle empty results
            if not filtered_tasks:
                return ResponseFormatter.format_empty_task_response(time_frame, username)
            
            # Create task summary and generate contextual response
            task_summary = ResponseFormatter.build_task_summary(filtered_tasks, today)
            system_prompt = PromptTemplates.build_task_query_response_prompt(
                username, time_frame, task_summary, today
            )
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {message}")
            ])
            
            return response.content.strip()
            
        except Exception as e:
            return "Let me check your tasks for you..."
    
    def _handle_task_update(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle task modification requests."""
        try:
            tasks = self.db_manager.get_user_tasks(user_id)
            recent_task = self.task_handlers.find_recent_task(tasks)
            
            # Try to update the task
            success, update_message, updated_task = self.task_agent.update_task_from_conversation(
                user_id, message, context, recent_task_hint=recent_task
            )
            
            if success:
                if 'reminder' in message.lower():
                    return f"Done! I've updated the reminder for your {updated_task['title'].lower()} ✅"
                else:
                    return f"All done ✅ I've updated your {updated_task['title'].lower()}."
            else:
                # Help user identify the task to update
                if tasks and len(tasks) > 1:
                    task_names = [t['title'] for t in tasks[:3]]
                    return ResponseFormatter.format_task_list_prompt(task_names)
                else:
                    return "Which task would you like to update? 🤔"
                
        except Exception as e:
            return "Which task should I update? 🤔"
    
    def _handle_general_conversation(self, user_id: int, message: str, intent: Dict, 
                                   context: str, username: str) -> str:
        """Handle general conversation and provide contextual responses."""
        try:
            emotional_context = intent.get("emotional_context", "")
            system_prompt = PromptTemplates.build_general_conversation_prompt(
                username, emotional_context, context
            )
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{username}: {message}")
            ])
            
            return response.content.strip()
            
        except Exception as e:
            return f"Thanks for sharing, {username}! How can I help you stay organized today?"
    

    
    def reset_conversation_state(self):
        """Reset conversation state for new session."""
        self.conversation_state = {
            "awaiting_clarification": False,
            "pending_task": None,
            "last_intent": None,
            "initial_message_causing_clarification": None
        }
        self.conversation_buffer = []
    
    def _track_conversation(self, user_id: int, user_message: str, assistant_response: str, username: str):
        """
        Track conversation exchanges for context and summarization.
        Summary is stored immediately when task is created (not on threshold).
        """
        try:
            # Add to conversation buffer
            self.conversation_buffer.append({
                "user": user_message,
                "assistant": assistant_response,
                "timestamp": datetime.now()
            })
            
            # Keep buffer size manageable (last 10 exchanges)
            if len(self.conversation_buffer) > 10:
                self.conversation_buffer = self.conversation_buffer[-10:]
                
        except Exception as e:
            pass
    
    def _store_conversation_summary_on_task_creation(self, user_id: int, username: str):
        """
        Generate and store LLM-based summary immediately when task is created.
        Delegates to TaskHandlers for consolidated logic.
        """
        self.task_handlers.store_conversation_summary(
            user_id, username, self.context_agent, self.conversation_buffer, self.llm
        )