from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .context_agent import ContextAgent
from .task_agent import TaskAgent

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
            temperature=0.7  # Slightly higher for more natural conversation
        )
        
        # Conversation state tracking (simplified for compatibility)
        
        # System persona
        self.system_persona = """
        You are Serani, a helpful and friendly contextual personal assistant. You are:
        
        **Personality:**
        - Warm, conversational, and supportive
        - Proactive in helping users stay organized
        - Understanding of stress and workload
        - Uses natural, human-like language with occasional emoji ■
        
        **Capabilities:**
        - Create and manage tasks and reminders
        - Remember past conversations and preferences  
        - Detect emotional context and provide motivation
        - Handle scheduling conflicts intelligently
        - Learn from user patterns and adapt
        
        **Communication Style:**
        - Keep responses concise but friendly
        - Ask clarifying questions when needed
        - Acknowledge user emotions and stress
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
            context_prompt = self.context_agent.build_context_prompt(
                user_id, message, []
            )
            
            # Determine intent and route to appropriate handler
            intent = self._analyze_intent(message, context_prompt, username)
            
            # Process based on intent
            response = self._handle_intent(user_id, message, intent, context_prompt, username)
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your message. Could you please try again?"
    
    def _analyze_intent(self, message: str, context: str, username: str) -> Dict:
        """
        Analyze user message to determine intent and required actions.
        """
        system_prompt = f"""
        {self.system_persona}
        
        Analyze the user's message to determine their intent. Consider the conversation context.
        
        Context: {context}
        
        Classify the intent as one of:
        1. "task_creation" - User explicitly wants to create a new task, reminder, or asks you to remind them about something
           - Keywords: "remind me", "schedule", "set a reminder", "I need to", "help me remember"
           - NOT for: sharing information, travel plans, or general updates
        2. "task_query" - User wants to see their tasks or ask about scheduling
        3. "task_update" - User wants to modify an existing task
        4. "general_chat" - General conversation, sharing information, travel updates, or unclear intent
           - Use this for: sharing travel plans, life updates, general information
        5. "clarification_response" - User is responding to a clarifying question
        
        IMPORTANT: Only classify as "task_creation" if the user explicitly asks you to remind them or create a task.
        Sharing information (like flight details) should be classified as "general_chat".
        
        Return JSON:
        {{
            "intent": string,
            "confidence": float (0-1),
            "requires_task_agent": boolean,
            "needs_clarification": boolean,
            "clarification_type": string or null,
            "emotional_context": string or null
        }}
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User ({username}): {message}")
            ])
            
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            intent = json.loads(content)
            return intent
            
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
            
            # Check if we need clarification
            missing_info = []
            if not task_intent.get("task_title"):
                missing_info.append("task description")
            
            # For demonstration, let's check if we have timing info
            if task_intent.get("task_title") and not task_intent.get("start_time"):
                # Ask for timing if it seems like a scheduled task
                self.conversation_state["awaiting_clarification"] = True
                self.conversation_state["pending_task"] = task_intent
                return f"Got it, {username}! When do you plan to start working on '{task_intent['task_title']}' — today or later this week?"
            
            if task_intent.get("start_time") and not task_intent.get("duration_minutes"):
                # Ask for duration
                self.conversation_state["awaiting_clarification"] = True  
                self.conversation_state["pending_task"] = task_intent
                return f"Alright ■ I'll remind you {self._format_time_reference(task_intent['start_time'])}. How long would you like to focus on it?"
            
            # Try to create the task
            success, message_result, task_id = self.task_agent.create_task_from_intent(
                user_id, task_intent, context
            )
            
            if success:
                # Add encouraging follow-up
                responses = [
                    message_result,
                    f"You focus on your goals — I'll handle the organization. Deal? ■"
                ]
                return "\\n\\n".join(responses)
            else:
                # Handle conflicts or errors gracefully
                if "conflict" in message_result.lower():
                    return f"I see there's a scheduling conflict. {message_result}\\n\\nWould you like me to suggest alternative times?"
                else:
                    return f"I had some trouble with that: {message_result}. Could you provide more details?"
        
        except Exception as e:
            print(f"Error handling task creation: {e}")
            return "I'm having trouble creating that task. Could you tell me more about what you'd like to schedule?"
    
    def _handle_clarification_response(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle responses to clarifying questions."""
        try:
            if not self.conversation_state.get("awaiting_clarification"):
                return self._handle_general_conversation(user_id, message, {}, context, username)
            
            pending_task = self.conversation_state.get("pending_task", {})
            
            # Parse the clarification response
            if not pending_task.get("start_time"):
                # User is providing timing information
                timing_intent = self.task_agent.parse_task_intent(
                    f"Schedule {pending_task.get('task_title', 'task')} {message}", 
                    user_id, context
                )
                
                if timing_intent.get("start_time"):
                    pending_task["start_time"] = timing_intent["start_time"]
                    
                    if not pending_task.get("duration_minutes"):
                        # Still need duration
                        return f"Alright ■ I'll set that for {self._format_time_reference(timing_intent['start_time'])}. How long would you like to focus on it?"
                
            elif not pending_task.get("duration_minutes"):
                # User is providing duration
                duration_intent = self.task_agent.parse_task_intent(
                    f"Duration: {message}", user_id, context
                )
                
                if duration_intent.get("duration_minutes"):
                    pending_task["duration_minutes"] = duration_intent["duration_minutes"]
            
            # Try to create the complete task
            success, result_message, task_id = self.task_agent.create_task_from_intent(
                user_id, pending_task, context
            )
            
            # Clear clarification state
            self.conversation_state["awaiting_clarification"] = False
            self.conversation_state["pending_task"] = None
            
            if success:
                duration = pending_task.get("duration_minutes", 0)
                hours = duration // 60
                mins = duration % 60
                duration_str = f"{hours} hour{'s' if hours != 1 else ''}" if hours > 0 else f"{mins} minutes"
                
                responses = [
                    f"Noted — {duration_str} of {pending_task.get('task_title', 'work')} time ■",
                    "Anytime! You focus on your goals — I'll handle the reminders. Deal? ■"
                ]
                return "\\n\\n".join(responses)
            else:
                return f"I had trouble setting that up: {result_message}. Let's try again?"
            
        except Exception as e:
            print(f"Error handling clarification: {e}")
            return "Sorry, I didn't quite get that. Could you clarify what you'd like me to help with?"
    
    def _handle_task_query(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle queries about existing tasks."""
        try:
            task_summary = self.task_agent.get_task_summary(user_id)
            
            # Generate contextual response
            system_prompt = f"""
            {self.system_persona}
            
            The user is asking about their tasks. Provide a helpful response using the task information.
            
            Task Summary: {task_summary}
            Context: {context}
            
            Be conversational and supportive. If they have many tasks, acknowledge their workload.
            If they have few tasks, be encouraging about staying organized.
            """
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {message}")
            ])
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error handling task query: {e}")
            return "Let me check your tasks for you..."
    
    def _handle_task_update(self, user_id: int, message: str, context: str, username: str) -> str:
        """Handle task modification requests."""
        try:
            success, update_message = self.task_agent.update_task_from_conversation(
                user_id, message, context
            )
            
            if success:
                return f"Done! {update_message} ■"
            else:
                return f"I'm having trouble with that update: {update_message}. Could you be more specific?"
                
        except Exception as e:
            print(f"Error handling task update: {e}")
            return "I'm not sure which task you'd like to update. Could you clarify?"
    
    def _handle_general_conversation(self, user_id: int, message: str, intent: Dict, 
                                   context: str, username: str) -> str:
        """Handle general conversation and provide contextual responses."""
        try:
            # Check for emotional context
            emotional_context = intent.get("emotional_context", "")
            
            system_prompt = f"""
            {self.system_persona}
            
            Respond to the user's message naturally and helpfully. Consider:
            - Their current emotional state: {emotional_context}
            - Previous conversation context: {context}
            - Be supportive and encouraging
            - Keep responses concise but warm
            
            If the message seems like it might be task-related but unclear, gently ask for clarification.
            """
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{username}: {message}")
            ])
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error handling general conversation: {e}")
            return f"Thanks for sharing, {username}! How can I help you stay organized today?"
    
    def _format_time_reference(self, time_str: str) -> str:
        """Format time string for natural conversation."""
        try:
            if isinstance(time_str, str):
                time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            else:
                time_obj = time_str
            
            now = datetime.now()
            
            if time_obj.date() == now.date():
                return f"today at {time_obj.strftime('%I:%M %p')}"
            elif time_obj.date() == (now.date() + timedelta(days=1)):
                return f"tomorrow at {time_obj.strftime('%I:%M %p')}"
            else:
                return time_obj.strftime('%B %d at %I:%M %p')
                
        except Exception as e:
            return str(time_str)
    
    def get_conversation_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get conversation history - now managed by Streamlit session state."""
        # Chat history is managed by Streamlit session state
        # This method returns empty for compatibility
        return []
    
    def reset_conversation_state(self):
        """Reset conversation state for new session."""
        self.conversation_state = {
            "awaiting_clarification": False,
            "pending_task": None,
            "last_intent": None
        }