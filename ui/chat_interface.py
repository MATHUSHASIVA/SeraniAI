import streamlit as st
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseManager
from agents import ContextAgent, TaskAgent, MainAgent

class ChatInterface:
    """Streamlit-based chat interface for the Contextual Personal Assistant."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_sidebar()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Serani - Personal Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for chat styling
        st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 2rem;
        }
        .assistant-message {
            background-color: #f5f5f5;
            margin-right: 2rem;
        }
        .message-timestamp {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.25rem;
        }
        .task-card {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.25rem;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.initializing = False
            st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
            st.session_state.username = "Viru"
            st.session_state.user_id = None
            st.session_state.chat_history = []
            st.session_state.db_manager = None
            st.session_state.main_agent = None
            st.session_state.context_agent = None
            st.session_state.task_agent = None
    
    def setup_sidebar(self):
        """Setup the sidebar with configuration options."""
        with st.sidebar:
            st.title("🤖 Serani Assistant")
            st.markdown("---")
            
            # Check if API key is available
            api_key = st.session_state.openai_api_key
            if not api_key:
                st.error("⚠️ OpenAI API key not found in .env file!")
                st.info("Please add your OpenAI API key to the .env file")
                st.code("OPENAI_API_KEY=your_api_key_here")
                return
            else:
                st.success("🔑 OpenAI API key loaded from .env")
            
            # Username input
            username = st.text_input(
                "Your Name", 
                value=st.session_state.username,
                help="How should Serani address you?"
            )
            
            # Initialize button - disable if already initialized or initializing
            button_disabled = st.session_state.get('initializing', False) or st.session_state.get('initialized', False)
            button_label = "✅ Initialized" if st.session_state.get('initialized', False) else "Initialize Assistant"
            
            if st.button(button_label, type="primary", disabled=button_disabled):
                if username.strip():
                    self.initialize_assistant(api_key, username.strip())
                else:
                    st.error("Please provide your name")
            
            st.markdown("---")
            
            # Show initialization status
            if st.session_state.get('initializing', False):
                st.warning("🔄 Initializing assistant...")
            elif st.session_state.initialized:
                st.success("✅ Assistant Ready")
                st.info(f"👤 User: {st.session_state.username}")
                
                # Task summary in sidebar
                if st.session_state.task_agent:
                    try:
                        task_summary = st.session_state.task_agent.get_task_summary(st.session_state.user_id)
                        st.markdown("### 📅 Tasks Overview")
                        st.text_area("Tasks", value=task_summary, height=200, disabled=True, label_visibility="collapsed")
                    except Exception as e:
                        st.error(f"Error loading tasks: {e}")
                
                # Clear chat button
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    if st.session_state.main_agent:
                        st.session_state.main_agent.reset_conversation_state()
                    st.rerun()
            else:
                st.warning("⚠️ Assistant not initialized")
            
            st.markdown("---")
            st.markdown("""
            ### About Serani
            Your contextual personal assistant for:
            - 📝 Task management
            - 🧠 Context-aware conversations
            - 🎯 Conflict resolution
            - 💾 Long-term memory
            """)
    
    def initialize_assistant(self, api_key: str, username: str):
        """Initialize the assistant with all components."""
        # Prevent re-initialization if already initialized or in progress
        if st.session_state.get('initialized', False):
            st.info("✅ Assistant is already initialized!")
            return
            
        if st.session_state.get('initializing', False):
            st.warning("⏳ Initialization already in progress...")
            return
            
        try:
            # Set flag to prevent multiple initialization attempts
            st.session_state.initializing = True
            
            with st.spinner("Initializing assistant..."):
                # Update session state
                st.session_state.openai_api_key = api_key
                st.session_state.username = username
                
                # Initialize database
                progress = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing database...")
                progress.progress(20)
                st.session_state.db_manager = DatabaseManager()
                
                # Get or create user
                status_text.text("Setting up user profile...")
                progress.progress(40)
                st.session_state.user_id = st.session_state.db_manager.get_or_create_user(username)
                
                # Initialize agents
                status_text.text("Loading context agent...")
                progress.progress(60)
                st.session_state.context_agent = ContextAgent(api_key)
                
                status_text.text("Loading task agent...")
                progress.progress(70)
                st.session_state.task_agent = TaskAgent(api_key, st.session_state.db_manager)
                
                status_text.text("Loading main agent...")
                progress.progress(80)
                st.session_state.main_agent = MainAgent(
                    api_key, 
                    st.session_state.db_manager,
                    st.session_state.context_agent,
                    st.session_state.task_agent
                )
                
                # Initialize empty chat history
                status_text.text("Initializing chat history...")
                progress.progress(90)
                st.session_state.chat_history = []
                
                status_text.text("Finalizing setup...")
                progress.progress(100)
                
                # Clear progress indicators
                progress.empty()
                status_text.empty()
                
                # Mark as successfully initialized - DO THIS BEFORE RERUN
                st.session_state.initialized = True
                st.session_state.initializing = False
                
                st.success("✅ Assistant initialized successfully!")
                
                # Small delay to show success message
                import time
                time.sleep(0.5)
                
                # Rerun to refresh the UI
                st.rerun()
                
        except Exception as e:
            # Reset initialization flags on error
            st.session_state.initialized = False
            st.session_state.initializing = False
            
            st.error(f"❌ Failed to initialize assistant: {str(e)}")
            st.error("Please check your API key and try again.")
            
            # Display detailed error information in expander
            with st.expander("🔍 Error Details"):
                st.exception(e)
    
    def render_chat_history(self):
        """Render the chat conversation history."""
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-message assistant-message">
                <div><strong>Serani:</strong> Hi there! I'm Serani, your personal assistant. 
                I'm here to help you manage your tasks and stay organized. 
                How can I help you today? ■</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                timestamp = message.get("timestamp", "")
                
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div><strong>{st.session_state.username}:</strong> {content}</div>
                        <div class="message-timestamp">{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div><strong>Serani:</strong> {content}</div>
                        <div class="message-timestamp">{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def handle_user_input(self, user_input: str):
        """Process user input and get assistant response."""
        if not st.session_state.initialized:
            st.error("Please initialize the assistant first")
            return
        
        try:
            # Add user message to history immediately for better UX
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.chat_history.append(user_message)
            
            # Get response from main agent
            with st.spinner("Serani is thinking..."):
                try:
                    response = st.session_state.main_agent.process_message(
                        st.session_state.user_id, 
                        user_input, 
                        st.session_state.username
                    )
                except Exception as process_error:
                    st.error(f"Error in process_message: {str(process_error)}")
                    import traceback
                    st.code(traceback.format_exc())
                    response = "I'm sorry, I encountered an error. Please try again."
            
            # Add assistant response to history
            assistant_message = {
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.chat_history.append(assistant_message)
            
            # Store conversation summary in ChromaDB every 4 messages (2 exchanges)
            if len(st.session_state.chat_history) % 4 == 0:
                try:
                    # Get last 4 messages (2 user + 2 assistant)
                    recent_messages = st.session_state.chat_history[-4:]
                    
                    # Create simple conversation text for summarization
                    conversation_text = ""
                    for msg in recent_messages:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        conversation_text += f"{role}: {msg['content']}\n"
                    
                    # Generate summary using LLM
                    from openai import OpenAI
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    
                    summary_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "user",
                            "content": f"Summarize this conversation in 1-2 sentences focusing on key actions, tasks, and context:\n\n{conversation_text}"
                        }],
                        max_tokens=100,
                        temperature=0.3
                    )
                    
                    summary = summary_response.choices[0].message.content.strip()
                    
                    # Store in ChromaDB with simple metadata (only user_id)
                    st.session_state.context_agent.store_conversation_summary(
                        user_id=st.session_state.user_id,
                        summary=summary,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        conversation_metadata={}  # Keep empty, only user_id in metadata
                    )
                    
                    print(f"✅ Stored summary: {summary}")
                
                except Exception as e:
                    print(f"⚠️ Failed to store summary: {e}")
            
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def render_main_interface(self):
        """Render the main chat interface."""
        st.title("💬 Chat with Serani")
        
        if not st.session_state.initialized:
            st.warning("👈 Please initialize the assistant using the sidebar first")
            return
        
        # Chat container
        chat_container = st.container()
        
        # Chat input at bottom
        with st.container():
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message Serani",
                    key="user_input",
                    placeholder="Type your message here...",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.button("Send", type="primary", use_container_width=True)
            
            # Handle input
            if send_button and user_input.strip():
                self.handle_user_input(user_input.strip())
                st.rerun()
            
            # Also handle enter key
            if user_input and user_input != st.session_state.get('last_input', ''):
                st.session_state.last_input = user_input
        
        # Display chat history in the container
        with chat_container:
            self.render_chat_history()
    
    def render_tasks_page(self):
        """Render a dedicated tasks management page."""
        st.title("📅 Task Management")
        
        if not st.session_state.initialized:
            st.warning("Please initialize the assistant first")
            return
        
        try:
            # Get user tasks
            tasks = st.session_state.db_manager.get_user_tasks(st.session_state.user_id)
            
            # Task statistics
            st.metric("Total Tasks", len(tasks))
            
            # Display tasks
            if tasks:
                st.subheader("📋 Your Tasks")
                for task in tasks:
                    status_icon = "✅" if task.get('status') == 'completed' else "⏳" if task.get('status') == 'pending' else "❌"
                    with st.expander(f"{status_icon} {task['title']} - {task.get('status', 'pending').upper()}", expanded=False):
                        if task['description']:
                            st.write(f"**Description:** {task['description']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Created:** {task['created_at']}")
                            if task['due_date'] and task['due_time']:
                                st.write(f"**Due:** {task['due_date']} at {task['due_time']}")
                            else:
                                st.write(f"**Due:** Not set")
                            st.write(f"**Status:** {task.get('status', 'pending').capitalize()}")
                        with col2:
                            if task['reminder_date'] and task['reminder_time']:
                                st.write(f"**Reminder:** {task['reminder_date']} at {task['reminder_time']}")
                            else:
                                st.write(f"**Reminder:** Not set")
                        
                        # Status update buttons
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        with col_btn1:
                            if st.button("Mark Complete", key=f"complete_{task['id']}"):
                                st.session_state.db_manager.update_task(task['id'], status='completed')
                                st.success("Task marked as completed!")
                                st.rerun()
                        with col_btn2:
                            if st.button("Mark Pending", key=f"pending_{task['id']}"):
                                st.session_state.db_manager.update_task(task['id'], status='pending')
                                st.success("Task marked as pending!")
                                st.rerun()
                        with col_btn3:
                            if st.button("Delete", key=f"delete_{task['id']}"):
                                st.session_state.db_manager.delete_task(task['id'])
                                st.success("Task deleted!")
                                st.rerun()
            
            if not tasks:
                st.info("No tasks yet. Start by chatting with Serani to create your first task!")
            
        except Exception as e:
            st.error(f"Error loading tasks: {e}")
    
    def run(self):
        """Run the Streamlit application."""
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["💬 Chat", "📅 Tasks"],
            key="page_selector"
        )
        
        if page == "💬 Chat":
            self.render_main_interface()
        elif page == "📅 Tasks":
            self.render_tasks_page()

def main():
    """Main function to run the Streamlit app."""
    app = ChatInterface()
    app.run()

if __name__ == "__main__":
    main()