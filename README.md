# Contextual Personal Assistant - Serani

A sophisticated multi-agent chatbot system built with OpenAI, ChromaDB, LangChain, and Streamlit for task management and contextual conversations.

## Features

- **🤖 Conversational AI**: Natural language interaction with context awareness
- **📝 Task Management**: Create, update, and manage tasks with conflict resolution
- **🧠 Long-term Memory**: ChromaDB-powered context retention across sessions
- **🎯 Multi-Agent Architecture**: Specialized agents for different functionalities
- **💬 Streamlit Interface**: Modern web-based chat UI
- **⚡ Real-time Processing**: Instant responses with background processing

## Architecture

```
├── agents/                 # Multi-agent system
│   ├── main_agent.py      # Central conversational router
│   ├── task_agent.py      # Task creation and management
│   └── context_agent.py   # Memory and context handling
├── database/              # Data persistence
│   └── db_manager.py      # SQLite database management
├── ui/                    # User interface
│   └── chat_interface.py  # Streamlit chat application
├── config/                # Configuration
│   └── settings.py        # Application settings
└── utils/                 # Utilities
    └── helpers.py         # Helper functions
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd contextual-assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment** (optional):
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

### Web Interface

1. **Start the Streamlit application**:
   ```bash
   streamlit run ui/chat_interface.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Enter your OpenAI API key** and username in the sidebar

4. **Start chatting** with Serani!

### Programmatic Usage (Optional)

If you need to use the assistant programmatically in your own scripts:

```python
from app import SeraniAssistant

# Initialize the assistant
assistant = SeraniAssistant("your-openai-api-key")
assistant.initialize("YourName")

# Chat with the assistant
response = assistant.chat("Remind me to prepare for my presentation tomorrow evening", "YourName")
print(response)

# Get tasks
tasks = assistant.get_tasks("YourName")
print(tasks)
```

## Sample Conversation

```
User: Hey Serani, remind me to start preparing for my project presentation.

Serani: Got it, Viru! When do you plan to start working on it — today or later this week?

User: Let's do it tomorrow evening.

Serani: Alright ■ I'll remind you tomorrow evening. How long would you like to focus on it?

User: Maybe two hours should be enough.

Serani: Noted — two hours of project prep time ■

By the way, you mentioned last week you were feeling a bit stressed about this project. Want me to set a short motivational note before you start tomorrow?

User: Haha yes, that would actually help.

Serani: Perfect ■ I'll drop a quick "You've got this, Viru ■" message right before your session.

User: Thanks, Serani.

Serani: Anytime! You focus on your goals — I'll handle the reminders. Deal? ■
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_PATH`: SQLite database path (default: `database/assistant.db`)
- `CHROMA_DB_PATH`: ChromaDB storage path (default: `database/chroma_db`)
- `DEBUG`: Enable debug mode (default: `False`)

### OpenAI API Configuration

The system uses OpenAI's GPT-3.5-turbo model by default. You can modify the model and parameters in `config/settings.py`.

**Estimated API Costs**:
- Conversation processing: ~$0.002-0.01 per message
- Task parsing and intent recognition: ~$0.001-0.003 per operation
- Context summarization: ~$0.005-0.02 per summary

## Components

### 1. Main Agent (`agents/main_agent.py`)
- Central conversation router
- Intent analysis and classification
- Coordinates other agents
- Maintains conversation flow

### 2. Task Agent (`agents/task_agent.py`)
- Natural language task parsing
- Schedule conflict detection
- Task creation and updates
- Conflict resolution suggestions

### 3. Context Agent (`agents/context_agent.py`)
- Long-term memory management
- ChromaDB integration for embeddings
- Conversation summarization
- Context retrieval for prompts

### 4. Database Manager (`database/db_manager.py`)
- SQLite database operations
- User, task, and conversation storage
- Conflict checking queries

## Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `created_at`: Account creation timestamp
- `preferences`: JSON preferences storage

### Tasks Table
- `id`: Primary key
- `user_id`: Foreign key to users
- `title`: Task title
- `description`: Task description
- `start_time`: Scheduled start time
- `end_time`: Calculated end time
- `duration_minutes`: Task duration
- `status`: pending/completed/cancelled
- `priority`: low/medium/high
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### Conversations Table
- `id`: Primary key
- `user_id`: Foreign key to users
- `message`: Message content
- `role`: user/assistant
- `timestamp`: Message timestamp

## Features in Detail

### Natural Language Processing
- Intent recognition for task creation
- Time parsing ("tomorrow evening" → specific datetime)
- Duration parsing ("two hours" → 120 minutes)
- Conflict detection and resolution

### Memory Management
- **Short-term**: Last 5 conversation messages
- **Long-term**: ChromaDB embeddings of conversation summaries
- **Contextual**: Relevant context retrieval for each interaction

### Task Management
- Create tasks from natural language
- Automatic conflict detection
- Schedule optimization suggestions
- Status tracking and updates

## Troubleshooting

### Common Issues

1. **"Assistant not initialized"**
   - Ensure OpenAI API key is provided
   - Check database permissions
   - Verify all dependencies are installed

2. **"ChromaDB connection failed"**
   - Check disk space for database files
   - Ensure write permissions in database directory

3. **"OpenAI API error"**
   - Verify API key is valid
   - Check API usage limits
   - Ensure internet connectivity

### Debug Mode

Enable debug mode by setting `DEBUG=True` in your environment or `config/settings.py` for detailed logging.

## Development

### Project Structure
```
contextual-assistant/
├── agents/           # Agent implementations
├── database/         # Database management
├── ui/              # User interface
├── config/          # Configuration
├── utils/           # Utility functions
├── app.py           # Main application
├── requirements.txt # Dependencies
└── README.md        # This file
```

### Adding New Features

1. **New Agent**: Create in `agents/` directory
2. **Database Changes**: Update `database/db_manager.py`
3. **UI Updates**: Modify `ui/chat_interface.py`
4. **Configuration**: Add to `config/settings.py`

## License

This project is part of a candidate assignment for Viso Labs.

## Support

For issues and questions, please refer to the project documentation or contact the development team.