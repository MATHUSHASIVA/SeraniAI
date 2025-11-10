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



