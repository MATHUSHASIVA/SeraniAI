import os
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import chromadb
from langchain_openai import OpenAIEmbeddings

class ContextAgent:
    """
    Context Manager Agent for handling long-term and short-term memory
    using ChromaDB for vector storage and embeddings.
    """
    
    # Class variable to track if initialization message has been shown
    _initialization_shown = False
    
    def __init__(self, openai_api_key: str, chroma_db_path: str = "database/chroma_db"):
        self.openai_api_key = openai_api_key
        self.chroma_db_path = chroma_db_path
        
        # Initialize ChromaDB - always required
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Initialize OpenAI embeddings - always required
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create collections for different types of memory
        self.conversation_collection = self.chroma_client.get_or_create_collection(
            name="conversation_summaries",
            metadata={"description": "Long-term conversation summaries"}
        )
        
        self.user_context_collection = self.chroma_client.get_or_create_collection(
            name="user_context",
            metadata={"description": "User preferences and behavioral patterns"}
        )
        
        # Only show initialization message once per session
        if not ContextAgent._initialization_shown:
            print("ChromaDB initialized successfully")
            ContextAgent._initialization_shown = True
    
    def store_conversation_summary(self, user_id: int, summary: str, 
                                 start_time: datetime, end_time: datetime,
                                 conversation_metadata: Dict = None):
        """
        Store a conversation summary in ChromaDB with embeddings.
        """
        try:
            # Generate embedding for the summary
            embedding = self.embeddings.embed_query(summary)
            
            # Prepare metadata
            metadata = {
                "user_id": str(user_id),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_summary"
            }
            
            if conversation_metadata:
                metadata.update(conversation_metadata)
            
            # Generate unique ID
            doc_id = f"conv_{user_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Store in ChromaDB
            self.conversation_collection.add(
                embeddings=[embedding],
                documents=[summary],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return doc_id
            
        except Exception as e:
            print(f"Error storing conversation summary: {e}")
            return None
    
    def store_user_context(self, user_id: int, context_type: str, 
                          content: str, metadata: Dict = None):
        """
        Store user context information (preferences, patterns, etc.).
        """
        try:
            # Generate embedding
            embedding = self.embeddings.embed_query(content)
            
            # Prepare metadata
            meta = {
                "user_id": str(user_id),
                "context_type": context_type,
                "timestamp": datetime.now().isoformat(),
                "type": "user_context"
            }
            
            if metadata:
                meta.update(metadata)
            
            # Generate unique ID
            doc_id = f"context_{user_id}_{context_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store in ChromaDB
            self.user_context_collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
                ids=[doc_id]
            )
            
            return doc_id
            
        except Exception as e:
            print(f"Error storing user context: {e}")
            return None
    
    def retrieve_relevant_context(self, user_id: int, query: str, 
                                n_results: int = 3) -> List[Dict]:
        """
        Retrieve most relevant context for a given query.
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search conversation summaries
            conv_results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"user_id": str(user_id)}
            )
            
            # Search user context
            context_results = self.user_context_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"user_id": str(user_id)}
            )
            
            # Combine and format results
            relevant_context = []
            
            # Process conversation summaries
            if conv_results['documents']:
                for i, doc in enumerate(conv_results['documents'][0]):
                    relevant_context.append({
                        "type": "conversation_summary",
                        "content": doc,
                        "metadata": conv_results['metadatas'][0][i],
                        "distance": conv_results['distances'][0][i] if 'distances' in conv_results else 0,
                        "source": "conversation"
                    })
            
            # Process user context
            if context_results['documents']:
                for i, doc in enumerate(context_results['documents'][0]):
                    relevant_context.append({
                        "type": "user_context",
                        "content": doc,
                        "metadata": context_results['metadatas'][0][i],
                        "distance": context_results['distances'][0][i] if 'distances' in context_results else 0,
                        "source": "context"
                    })
            
            # Sort by relevance (lower distance = more relevant)
            relevant_context.sort(key=lambda x: x.get('distance', 0))
            
            return relevant_context[:n_results]
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def summarize_recent_conversations(self, conversations: List[Dict], 
                                    user_id: int) -> Optional[str]:
        """
        Create a summary of recent conversations using OpenAI.
        """
        if not conversations:
            return None
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # Prepare conversation text
            conversation_text = ""
            for conv in conversations:
                role = conv['role']
                message = conv['message']
                timestamp = conv.get('timestamp', 'Unknown time')
                conversation_text += f"{role}: {message}\\n"
            
            # Create summarization prompt
            prompt = f"""
            Please create a concise summary of the following conversation between a user and their personal assistant.
            Focus on:
            1. Key tasks mentioned or created
            2. User preferences and patterns
            3. Emotional context or stress points
            4. Important decisions or commitments
            
            Conversation:
            {conversation_text}
            
            Summary:
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error creating summary: {e}")
            return None
    
    def extract_user_insights(self, conversations: List[Dict], 
                            user_id: int) -> Dict[str, str]:
        """
        Extract user insights from conversations for context storage.
        """
        if not conversations:
            return {}
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # Prepare conversation text
            conversation_text = ""
            for conv in conversations:
                role = conv['role']
                message = conv['message']
                conversation_text += f"{role}: {message}\\n"
            
            # Create insight extraction prompt
            prompt = f"""
            Analyze the following conversation and extract key insights about the user:
            
            {conversation_text}
            
            Please extract:
            1. Communication style and preferences
            2. Work patterns and habits  
            3. Stress indicators and emotional patterns
            4. Task management preferences
            5. Time management patterns
            
            Format as JSON with keys: communication_style, work_patterns, emotional_patterns, task_preferences, time_patterns
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            try:
                insights = json.loads(response.choices[0].message.content.strip())
                return insights
            except json.JSONDecodeError:
                # Fallback: store as plain text
                return {"general_insights": response.choices[0].message.content.strip()}
            
        except Exception as e:
            print(f"Error extracting insights: {e}")
            return {}
    
    def build_context_prompt(self, user_id: int, current_query: str,
                           recent_conversations: List[Dict]) -> str:
        """
        Build a comprehensive context prompt combining short-term and long-term memory.
        """
        try:
            # Get relevant long-term context
            relevant_context = self.retrieve_relevant_context(user_id, current_query)
            
            # Build context prompt
            context_parts = []
            
            # Add recent conversations (short-term memory)
            if recent_conversations:
                context_parts.append("## Recent Conversation History:")
                for conv in recent_conversations[-5:]:  # Last 5 messages
                    role = conv['role']
                    message = conv['message']
                    context_parts.append(f"{role}: {message}")
            
            # Add relevant long-term context
            if relevant_context:
                context_parts.append("\\n## Relevant Past Context:")
                for ctx in relevant_context[:3]:  # Top 3 most relevant
                    context_parts.append(f"- {ctx['content']}")
            
            # Add user patterns if available
            user_patterns = self.get_user_patterns(user_id)
            if user_patterns:
                context_parts.append("\\n## User Patterns:")
                context_parts.append(user_patterns)
            
            return "\\n".join(context_parts)
            
        except Exception as e:
            print(f"Error building context prompt: {e}")
            return ""
    
    def get_user_patterns(self, user_id: int) -> str:
        """
        Get user behavioral patterns from stored context.
        """
        try:
            # Query for user patterns
            try:
                pattern_results = self.user_context_collection.get(
                    where={
                        "user_id": str(user_id),
                        "context_type": "patterns"
                    }
                )
                
                if pattern_results['documents']:
                    patterns = []
                    for doc in pattern_results['documents']:
                        patterns.append(f"- {doc}")
                    return "\\n".join(patterns)
            except Exception:
                # If no patterns exist, return empty
                pass
            
            return ""
            
        except Exception as e:
            print(f"Error getting user patterns: {e}")
            return ""
    
