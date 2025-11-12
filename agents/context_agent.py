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
        Only user_id is stored in metadata for simple vector retrieval.
        """
        try:
            # Generate embedding for the summary
            embedding = self.embeddings.embed_query(summary)
            
            # Simple metadata - only user_id
            metadata = {
                "user_id": str(user_id)
            }
            
            # Generate unique ID
            doc_id = f"conv_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Store in ChromaDB
            self.conversation_collection.add(
                embeddings=[embedding],
                documents=[summary],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"💾 Stored: {summary[:60]}...")
            
            return doc_id
            
        except Exception as e:
            print(f"❌ Error storing summary: {e}")
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
        Retrieve most relevant conversation summaries using vector similarity search.
        Simple RAG: query -> embed -> find similar -> return results
        """
        try:
            # Generate query embedding (silent operation)
            query_embedding = self.embeddings.embed_query(query)
            
            # Search conversation summaries only
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"user_id": str(user_id)}
            )
            
            # Format results
            relevant_context = []
            
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    relevant_context.append({
                        "content": doc,
                        "distance": results['distances'][0][i] if 'distances' in results else 0
                    })
            
            return relevant_context
            
        except Exception as e:
            print(f"   ❌ Error retrieving context: {e}")
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
        Build context prompt by retrieving similar conversation summaries from ChromaDB.
        Simple RAG pattern: Query -> Retrieve similar vectors -> Build context.
        """
        try:
            print(f"🗄️  ChromaDB Retrieval | User: {user_id} | Query: {current_query[:50]}...")
            
            # Get relevant context from vector search
            relevant_context = self.retrieve_relevant_context(user_id, current_query, n_results=5)
            
            # Build simple context prompt
            context_parts = []
            
            # Add retrieved similar conversations
            if relevant_context:
                print(f"   ✓ Found {len(relevant_context)} similar conversations")
                context_parts.append("## Relevant Past Context:")
                for ctx in relevant_context:
                    context_parts.append(f"- {ctx['content']}")
            else:
                print(f"   ℹ️ No similar past conversations found")
            
            # Add recent short-term memory (last 3 messages for immediate context)
            if recent_conversations and len(recent_conversations) > 0:
                print(f"   ✓ Added {min(3, len(recent_conversations))} recent messages")
                context_parts.append("\n## Recent Conversation:")
                for conv in recent_conversations[-3:]:
                    role = conv.get('role', 'unknown')
                    message = conv.get('message', '')
                    context_parts.append(f"{role}: {message}")
            
            final_context = "\n".join(context_parts)
            print(f"   📊 Total context: {len(final_context)} chars\n")
            
            return final_context
            
        except Exception as e:
            print(f"   ❌ Error building context: {e}")
            import traceback
            traceback.print_exc()
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
    
