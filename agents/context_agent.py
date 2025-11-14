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
    
    _initialization_shown = False
    
    def __init__(self, openai_api_key: str, chroma_db_path: str = "database/chroma_db"):
        self.openai_api_key = openai_api_key
        self.chroma_db_path = chroma_db_path
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
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
        
        # Show initialization message once per session
        if not ContextAgent._initialization_shown:
            ContextAgent._initialization_shown = True
    
    def store_conversation_summary(self, user_id: int, summary: str, 
                                 start_time: datetime, end_time: datetime,
                                 conversation_metadata: Dict = None):
        """
        Store a conversation summary in ChromaDB with embeddings.
        Only user_id is stored in metadata for simple vector retrieval.
        """
        try:
            # Generate embedding and metadata
            embedding = self.embeddings.embed_query(summary)
            metadata = {"user_id": str(user_id)}
            doc_id = f"conv_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Store in ChromaDB
            self.conversation_collection.add(
                embeddings=[embedding],
                documents=[summary],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return doc_id
            
        except Exception as e:
            return None
    
    def retrieve_relevant_context(self, user_id: int, query: str, 
                                n_results: int = 3) -> List[Dict]:
        """
        Retrieve most relevant conversation summaries using vector similarity search.
        Simple RAG: query -> embed -> find similar -> return results
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search conversation summaries
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
            return []
    
    def build_context_prompt(self, user_id: int, current_query: str,
                           recent_conversations: List[Dict]) -> str:
        """
        Build context prompt by retrieving similar conversation summaries from ChromaDB.
        Simple RAG pattern: Query -> Retrieve similar vectors -> Build context.
        """
        try:
            # Get relevant context from vector search
            relevant_context = self.retrieve_relevant_context(user_id, current_query, n_results=5)
            
            context_parts = []
            
            # Add retrieved similar conversations
            if relevant_context:
                context_parts.append("## Relevant Past Context:")
                for ctx in relevant_context:
                    context_parts.append(f"- {ctx['content']}")
            
            # Add recent short-term memory (last 3 messages)
            if recent_conversations and len(recent_conversations) > 0:
                context_parts.append("\n## Recent Conversation:")
                for conv in recent_conversations[-3:]:
                    role = conv.get('role', 'unknown')
                    message = conv.get('message', '')
                    context_parts.append(f"{role}: {message}")
            
            final_context = "\n".join(context_parts)
            
            return final_context
            
        except Exception as e:
            return ""
