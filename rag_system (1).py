import os
from typing import List, Dict, Any
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from document_processor import DocumentProcessor

class GeographyRAGSystem:
    def __init__(self, groq_api_key: str, vector_store_path: str = "vector_store"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.vector_store_path = vector_store_path
        self.load_vector_store()
        
    def load_vector_store(self):
        """Load the vector store"""
        try:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Vector store loaded successfully!")
            else:
                print("Vector store not found. Please process documents first.")
        except Exception as e:
            print(f"Error loading vector store: {e}")
    
    def retrieve_relevant_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents for the query"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def create_context(self, relevant_docs: List[str]) -> str:
        """Create context from relevant documents"""
        if not relevant_docs:
            return "No relevant information found in the geography textbook."
        
        context = "Based on the NCERT Class 10 Geography textbook:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Reference {i}:\n{doc}\n\n"
        
        return context
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Groq API"""
        try:
            system_prompt = """You are an AI geography tutor specializing in NCERT Class 10 Geography. 
            Your role is to help students understand geographical concepts, processes, and phenomena.
            
            Instructions:
            1. Use the provided context from the NCERT textbook to answer questions
            2. Provide clear, educational explanations suitable for Class 10 students
            3. Include relevant examples and real-world applications when possible
            4. If the context doesn't contain enough information, acknowledge this and provide general guidance
            5. Encourage further learning and exploration of the topic
            6. Use simple language that students can easily understand
            """
            
            user_prompt = f"""Context from NCERT Class 10 Geography textbook:
            {context}
            
            Student Question: {query}
            
            Please provide a comprehensive answer based on the context provided. If the context is insufficient, 
            provide what you can and suggest where the student might find more information."""
            
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Using a supported model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def get_streaming_response(self, query: str, context: str):
        """Generate streaming response using Groq API"""
        try:
            system_prompt = """You are an AI geography tutor specializing in NCERT Class 10 Geography. 
            Your role is to help students understand geographical concepts, processes, and phenomena.
            
            Instructions:
            1. Use the provided context from the NCERT textbook to answer questions
            2. Provide clear, educational explanations suitable for Class 10 students
            3. Include relevant examples and real-world applications when possible
            4. If the context doesn't contain enough information, acknowledge this and provide general guidance
            5. Encourage further learning and exploration of the topic
            6. Use simple language that students can easily understand
            """
            
            user_prompt = f"""Context from NCERT Class 10 Geography textbook:
            {context}
            
            Student Question: {query}
            
            Please provide a comprehensive answer based on the context provided."""
            
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=True
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error generating response: {e}"
    
    def answer_question(self, query: str, use_streaming: bool = False) -> str:
        """Main method to answer geography questions"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query)
        
        # Create context
        context = self.create_context(relevant_docs)
        
        # Generate response
        if use_streaming:
            return self.get_streaming_response(query, context)
        else:
            return self.generate_response(query, context)
    
    def get_topic_suggestions(self) -> List[str]:
        """Get topic suggestions based on the textbook content"""
        topics = [
            "Resources and Development",
            "Forest and Wildlife Resources", 
            "Water Resources",
            "Agriculture",
            "Minerals and Energy Resources",
            "Manufacturing Industries",
            "Lifelines of National Economy",
            "Development",
            "Sectors of Indian Economy",
            "Money and Credit",
            "Globalization and the Indian Economy",
            "Consumer Rights"
        ]
        return topics

if __name__ == "__main__":
    # Test the RAG system
    api_key = "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"
    rag_system = GeographyRAGSystem(api_key)
    
    # Test query
    test_query = "What are the different types of resources?"
    response = rag_system.answer_question(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {response}")

