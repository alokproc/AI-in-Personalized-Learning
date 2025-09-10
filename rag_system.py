import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class GeographyRAGSystem:
    def __init__(self, api_key: str):
        """Initialize the Geography RAG System"""
        self.api_key = api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM and load vector store"""
        try:
            # Initialize Groq LLM
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="openai/gpt-oss-20b",
                temperature=0.1
            )
            
            # Load vector store if it exists
            if os.path.exists("vector_store"):
                self.vector_store = FAISS.load_local(
                    "vector_store",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Create QA chain
                self._create_qa_chain()
                
        except Exception as e:
            print(f"Error initializing components: {e}")
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        if not self.vector_store or not self.llm:
            return
        
        # Custom prompt template for geography questions
        prompt_template = """
        You are a helpful Geography tutor specializing in NCERT Class 10 Geography. 
        Use the following context to answer the question in a clear, educational manner.
        
        Context: {context}
        
        Question: {question}
        
        Instructions:
        1. Answer based on the provided context from NCERT Class 10 Geography textbook
        2. Be clear, concise, and educational
        3. If the context doesn't contain enough information, say so
        4. Use simple language suitable for Class 10 students
        5. Include relevant examples when possible
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def retrieve_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def answer_question(self, question: str) -> str:
        """Answer a geography question using RAG"""
        if not self.qa_chain:
            return "Sorry, the system is not properly initialized. Please check if the vector store exists."
        
        try:
            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            answer = result["result"]
            
            # Add source information
            source_docs = result.get("source_documents", [])
            if source_docs:
                answer += "\n\n*This answer is based on NCERT Class 10 Geography textbook content.*"
            
            return answer
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def get_topic_suggestions(self) -> List[str]:
        """Get topic suggestions for geography learning"""
        topics = [
            "Resources and Development",
            "Forest and Wildlife Resources", 
            "Water Resources",
            "Agriculture",
            "Minerals and Energy Resources",
            "Manufacturing Industries",
            "Lifelines of National Economy",
            "Renewable and Non-renewable Resources",
            "Sustainable Development",
            "Conservation of Resources",
            "Types of Agriculture",
            "Major Crops in India",
            "Industrial Development",
            "Transportation and Communication",
            "International Trade"
        ]
        return topics
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready to use"""
        return self.vector_store is not None and self.qa_chain is not None

if __name__ == "__main__":
    # Test the RAG system
    api_key = "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"
    rag_system = GeographyRAGSystem(api_key)
    
    if rag_system.is_ready():
        print("RAG system is ready!")
        
        # Test with a sample question
        question = "What are renewable resources?"
        answer = rag_system.answer_question(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    else:
        print("RAG system is not ready. Please check vector store and API key.")
