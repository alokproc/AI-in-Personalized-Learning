import os
import pickle
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(self.pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def split_text_into_chunks(self, text: str) -> List[Document]:
        """Split text into chunks for processing"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    
    def save_vector_store(self, vector_store: FAISS, save_path: str = "vector_store"):
        """Save vector store to disk"""
        try:
            vector_store.save_local(save_path)
            print(f"Vector store saved to {save_path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def load_vector_store(self, load_path: str = "vector_store") -> FAISS:
        """Load vector store from disk"""
        try:
            vector_store = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {load_path}")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def process_document(self, save_vector_store: bool = True) -> FAISS:
        """Complete document processing pipeline"""
        print("Extracting text from PDF...")
        text = self.extract_text_from_pdf()
        
        if not text:
            print("No text extracted from PDF")
            return None
        
        print("Splitting text into chunks...")
        documents = self.split_text_into_chunks(text)
        print(f"Created {len(documents)} chunks")
        
        print("Creating vector store...")
        vector_store = self.create_vector_store(documents)
        
        if vector_store and save_vector_store:
            self.save_vector_store(vector_store)
        
        self.vector_store = vector_store
        return vector_store

if __name__ == "__main__":
    # Process the NCERT Geography PDF
    processor = DocumentProcessor("Geography_AI_Tutor_RAG-based_Personalized_Learning_System.pdf")
    vector_store = processor.process_document()
    
    if vector_store:
        print("Document processing completed successfully!")
    else:
        print("Document processing failed!")

