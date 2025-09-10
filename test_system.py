#!/usr/bin/env python3
"""
Test script for the Geography RAG system
"""

import os
import sys
from document_processor import DocumentProcessor
from rag_system import GeographyRAGSystem

def test_document_processing():
    """Test document processing functionality"""
    print("=" * 50)
    print("Testing Document Processing...")
    print("=" * 50)
    
    try:
        # Check if PDF exists
        pdf_path = "Geography_AI_Tutor_RAG-based_Personalized_Learning_System.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        print(f"✅ PDF file found: {pdf_path}")
        
        # Initialize processor
        processor = DocumentProcessor(pdf_path)
        print("✅ Document processor initialized")
        
        # Extract text
        text = processor.extract_text_from_pdf()
        if text:
            print(f"✅ Text extracted successfully ({len(text)} characters)")
        else:
            print("❌ Failed to extract text from PDF")
            return False
        
        # Split into chunks
        documents = processor.split_text_into_chunks(text)
        print(f"✅ Text split into {len(documents)} chunks")
        
        # Create vector store
        vector_store = processor.create_vector_store(documents)
        if vector_store:
            print("✅ Vector store created successfully")
            
            # Save vector store
            processor.save_vector_store(vector_store)
            print("✅ Vector store saved successfully")
            
            return True
        else:
            print("❌ Failed to create vector store")
            return False
            
    except Exception as e:
        print(f"❌ Error in document processing: {e}")
        return False

def test_rag_system():
    """Test RAG system functionality"""
    print("\n" + "=" * 50)
    print("Testing RAG System...")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        api_key = "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"
        rag_system = GeographyRAGSystem(api_key)
        print("✅ RAG system initialized")
        
        # Test queries
        test_queries = [
            "What are renewable resources?",
            "Explain the importance of forests",
            "What is sustainable development?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Question: {query}")
            
            try:
                # Test document retrieval
                relevant_docs = rag_system.retrieve_relevant_documents(query)
                print(f"✅ Retrieved {len(relevant_docs)} relevant documents")
                
                # Test answer generation
                answer = rag_system.answer_question(query)
                if answer and len(answer) > 50:  # Basic check for meaningful response
                    print(f"✅ Generated answer ({len(answer)} characters)")
                    print(f"Answer preview: {answer[:100]}...")
                else:
                    print("❌ Failed to generate meaningful answer")
                    return False
                    
            except Exception as e:
                print(f"❌ Error processing query '{query}': {e}")
                return False
        
        print("\n✅ All RAG system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in RAG system testing: {e}")
        return False

def test_topic_suggestions():
    """Test topic suggestions functionality"""
    print("\n" + "=" * 50)
    print("Testing Topic Suggestions...")
    print("=" * 50)
    
    try:
        api_key = "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"
        rag_system = GeographyRAGSystem(api_key)
        
        topics = rag_system.get_topic_suggestions()
        print(f"✅ Retrieved {len(topics)} topic suggestions:")
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing topic suggestions: {e}")
        return False

def main():
    """Main test function"""
    print("🌍 Geography RAG System - Test Suite")
    print("=" * 50)
    
    # Check if vector store already exists
    vector_store_exists = os.path.exists("vector_store")
    
    if not vector_store_exists:
        print("Vector store not found. Running document processing test...")
        if not test_document_processing():
            print("\n❌ Document processing test failed!")
            sys.exit(1)
    else:
        print("✅ Vector store already exists, skipping document processing test")
    
    # Test RAG system
    if not test_rag_system():
        print("\n❌ RAG system test failed!")
        sys.exit(1)
    
    # Test topic suggestions
    if not test_topic_suggestions():
        print("\n❌ Topic suggestions test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
    print("The Geography RAG system is ready to use.")
    print("Run 'streamlit run app.py' to start the web interface.")
    print("=" * 50)

if __name__ == "__main__":
    main()

