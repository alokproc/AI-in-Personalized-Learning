import streamlit as st
import os
from rag_system import GeographyRAGSystem
from document_processor import DocumentProcessor

# Page configuration
st.set_page_config(
    page_title="Geography AI Tutor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E6F3FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4682B4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def setup_rag_system():
    """Setup the RAG system"""
    api_key = "your key"
    
    try:
        # Check if vector store exists
        if not os.path.exists("vector_store"):
            st.warning("Vector store not found. Processing document...")
            with st.spinner("Processing NCERT Geography PDF... This may take a few minutes."):
                processor = DocumentProcessor("Geography_AI_Tutor_RAG-based_Personalized_Learning_System.pdf")
                vector_store = processor.process_document()
                if vector_store:
                    st.success("Document processed successfully!")
                    st.session_state.vector_store_ready = True
                else:
                    st.error("Failed to process document.")
                    return None
        else:
            st.session_state.vector_store_ready = True
        
        # Initialize RAG system
        rag_system = GeographyRAGSystem(api_key)
        return rag_system
        
    except Exception as e:
        st.error(f"Error setting up RAG system: {e}")
        return None

def display_chat_history():
    """Display chat history"""
    if st.session_state.chat_history:
        st.markdown('<div class="sub-header">📚 Previous Questions & Answers</div>', unsafe_allow_html=True)
        
        for i, (question, answer) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 conversations
            with st.expander(f"Q{len(st.session_state.chat_history)-4+i}: {question[:50]}..."):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🌍 Geography AI Tutor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Your personal AI assistant for NCERT Class 10 Geography</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📖 About")
        st.markdown("""
        This AI tutor helps you learn geography using the NCERT Class 10 textbook. 
        Ask questions about:
        - Resources and Development
        - Forest and Wildlife Resources
        - Water Resources
        - Agriculture
        - Manufacturing Industries
        - And more!
        """)
        
        st.markdown("## 🎯 Sample Questions")
        sample_questions = [
            "What are renewable resources?",
            "Explain the importance of forests",
            "What is sustainable development?",
            "Types of agriculture in India",
            "What are the factors affecting agriculture?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.current_question = question
    
    # Setup RAG system if not already done
    if st.session_state.rag_system is None:
        with st.spinner("Initializing AI Tutor..."):
            st.session_state.rag_system = setup_rag_system()
    
    if st.session_state.rag_system is None:
        st.error("Failed to initialize the AI tutor. Please check the setup.")
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">💬 Ask Your Geography Question</div>', unsafe_allow_html=True)
        
        # Handle sample question selection first
        default_question = ""
        if hasattr(st.session_state, 'current_question'):
            default_question = st.session_state.current_question
            del st.session_state.current_question
        
        # Question input
        question = st.text_area(
            "Enter your question here:",
            height=100,
            placeholder="e.g., What are the different types of resources and how are they classified?",
            value=default_question,
            key="question_input"
        )
        
        # Update question variable if default_question was set
        if default_question and not question.strip():
            question = default_question
        
        col_ask, col_clear = st.columns([1, 1])
        
        with col_ask:
            ask_button = st.button("🤔 Ask Question", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process question
        if ask_button:
            if question.strip():
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Get answer from RAG system
                        answer = st.session_state.rag_system.answer_question(question)
                        
                        # Display answer
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Answer:** {answer}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
            else:
                st.warning("Please enter a question first!")
    
    with col2:
        st.markdown('<div class="sub-header">📊 Quick Stats</div>', unsafe_allow_html=True)
        
        # Display stats
        stats_container = st.container()
        with stats_container:
            st.metric("Questions Asked", len(st.session_state.chat_history))
            st.metric("Vector Store Status", "✅ Ready" if st.session_state.vector_store_ready else "❌ Not Ready")
        
        # Topic suggestions
        st.markdown('<div class="sub-header">📚 Topics Covered</div>', unsafe_allow_html=True)
        if st.session_state.rag_system:
            topics = st.session_state.rag_system.get_topic_suggestions()
            for topic in topics[:6]:  # Show first 6 topics
                st.markdown(f"• {topic}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        display_chat_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; margin-top: 2rem;">'
        '🌍 Geography AI Tutor | Powered by NCERT Class 10 Geography Textbook'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

