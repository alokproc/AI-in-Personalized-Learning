# AI-in-Personalized-Learning
Develop AI systems that tailor educational content to individual student needs and learning styles.
# AI in Personalized Learning (Geography AI Tutor)

## 📌 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** designed to assist students in personalized learning.  
It uses **Groq LLM**, **FAISS vector database**, and a **Streamlit-based UI** to provide **curriculum-aligned answers** for NCERT Class 10 Geography.

The system tailors content to **individual student needs and learning styles**, making education more **accessible, interactive, and personalized**.

---

## 🚀 Features
- **Interactive Q&A Interface** – Students can ask questions directly from the NCERT Geography curriculum.  
- **RAG Pipeline with FAISS** – Efficient semantic search and retrieval from textbook PDFs.  
- **LLM Integration (Groq)** – Generates contextual and student-friendly answers.  
- **Document Processing** – Automatically extracts and indexes text from uploaded PDFs.  
- **Streamlit UI** – Simple and intuitive interface for students and educators.  
- **Testing Framework** – Includes test cases to validate pipeline performance.  

---

## 📂 Project Structure
```
.
├── app.py                # Streamlit app entry point
├── rag_system.py         # Core RAG system with Groq LLM + FAISS
├── document_processor.py # Handles PDF text extraction & preprocessing
├── test_system.py        # Unit tests for validation
├── requirements.txt      # Python dependencies
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/personalized-learning-ai.git
cd personalized-learning-ai
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application
```bash
streamlit run app.py
```

Once launched, open your browser at:  
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧪 Testing
Run test cases with:
```bash
pytest test_system.py
```

---

## 📖 Example Usage
- Upload the **NCERT Class 10 Geography PDF**.  
- Ask: *“Explain types of resources with examples.”*  
- The system retrieves relevant textbook sections and provides a simplified explanation.  

---

## 📌 Dependencies
Main libraries used (see `requirements.txt`):
- `langchain` & `langchain-community`  
- `langchain-groq` + `groq`  
- `streamlit`  
- `faiss-cpu`  
- `sentence-transformers`  
- `transformers`  
- `pypdf`  

---

## 🎯 Future Enhancements
- Adaptive assessments for students  
- Personalized learning paths  
- Multilingual support  
- Integration with classroom management tools  
