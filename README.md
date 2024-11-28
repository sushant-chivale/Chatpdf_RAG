# ChatPDF with RAG

## Overview
This project involves developing an intelligent chatbot that uses **Langchain**, **Retrieval Augmented Generation (RAG)** technique, **Gemini-Pro LLM**, and **FAISS Vector Database** to extract insights from PDFs and articles. The chatbot is designed to deliver accurate and context-aware responses to user queries, leveraging a streamlined **Streamlit** interface for accessibility and ease of use.

## Features
- **PDF and Article Data Ingestion**: Extracts text from PDF documents and web articles.
- **Chunking for Personalized Search**: Splits extracted text into manageable chunks to facilitate efficient search.
- **Vector Database Integration**: Generates and stores embeddings in a FAISS vector database for fast and accurate similarity searches.
- **Conversational Chain**: Employs Gemini-Pro LLM with a custom prompt template for query-based response generation.
- **User-Friendly UI**: Built using Streamlit to ensure seamless interaction with the chatbot.

---

## Tech Stack
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - [Langchain](https://docs.langchain.com/)
  - [Streamlit](https://streamlit.io/)
  - [FAISS Vector Database](https://github.com/facebookresearch/faiss)
  - [Gemini-Pro LLM](https://gemini-pro.docs/)
  - [Google Generative AI Embeddings](https://cloud.google.com/genai)
  - [PyPDF2](https://pypi.org/project/PyPDF2/)

---

## File Structure
- **`helper.py`**: Contains backend functions for data processing and chatbot functionality.
- **`app.py`**: Implements the frontend using Streamlit.

---

## Functional Walkthrough

### Helper Functions (`helper.py`)

#### **1. Extracting Text from PDFs**
- Utilizes the PyPDF2 library to extract text from PDFs and combines it with text scraped from article links.
- Merges all extracted data into a unified corpus for processing.

#### **2. Creating Chunks of Text**
- Splits the extracted text into smaller chunks for efficient querying.
- Ensures personalized search by calculating similarity scores using the FAISS vector database.

#### **3. Generating and Storing Embeddings**
- Uses **Google Generative AI Embeddings** to create embeddings for the text chunks.
- Stores embeddings in a FAISS vector database for reusability, eliminating the need for repeated embedding generation.

#### **4. Chain Initialization**
- **`get_conversational_chain`**: Sets up a conversational chain using Gemini-Pro LLM with a prompt template to generate contextually rich responses.
- **`user_input`**: Handles user interaction by generating embeddings for user queries, searching the FAISS index, and providing detailed responses.

---

### UI Creation (`app.py`)
- A sleek and interactive **Streamlit** interface for users to:
  - Upload PDFs or provide article links.
  - Enter queries and receive insightful responses.
- Intuitive design ensures ease of use for both technical and non-technical users.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/chatpdf-rag.git
   cd chatpdf-rag
