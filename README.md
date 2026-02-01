Agentic AI RAG Chatbot 🤖

An AI Engineer interview submission implementing a RAG (Retrieval-Augmented Generation) pipeline using LangGraph, FAISS, and Streamlit. This chatbot strictly answers questions based on the provided Agentic AI eBook.

📋 Features

Strict Grounding: Answers are generated only using context retrieved from the provided PDF.

Agentic Workflow: Built with LangGraph to manage state and tool calling.

Vector Search: Uses FAISS for fast, local similarity search with HuggingFace embeddings.

Confidence Scoring: Displays a calculated confidence score for every retrieved answer.

Interactive UI: A clean Streamlit interface with session history and expandable context citations.

🛠️ Architecture

The application follows a modular RAG architecture:

Ingestion: The Ebook-Agentic-AI.pdf is loaded, split into chunks (1000 chars), and embedded using sentence-transformers/all-mpnet-base-v2.

Storage: Embeddings are stored locally in a FAISS vector index.

Graph Logic (LangGraph):

The system uses a StateGraph with a defined start and chat node.

The LLM (Groq/Llama-3) decides whether to call the retrieval tool (agentic_ai_rag_tool) or respond directly.

Retrieval: When the tool is called, it fetches the top 5 relevant chunks and calculates a confidence score based on vector distance.

🚀 Setup Instructions

Prerequisites

Python 3.9+

A Groq API Key

Installation

Clone the repository:

git clone <YOUR_REPO_LINK_HERE>
cd <YOUR_REPO_NAME>


Create a Virtual Environment:

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install Dependencies:

pip install -r requirements.txt


Configure Environment:
Create a .env file in the root directory and add your API key:

GROQ_API_KEY=your_groq_api_key_here


Add the Data:
Ensure the file Ebook-Agentic-AI.pdf is present in the root directory.

Running the Application

Run the Streamlit app:

streamlit run app.py


Note: On the first run, the system will process the PDF and create the vector database. Subsequent runs will load the existing DB.

@ Sample Queries

Here are 5 questions to test the specific knowledge base of the bot:

"What is the definition of Agentic AI?"

"How does Agentic AI differ from Generative AI?"

"What are the four levels of AI agency?"

"Explain the concept of 'Planning' in AI agents."

"What are the major challenges in implementing Agentic AI?"

📂 Project Structure

app.py - Streamlit frontend and UI logic.

rag1.py - Backend logic (LangGraph definition, FAISS setup, retrieval tool).

vectorstore/ - Directory where the FAISS index is saved locally.

requirements.txt - List of Python dependencies.

