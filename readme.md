# 🧠 Autonomous Knowledge Agent with RAG and LLM Planning

A personal project demonstrating the power of **Large Language Models (LLMs)** combined with **Retrieval-Augmented Generation (RAG)** for advanced, grounded, and complex question-answering over local data. This agent goes beyond simple Q&A by incorporating **planning and reasoning** capabilities.

---

## ✨ Key Features

* **Local Data Ingestion:** Automatically processes various file types placed in the `data/` directory.
* **Vector Search Engine:** Leverages the **LlamaIndex** framework to efficiently build a searchable vector index.
* **Persistent Storage:** Utilizes **Chroma** as the robust, local vector database for fast and reliable retrieval.
* **Grounded Q&A (RAG):** Answers questions based *only* on the content of your private documents.
* **Advanced LLM Agent:** Implements **planning and thinking** processes (e.g., ReAct, structured thought) to handle multi-step queries and complex reasoning tasks.

---

## 🚀 Getting Started

### Prerequisites

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ducanhdt/RAG_agent
    cd RAG_agent
    ```

2.  **Set up the Python Environment:**
    Create a virtual environment and install the required dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    Get your free **Gemini API key** [here](https://aistudio.google.com/app/prompts/new_chat).

    Create your `.env` file from the template and paste your key inside:

    ```bash
    cp .env_copy .env
    # Edit the .env file and set GEMINI_API_KEY="YOUR_API_KEY"
    ```

---

## 📂 Usage

### 1. Data Preparation

Place your documents (PDFs, text files, Markdown, etc.) inside the **`data/`** folder. The agent will automatically process these files on the first run.

### 2. Simple Grounded Q&A (RAG)

This script demonstrates the core RAG pipeline: indexing your data and performing a direct, grounded Q&A.

```bash
python simple_qa.py
```

### 3. Advanced Planning Agent (The Core Project)

This script activates the more advanced agent, which uses LLM thinking to break down complex requests, decide which tool (your vector index) to use, and execute a multi-step plan to arrive at a well-reasoned answer. This is ideal for questions requiring synthesis across multiple documents or complex logic.
```bash
python simple_agent.py
```
