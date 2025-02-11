# RAGSwitch

A Chatbot with Retrieval-Augmented Generation (RAG) for exploring and comparing language models via Ollama. RAGSwitch leverages LangChain to process user-uploaded documents and uses vector stores to provide context during chat conversations. It supports both single-model and model-comparison modes in a sleek Streamlit interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Folder Structure](#architecture--folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Code Walkthrough](#code-walkthrough)
  - [app.py](#apppy)
  - [src/config.py](#srcconfigpy)
  - [src/utils.py](#srcutilspy)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

RAGSwitch is an experimental project that combines document retrieval with language model generation. By allowing users to upload documents (PDF, DOCX, TXT, CSV, and other unstructured files), the system creates a vector store using FAISS and HuggingFace embeddings. When a user submits a query, the application performs a similarity search to retrieve relevant document chunks and augments the prompt with context. In addition, it supports model comparison by running two separate chains side by side.

This project showcases:
- Integration of LangChain components with both Ollama and OpenAI LLMs.
- Real-time, streaming chat responses via Streamlit.
- A robust method for document ingestion and vector-based retrieval.
- Dual-model comparison for evaluating different language model performances.

---

## Features

- **Document Ingestion:** Supports multiple file formats and automatically processes them to create a searchable vector store.
- **Text Splitting & Embedding:** Uses a recursive character splitter and HuggingFace embeddings ("sentence-transformers/all-MiniLM-L6-v2") to convert documents into chunks.
- **Vector Store with FAISS:** Enables efficient similarity search for context retrieval.
- **Chat Interface:** Provides a dynamic, streaming chat UI built with Streamlit.
- **Model Comparison Mode:** Allows users to select two different models from a curated list (from Ollama) to compare their responses side by side.
- **Automatic Model Management:** Checks for model availability via the `ollama list` command and pulls missing models automatically.
- **OpenAI Integration:** OpenAI models are accessed via the cloud. Ensure your `OPENAI_API_KEY` is set (see Installation) to use models like "openai:o1" and "openai:o3-mini".

---

## Architecture & Folder Structure

The project is structured as follows:

```
├── app.py
├── ragswitchvenv
├── requirements.txt
└── src
    ├── config.py
    ├── __init__.py
    └── utils.py

```


Each module is designed to keep concerns separated:
- **app.py** orchestrates UI logic, handles model selection (supporting both Ollama and OpenAI), and streams chat responses.
- **config.py** centralizes settings such as the models list and the prompt template.
- **utils.py** handles file processing and ensures Ollama model availability. OpenAI models are cloud-hosted and require an API key.

---

## Installation

### Prerequisites

- Python 3.9 or higher.
- Git (for cloning the repository).
- [Ollama](https://ollama.com) installed and configured (for Ollama-based models).
- An OpenAI account with an API key. Set your API key in the environment variable `OPENAI_API_KEY` (required for OpenAI models).

### Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/RAGSwitch.git
   cd RAGSwitch


2. **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv ragswitchvenv

    # On macOS/Linux:
    source ragswitchvenv/bin/activate

    # On Windows:
    ragswitchvenv\Scripts\activate
    ```
3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set OpenAI API Key:**

    ```bash
    # In the .env file:
    OPENAI_API_KEY="your_openai_api_key"
    ```
---
## Usage

1. **Launch the Application:**
Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. **Upload Documents:**
Use the sidebar to upload documents (supported formats: PDF, DOCX, TXT, CSV). The documents will be processed, split into text chunks, embedded, and indexed using FAISS.
Select Models:
    - In single-model mode, choose one model from the dropdown.
    - Enable "Model Comparison" to select two models for side-by-side response comparison.
    - OpenAI models should be prefixed with ```openai:``` (e.g., ```openai:o1```, ```openai:o3-mini```, ```openai:gpt-4```).

3. **Chat Interaction:**
Type your query into the chat input. If a vector store exists from uploaded documents, the query is augmented with context extracted from similar document chunks. Responses will stream in real time.

---

## Code Walkthrough
1. **app.py**
    - Streamlit Setup: Configures page layout, initializes session state, and builds the chat UI.
    - Session Management: Manages chat history and memory using LangChain’s ```ChatMessageHistory```.
    - File Processing & Vector Store Creation: Calls the ```process_uploaded_files``` function from ```src/utils.py```.
    - Model Chain Initialization: Instantiates LLM chains based on model selection.
        - For OpenAI models (model strings starting with ```openai:```), it uses ```ChatOpenAI``` from the ```langchain_openai``` package (and checks that the ```OPENAI_API_KEY``` is set).
        - For Ollama models, it uses ```ChatOllama``` and invokes ```ensure_model_available``` to pull models if needed.
    - Chat Streaming: Streams responses from the language models, appending context when available.

2. **src/config.py**
    - Model Definitions: Provides a curated list of available models. The list now includes both Ollama models (e.g., ```phi4:14B```, ```llama3.1:8B```, etc.) and OpenAI models prefixed with ```openai:``` (e.g., ```"openai:o1"```, ```"openai:o3-mini"```, ```"openai:gpt-4"```, ```"openai:gpt-3.5-turbo"```).
    - Prompt Template: Defines the conversation structure using LangChain’s ```ChatPromptTemplate```, including system instructions and a placeholder for chat history.

3. **src/utils.py**
    - Document Loading: Uses various loaders (  e.g., ```PyPDFLoader```, ```Docx2txtLoader```) based on file type.
    - Text Splitting: Implements a ```RecursiveCharacterTextSplitter``` for dividing documents into contextually meaningful chunks.
    - Embedding & Vector Store: Generates embeddings using the HuggingFace model ```"sentence-transformers/all-MiniLM-L6-v2"``` and constructs a FAISS vector store.
    - Model Availability: Checks model availability via the ```ollama list``` command and pulls missing models automatically if needed.

---

## Dependencies
Key libraries include:
- **LangChain & Extensions:** For prompt chaining, message history, and document processing.
- **Streamlit:** For the interactive web UI.
- **FAISS (faiss-gpu-cu12)**: For efficient vector-based similarity search.
- **HuggingFaceEmbeddings:** For converting text to vector representations.
- **Ollama Integration:** Via langchain-ollama for interacting with Ollama models.
- **OpenAI Integration:** Via langchain_openai for using GPT models. Ensure your OPENAI_API_KEY is set.
- **Other libraries** as listed in requirements.txt.

---

## Troubleshooting
- Model Initialization Errors:
    - For Ollama models, ensure Ollama is installed and configured.
    - For OpenAI models, verify that the OPENAI_API_KEY environment variable is properly set.
- File Processing Issues:
Ensure that the documents are not corrupted and are in a supported format.
- Session State Resets:
Changing models mid-session will clear session data to prevent inconsistencies. This is by design—if you see unexpected resets, check your model selection.Contact

## Contact
For questions, suggestions, or contributions, please contact:

**Mohak Gangwani**
- Email: mohakmg99@gmail.com
- Website: https://mohakgangwani.onrender.com/
- LinkedIn: https://www.linkedin.com/in/mohak-gangwani/
- GitHub: https://github.com/MohakGangwani