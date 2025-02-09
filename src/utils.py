import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
import subprocess


# --- File Processing Function ---
def process_uploaded_files(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Choose loader based on file type
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(tmp_file_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_file_path)
        elif uploaded_file.type == "text/csv":
            loader = CSVLoader(tmp_file_path)
        else:
            loader = UnstructuredFileLoader(tmp_file_path)

        documents.extend(loader.load())
        os.remove(tmp_file_path)  # Clean up temporary file

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

# --- Ensure Model is Available ---
def ensure_model_available(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            with st.spinner(f"Pulling model '{model_name}'... This may take a few minutes"):
                process = subprocess.run(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            if process.returncode == 0:
                st.toast(f"Model '{model_name}' pulled successfully!", icon="✅")
            else:
                st.error(f"Failed to pull model '{model_name}'.")
                st.stop()
    except Exception as e:
        st.error(f"Failed to pull model '{model_name}': {e}")
        st.stop()
