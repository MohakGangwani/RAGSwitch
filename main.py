import streamlit as st
st.set_page_config(layout="wide")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
import subprocess
import concurrent.futures

# List of available models
models = [
    "phi4:14B", "llama3.2:1B", "llama3.2:3B", "llama3.1:8B", "llama3.1:70B",
    "qwen:0.5B", "qwen:1.8B", "qwen:4B", "qwen:7B", "qwen:14B", "qwen:32B",
    "qwen2:0.5B", "qwen2:1.5B", "qwen2:7B", "qwen2.5:0.5B", "qwen2.5:1.5B",
    "qwen2.5:3B", "qwen2.5:7B", "qwen2.5:14B", "qwen2.5:32B", "mistral:7b",
    "gemma2:2B", "gemma2:9B", "gemma2:27B", "phi3.5:3.8B", "deepseek-coder-v2:16B",
    "codegemma:2B", "codegemma:7B", "command-r:35B", "deepseek-r1:1.5b",
    "deepseek-r1:7B", "deepseek-r1:8B", "deepseek-r1:14B", "deepseek-r1:32B"
]

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

# --- Initialize App ---
st.title("💬 AI Chatbot with RAG")
st.caption("🚀 A chatbot with Retrieval-Augmented Generation (RAG) using Ollama")

# Sidebar: File upload, model selection, clear button, and model change check
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv"]
    )
    comparison_mode = st.toggle("Enable Model Comparison", value=False)
    if comparison_mode:
        model_1 = st.selectbox("Choose Model 1", models, index=0)
        model_2 = st.selectbox("Choose Model 2", models, index=1)
    else:
        model_1 = st.selectbox("Choose Model", models, index=0)
        model_2 = None

    # Check if the model selection changed; if so, clear the session state.
    if "selected_model_1" in st.session_state and st.session_state.selected_model_1 != model_1:
        st.session_state.clear()
    st.session_state.selected_model_1 = model_1
    if comparison_mode:
        if "selected_model_2" in st.session_state and st.session_state.selected_model_2 != model_2:
            st.session_state.clear()
        st.session_state.selected_model_2 = model_2

    if st.button("Clear Memory and Uploaded Files"):
        st.session_state.messages = []
        st.session_state.memory = ChatMessageHistory()
        st.session_state.chain_1 = None
        st.session_state.chain_2 = None
        st.session_state.current_model_1 = None
        st.session_state.current_model_2 = None
        st.session_state.vector_store = None
        st.toast("Memory and uploaded files cleared!", icon="✅")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # For storing chat history
if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process files if uploaded
if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        process_uploaded_files(uploaded_files)
    st.toast("Files processed and vector store created!", icon="✅")

# --- Initialize Model Chains ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Respond to all the queries politely."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Initialize chain_1
if "chain_1" not in st.session_state or st.session_state.current_model_1 != model_1:
    ensure_model_available(model_1)
    st.session_state.current_model_1 = model_1
    try:
        llm_1 = ChatOllama(model=model_1)
    except Exception as e:
        st.error(f"Failed to initialize Model 1: {e}")
        st.stop()
    chain_1 = prompt_template | llm_1
    st.session_state.chain_1 = RunnableWithMessageHistory(
        chain_1,
        lambda session_id: st.session_state.memory,
        input_messages_key="input",
        history_messages_key="history",
    )

# Initialize chain_2 if in comparison mode
if comparison_mode:
    if "chain_2" not in st.session_state or st.session_state.current_model_2 != model_2:
        ensure_model_available(model_2)
        st.session_state.current_model_2 = model_2
        try:
            llm_2 = ChatOllama(model=model_2)
        except Exception as e:
            st.error(f"Failed to initialize Model 2: {e}")
            st.stop()
        chain_2 = prompt_template | llm_2
        st.session_state.chain_2 = RunnableWithMessageHistory(
            chain_2,
            lambda session_id: st.session_state.memory,
            input_messages_key="input",
            history_messages_key="history",
        )

# --- Display Chat History ---
if comparison_mode:
    # In comparison mode, each chat entry is stored as a dictionary with keys: user, model_1_response, model_2_response
    for message in st.session_state.messages:
        # Display the user message
        st.chat_message("user").markdown(message.get("user", ""))
        # Then display the responses side by side
        col1, col2 = st.columns(2)
        with col1:
            st.header(f"**{model_1}**")
            st.chat_message("assistant").markdown(message.get("model_1_response", ""))
        with col2:
            st.header(f"**{model_2}**")
            st.chat_message("assistant").markdown(message.get("model_2_response", ""))
else:
    # Single-model mode: Display messages in order (both user and assistant messages are stored)
    for message in st.session_state.messages:
        if "role" in message and "content" in message:
            st.chat_message(message["role"]).markdown(message["content"])

# --- Chat Input and Processing ---
if prompt := st.chat_input("What's up?"):
    prompt = prompt.replace('#', '\\#')
    # For single-model mode, store the user's message in the chat history
    if not comparison_mode:
        st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.memory.add_user_message(prompt)
    st.chat_message("user").markdown(prompt)

    # Prepare the augmented prompt with context if a vector store exists
    if st.session_state.vector_store:
        relevant_docs = st.session_state.vector_store.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        augmented_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
    else:
        augmented_prompt = prompt

    session_id = "user_session"  # Adjust as needed

    if comparison_mode:
        # In comparison mode, get responses from both models and display side by side
        col1, col2 = st.columns(2)
        full_response_1 = ""
        full_response_2 = ""
        try:
            with col1:
                st.header(f"**{model_1}**")
                response_placeholder_1 = st.empty()
                for chunk in st.session_state.chain_1.stream(
                    {"input": augmented_prompt},
                    config={"configurable": {"session_id": session_id}}
                ):
                    full_response_1 += chunk.content
                    response_placeholder_1.markdown(full_response_1 + "▌")
                response_placeholder_1.markdown(full_response_1)

            with col2:
                st.header(f"**{model_2}**")
                response_placeholder_2 = st.empty()
                for chunk in st.session_state.chain_2.stream(
                    {"input": augmented_prompt},
                    config={"configurable": {"session_id": session_id}}
                ):
                    full_response_2 += chunk.content
                    response_placeholder_2.markdown(full_response_2 + "▌")
                response_placeholder_2.markdown(full_response_2)
        except Exception as e:
            st.error(f"Error generating responses: {e}")
            st.stop()

        # Store the chat history as a dictionary for side-by-side display
        st.session_state.messages.append({
            "user": prompt,
            "model_1_response": full_response_1,
            "model_2_response": full_response_2
        })
        st.session_state.memory.add_ai_message(full_response_1)
        st.session_state.memory.add_ai_message(full_response_2)
    else:
        # Single-model mode: stream response in one container
        full_response = ""
        try:
            response_placeholder = st.empty()
            for chunk in st.session_state.chain_1.stream(
                {"input": augmented_prompt},
                config={"configurable": {"session_id": session_id}}
            ):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error generating response from {model_1}: {e}")
            st.stop()

        # Store the assistant's response in the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.memory.add_ai_message(full_response)
