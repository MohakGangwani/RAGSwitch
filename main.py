import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_ollama import ChatOllama
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from PIL import Image
import pytesseract
import io
import os

# Initialize the app
st.title("💬 AI Chatbot")
st.caption("🚀 A basic chatbot with model switching using Ollama")

# Sidebar for model selection and file upload
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV, Excel, Images, Code)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xlsx", "jpg", "jpeg", "png", "py", "js", "java", "html", "css"]
    )
    model_name = st.selectbox(
        "Choose LLM Model",
        ["llama3.2", "mistral", "orca2", "phi", "dolphin-mistral"],
        index=0
    )

# Initialize session state for chat history and chain
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()

if "chain" not in st.session_state or st.session_state.current_model != model_name:
    st.session_state.current_model = model_name
    try:
        llm = ChatOllama(model=model_name)
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
        st.stop()
    
    # Initialize prompt template and chain
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                'You are a helpful AI assistant. Respond to all the queries politely.',
            ),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}'),
        ]
    )
    
    chain = prompt_template | llm
    st.session_state.chain = RunnableWithMessageHistory(
        chain,
        lambda session_id: st.session_state.memory,
        input_messages_key="input",
        history_messages_key="history",
    )

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    text = ""
    
    if file_type == "application/pdf":
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Extract text from DOCX
        doc = Document(io.BytesIO(uploaded_file.read()))
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    
    elif file_type == "text/plain":
        # Extract text from TXT
        text = uploaded_file.read().decode("utf-8")
    
    elif file_type == "text/csv":
        # Extract text from CSV
        df = pd.read_csv(uploaded_file)
        text = df.to_string()
    
    elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        # Extract text from Excel
        df = pd.read_excel(uploaded_file)
        text = df.to_string()
    
    elif file_type in ["image/jpeg", "image/png", "image/jpg"]:
        # Extract text from images using OCR
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
    
    elif file_type in ["text/x-python", "application/javascript", "text/x-java", "text/html", "text/css"]:
        # Extract text from code files
        text = uploaded_file.read().decode("utf-8")
    
    return text

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            file_text = extract_text_from_file(uploaded_file)
            if file_text.strip():  # Only add non-empty text
                st.session_state.memory.add_user_message(f"Uploaded file: {uploaded_file.name}")
                st.session_state.memory.add_ai_message(f"File content: {file_text}")
                st.success(f"Processed file: {uploaded_file.name}")
            else:
                st.warning(f"File {uploaded_file.name} is empty or could not be processed.")
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.memory.add_user_message(prompt)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        session_id = "user_session"  # You can make this dynamic if needed
        try:
            for chunk in st.session_state.chain.stream(
                {"input": prompt}, config={"configurable": {"session_id": session_id}}
            ):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.stop()
        
        response_placeholder.markdown(full_response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.memory.add_ai_message(full_response)