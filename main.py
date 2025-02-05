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
import json
import xml.etree.ElementTree as ET
import yaml
import nbformat
import subprocess


models = [
    "phi4:14B",
    "llama3.2:1B",
    "llama3.2:3B",
    "llama3.1:8B",
    "llama3.1:70B",
    "qwen:0.5B",
    "qwen:1.8B",
    "qwen:4B",
    "qwen:7B",
    "qwen:14B",
    "qwen:32B",
    "qwen2:0.5B",
    "qwen2:1.5B",
    "qwen2:7B",
    "qwen2.5:0.5B",
    "qwen2.5:1.5B",
    "qwen2.5:3B",
    "qwen2.5:7B",
    "qwen2.5:14B",
    "qwen2.5:32B",
    "mistral:7b",
    "gemma2:2B",
    "gemma2:9B",
    "gemma2:27B",
    "qwen2.5:0.5B",
    "qwen2.5:1.5B",
    "qwen2.5:3B",
    "qwen2.5:7B",
    "qwen2.5:14B",
    "qwen2.5:32B",
    "phi3.5:3.8B",
    "deepseek-coder-v2:16B",
    "codegemma:2B",
    "codegemma:7B",
    "command-r:35B",
    "deepseek-r1:1.5b",
    "deepseek-r1:7B",
    "deepseek-r1:8B",
    "deepseek-r1:14B",
    "deepseek-r1:32B"
]

def ensure_model_available(model_name):
    """
    Ensure the specified model is available. If not, pull it using Ollama and show a spinner.
    """
    try:
        # Check if the model is available by listing all models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            with st.spinner(f"Pulling model '{model_name}'..."):
                # Start the pull process
                process = subprocess.run(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                while True:
                    line = process.stdout
                    if not line:
                        break
                    print(line)
            
            # Check if the pull was successful
            if process.returncode == 0:
                st.toast(f"Model '{model_name}' pulled successfully!", icon="✅")
            else:
                st.error(f"Failed to pull model '{model_name}'.")
                st.stop()
    except Exception as e:
        st.error(f"Failed to pull model '{model_name}': {e}")
        st.stop()




# Initialize the app
st.title("💬 AI Chatbot")
st.caption("🚀 A basic chatbot with model switching and file upload support using Ollama")
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
# Sidebar for model selection, file upload, and clear memory button
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xlsx", "jpg", "jpeg", "png", "py", "js", "java", "html", "css", "json", "xml", "md", "yaml", "yml", "ipynb", "sql", "r"]
    )
    
    # Toggle switch for single-model vs comparison mode
    comparison_mode = st.toggle("Enable Model Comparison", value=False)
    
    # Model selection based on mode
    if comparison_mode:
        model_1 = st.selectbox(
            "Choose Model 1",
            models,
            index=0
        )
        model_2 = st.selectbox(
            "Choose Model 2",
            models,
            index=1
        )
    else:
        model_1 = st.selectbox(
            "Choose Model",
            models,
            index=0
        )
        model_2 = None  # No second model in single-model mode
    
    # Add a button to clear memory and uploaded files
    if st.button("Clear Memory and Uploaded Files"):
        st.session_state.messages = []
        st.session_state.memory = ChatMessageHistory()
        st.session_state.chain_1 = None
        st.session_state.chain_2 = None
        st.session_state.current_model_1 = None
        st.session_state.current_model_2 = None
        st.toast("Memory and uploaded files cleared!", icon="✅")

# Initialize session state for chat history and chain
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()

# Initialize Model 1
if "chain_1" not in st.session_state or st.session_state.current_model_1 != model_1:
    ensure_model_available(model_1)  # Ensure the model is available
    st.session_state.current_model_1 = model_1
    try:
        llm_1 = ChatOllama(model=model_1)
    except Exception as e:
        st.error(f"Failed to initialize Model 1: {e}")
        st.stop()
    
    # Initialize chain for Model 1
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
    chain_1 = prompt_template | llm_1
    st.session_state.chain_1 = RunnableWithMessageHistory(
        chain_1,
        lambda session_id: st.session_state.memory,
        input_messages_key="input",
        history_messages_key="history",
    )

# Initialize Model 2 (only if comparison mode is enabled)
if comparison_mode:
    if "chain_2" not in st.session_state or st.session_state.current_model_2 != model_2:
        ensure_model_available(model_2)  # Ensure the model is available
        st.session_state.current_model_2 = model_2
        try:
            llm_2 = ChatOllama(model=model_2)
        except Exception as e:
            st.error(f"Failed to initialize Model 2: {e}")
            st.stop()
        
        # Initialize chain for Model 2
        chain_2 = prompt_template | llm_2
        st.session_state.chain_2 = RunnableWithMessageHistory(
            chain_2,
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
    
    elif file_type == "application/json":
        # Extract text from JSON
        data = json.load(uploaded_file)
        text = json.dumps(data, indent=2)
    
    elif file_type == "application/xml" or file_type == "text/xml":
        # Extract text from XML
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        text = ET.tostring(root, encoding="unicode")
    
    elif file_type == "text/markdown":
        # Extract text from Markdown
        text = uploaded_file.read().decode("utf-8")
    
    elif file_type in ["application/x-yaml", "text/yaml", "text/x-yaml"]:
        # Extract text from YAML
        data = yaml.safe_load(uploaded_file)
        text = yaml.dump(data, default_flow_style=False)
    
    elif file_type == "application/x-ipynb+json":
        # Extract text from Jupyter Notebook
        notebook = nbformat.read(uploaded_file, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                text += cell.source + "\n"
            elif cell.cell_type == "code":
                text += f"```python\n{cell.source}\n```\n"
    
    elif file_type in ["application/sql", "text/x-sql"]:
        # Extract text from SQL files
        text = uploaded_file.read().decode("utf-8")
    
    elif file_type in ["text/x-r", "text/x-r-script"]:
        # Extract text from R script files
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
                st.toast(f"Processed file: {uploaded_file.name}", icon="✅")  # Success toast
            else:
                st.toast(f"File {uploaded_file.name} is empty or could not be processed.", icon="⚠️")  # Warning toast
        except Exception as e:
            st.toast(f"Error processing file {uploaded_file.name}: {e}", icon="❌")  # Error toast

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("What's up?"):
    prompt = prompt.replace('#', '\#')
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.memory.add_user_message(prompt)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if comparison_mode:
        # Comparison mode: Display outputs side by side
        col1, col2 = st.columns(2)
        
        # Get Model 1 response
        with col1:
            st.header(f"**{model_1}**")
            response_placeholder_1 = st.empty()
            full_response_1 = ""
            
            # Stream the response for Model 1
            session_id = "user_session"  # You can make this dynamic if needed
            try:
                for chunk in st.session_state.chain_1.stream(
                    {"input": prompt}, config={"configurable": {"session_id": session_id}}
                ):
                    full_response_1 += chunk.content
                    response_placeholder_1.markdown(full_response_1 + "▌")
            except Exception as e:
                st.error(f"Error generating response from Model 1: {e}")
                st.stop()
            
            response_placeholder_1.markdown(full_response_1)
        
        # Get Model 2 response
        with col2:
            st.header(f"**{model_2}**")
            response_placeholder_2 = st.empty()
            full_response_2 = ""
            
            # Stream the response for Model 2
            try:
                for chunk in st.session_state.chain_2.stream(
                    {"input": prompt}, config={"configurable": {"session_id": session_id}}
                ):
                    full_response_2 += chunk.content
                    response_placeholder_2.markdown(full_response_2 + "▌")
            except Exception as e:
                st.error(f"Error generating response from Model 2: {e}")
                st.stop()
            
            response_placeholder_2.markdown(full_response_2)
        
        # Add AI responses to chat history
        st.session_state.messages.append({"role": "assistant", "content": f"{model_1}: {full_response_1}"})
        st.session_state.messages.append({"role": "assistant", "content": f"{model_2}: {full_response_2}"})
        st.session_state.memory.add_ai_message(f"{model_1}: {full_response_1}")
        st.session_state.memory.add_ai_message(f"{model_2}: {full_response_2}")
    
    else:
        # Single-model mode: Display output in a single column
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response for Model 1
        session_id = "user_session"  # You can make this dynamic if needed
        try:
            for chunk in st.session_state.chain_1.stream(
                {"input": prompt}, config={"configurable": {"session_id": session_id}}
            ):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
        except Exception as e:
            st.error(f"Error generating response from Model 1: {e}")
            st.stop()
        
        response_placeholder.markdown(full_response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": f"{model_1}: {full_response}"})
        st.session_state.memory.add_ai_message(f"{model_1}: {full_response}")