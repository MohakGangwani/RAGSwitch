import streamlit as st
st.set_page_config(layout="wide")
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama
from src.config import *
from src.utils import *

# --- Initialize App ---
st.title("💬 RAGSwitch")
st.caption("🚀 A chatbot with Retrieval-Augmented Generation (RAG)")

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

if model_1.startswith("openai:") or (model_2 and model_2.startswith("openai:")):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable to use OpenAI models.")
        st.stop()

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

# --- Initialize chain_1 ---
if "chain_1" not in st.session_state or st.session_state.current_model_1 != model_1:
    # Only ensure availability if using Ollama (OpenAI models are cloud-hosted)
    if not model_1.startswith("openai:"):
        ensure_model_available(model_1)
    st.session_state.current_model_1 = model_1
    try:
        if model_1.startswith("openai:"):
            from langchain_openai import ChatOpenAI
            # Extract the OpenAI model name after the "openai:" prefix
            openai_model = model_1.split("openai:")[1]
            llm_1 = ChatOpenAI(model_name=openai_model, temperature=0)
        else:
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

# --- Initialize chain_2 if in comparison mode ---
if comparison_mode:
    if "chain_2" not in st.session_state or st.session_state.current_model_2 != model_2:
        if not model_2.startswith("openai:"):
            ensure_model_available(model_2)
        st.session_state.current_model_2 = model_2
        try:
            if model_2.startswith("openai:"):
                from langchain_openai import ChatOpenAI
                openai_model = model_2.split("openai:")[1]
                llm_2 = ChatOpenAI(model_name=openai_model, temperature=0)
            else:
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
