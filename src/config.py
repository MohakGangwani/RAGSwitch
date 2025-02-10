from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# List of available models (both OpenAI and Ollama)
models = models = [
    "openai:o1", "openai:o3-mini", "openai:gpt-4", "openai:gpt-3.5-turbo",
    "codegemma:2B", "codegemma:7B", "command-r:35B", "deepseek-coder-v2:16B",
    "deepseek-r1:1.5b", "deepseek-r1:7B", "deepseek-r1:8B", "deepseek-r1:14B",
    "deepseek-r1:32B", "gemma2:2B", "gemma2:9B", "gemma2:27B", "llama3.1:8B",
    "llama3.1:70B", "llama3.2:1B", "llama3.2:3B", "mistral:7b", "phi3.5:3.8B",
    "phi4:14B", "qwen:0.5B", "qwen:1.8B", "qwen:4B", "qwen:7B", "qwen:14B",
    "qwen:32B", "qwen2:0.5B", "qwen2:1.5B", "qwen2:7B", "qwen2.5:0.5B",
    "qwen2.5:1.5B", "qwen2.5:3B", "qwen2.5:7B", "qwen2.5:14B", "qwen2.5:32B"
]


# --- Initialize Model Chains ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a hiring manager. Respond to all user queries within 2-3 lines and be completely honest. For your context - we're in Feb, 2025 right now."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
