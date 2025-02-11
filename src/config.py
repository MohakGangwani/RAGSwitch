from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# List of available models (both OpenAI and OLlama)
models = models = [
    "OpenAI(o1)", "OpenAI(o3-mini)", "OpenAI(gpt-4)", "OpenAI(gpt-3.5-turbo)",
    "Codegemma(2B)", "Codegemma(7B)", "command-r(35B)", "DeepSeek-coder-v2(16B)",
    "DeepSeek-r1(1.5B)", "DeepSeek-r1(7B)", "DeepSeek-r1(8B)", "DeepSeek-r1(14B)",
    "DeepSeek-r1(32B)", "Gemma2(2B)", "Gemma2(9B)", "Gemma2(27B)", "Llama3.1(8B)",
    "Llama3.1(70B)", "Llama3.2(1B)", "Llama3.2(3B)", "Mistral(7b)", "Phi3.5(3.8B)",
    "Phi4(14B)", "Qwen(0.5B)", "Qwen(1.8B)", "Qwen(4B)", "Qwen(7B)", "Qwen(14B)",
    "Qwen(32B)", "Qwen2(0.5B)", "Qwen2(1.5B)", "Qwen2(7B)", "Qwen2.5(0.5B)",
    "Qwen2.5(1.5B)", "Qwen2.5(3B)", "Qwen2.5(7B)", "Qwen2.5(14B)", "Qwen2.5(32B)"
]


# --- Initialize Model Chains ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and informative assistant, always strive to provide accurate and comprehensive answers to user questions in a friendly and polite manner. Respond in a clear and concise way, and if you are unsure of the answer, politely ask for clarification."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
