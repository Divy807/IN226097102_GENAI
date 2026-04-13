🚀 LangChain Deep Dive (Free Version - No API Key Required)

This repository contains a complete hands-on implementation of LangChain concepts using a free HuggingFace model (GPT-2) instead of paid APIs.

It is designed to demonstrate how to build LLM-powered applications with modular components like prompts, chains, memory, tools, and vector stores.

📌 Overview

LangChain is a framework that helps developers build applications using Large Language Models (LLMs) by connecting them with:

Prompts
Memory
Tools
External data sources

This project focuses on:

Conceptual understanding
Practical implementation
Running everything without API keys
🎯 Objectives
Understand LangChain architecture
Build modular LLM pipelines
Implement chains, memory, and tools
Work with document loaders and vector stores
Avoid API dependency using free models
🧠 Key Concepts Covered
LLMs (HuggingFace GPT-2)
Prompt Templates
Chains (LCEL)
Memory (manual implementation)
Tools (custom functions)
Document Loaders
Vector Stores (FAISS)
🏗️ Architecture
User Input
   ↓
Prompt Template
   ↓
LLM (HuggingFace)
   ↓
Chain
   ↓
Tool (if needed)
   ↓
Final Output
📂 Project Structure
langchain-project/
│
├── notebook.ipynb        # Main implementation
├── data.txt              # Sample document for loader
├── requirements.txt      # Dependencies
└── README.md             # Documentation
⚙️ Installation
1. Clone Repository
git clone https://github.com/your-username/langchain-project.git
cd langchain-project
2. Install Dependencies
pip install -r requirements.txt
📦 Requirements
langchain
langchain-community
transformers
faiss-cpu
💻 Code Examples
🔹 Initialize Free LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline("text-generation", model="gpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

print(llm.invoke("What is LangChain?"))
🔹 Prompt + Chain
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Explain {topic} in simple terms"
)

chain = template | llm

print(chain.invoke({"topic": "Artificial Intelligence"}))
🔹 Memory (Manual)
history = []

def chat(user_input):
    history.append(f"User: {user_input}")
    prompt = "\n".join(history) + "\nAI:"
    
    response = llm.invoke(prompt)
    history.append(f"AI: {response}")
    
    return response

print(chat("Hi, I am Divya"))
print(chat("What is my name?"))
🔹 Vector Store (FAISS)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

vectorstore = FAISS.from_texts(
    ["LangChain is powerful", "AI is the future"],
    embedding=embeddings
)

print(vectorstore.similarity_search("What is AI?"))
🌍 Real-World Use Cases
🤖 AI Chatbot
Uses memory for conversation
Maintains context
📄 Document Q&A
Loads external data
Uses vector search
🧠 AI Automation
Uses tools for tasks like calculations
Simulates agent behavior
✅ Advantages
No API key required
Fully runnable locally
Modular design
Easy to understand and extend
⚠️ Limitations
GPT-2 is less powerful than modern LLMs
Weak reasoning and repetition issues
Not suitable for production
🚫 When NOT to Use This Setup
Production-grade applications
Complex reasoning tasks
Real-time or high-accuracy systems
🔮 Future Scope
Upgrade to GPT-4 / Claude
Implement RAG (Retrieval-Augmented Generation)
Use LangGraph for multi-agent workflows
💡 Important Note

This project uses a free HuggingFace model (GPT-2) for demonstration purposes.
In real-world applications, more advanced models like GPT-4 provide significantly better performance.

🤝 Contributing

Feel free to fork the repo and improve it. Contributions are welcome!

📜 License

This project is open-source and available under the MIT License.
