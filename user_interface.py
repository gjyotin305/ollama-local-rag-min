import gradio as gr
from setup.llm import chat_with_ollama


gr.ChatInterface(
    title="Ollama RAG",
    description="Doc Helper",
    fn=chat_with_ollama,
    type="messages"
).launch(root_path="/gradio-demo", pwa=True)