from langchain_ollama import OllamaEmbeddings

selected_model = "mistral:latest"


def get_embedding_function():
    embeddings = OllamaEmbeddings(model=selected_model)
    return embeddings
