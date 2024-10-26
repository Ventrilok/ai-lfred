from langchain_ollama import OllamaEmbeddings

selected_model = "mistral"
# selected_model = "gemma2:2b"


def get_embedding_function():
    embeddings = OllamaEmbeddings(model=selected_model)
    return embeddings
