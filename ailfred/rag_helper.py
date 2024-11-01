from langchain_ollama import OllamaEmbeddings

selected_model = "llama3.2"  # "mistral"


def get_embedding_function():
    embeddings = OllamaEmbeddings(model=selected_model)
    return embeddings
