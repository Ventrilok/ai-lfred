import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
from langchain_ollama.llms import OllamaLLM
from rag_helper import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
<s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments de contexte suivants pour répondre à la question. 
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.. Utilisez trois phrases maximum et soyez concis dans votre réponse. [/INST] </s> 
[INST] Question: {question} 
Context: {context} 
Answer: [/INST]
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()