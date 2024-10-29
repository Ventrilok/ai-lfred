from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from helper import ChromaDBHelper
from rag_helper import get_embedding_function
import streamlit as st
import constants


chroma = ChromaDBHelper(db_path=constants.CHROMA_PATH, data_path=constants.DATA_PATH)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=constants.CHROMA_PATH, embedding_function=embedding_function
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=7)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(constants.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text, sources


st.set_page_config(page_title="Admin Buddy > Ask AI-lfred")


def rebuild_database():
    chroma.reset_database()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.session_state.messages = [
    {
        "role": "assistant",
        "content": "Bonjour Maître Bruce, que puis-je faire pour vous ?",
    }
]

# React to user input
if prompt := st.chat_input("Bonjour Maître Bruce."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Un instant je vous prie, je consulte vos documents..."):
        response, sources = query_rag(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(f"{response}")
        st.markdown(f"Sources: {sources}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
