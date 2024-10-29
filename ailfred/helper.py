import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag_helper import get_embedding_function
from langchain_chroma import Chroma


def load_documents(path: str):
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


class ChromaDBHelper:
    def __init__(self, db_path: str, data_path: str):
        self.db_path = db_path
        self.data_path = data_path

    def populate_database(self):
        # Create (or update) the data store.
        documents = load_documents(self.data_path)
        chunks = split_documents(documents)
        self.add_documents(chunks)

    def add_document(self, path):
        document_loader = PyPDFLoader(file_path=path)
        document = document_loader.load()
        chunks = split_documents(document)
        self.add_documents(chunks)

    def add_documents(self, chunks: list[Document]):
        db = Chroma(
            persist_directory=self.db_path, embedding_function=get_embedding_function()
        )  # Load the existing database.

        # Calculate Page IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")

    def reset_database(self):
        self.drop_database()
        self.populate_database()

    def drop_database(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
