import streamlit as st
import os
from helper import ChromaDBHelper
import constants


st.set_page_config(page_title="Admin Buddy > Upload new document")


# Ensure the 'data' directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Title of the app
st.title("Import New Document")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Define the file path to save
    file_path = os.path.join("data", uploaded_file.name)

    # Write file to the directory
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    chroma = ChromaDBHelper(
        db_path=constants.CHROMA_PATH, data_path=constants.DATA_PATH
    )
    chroma.add_document(file_path)

    st.success("File successfully ingested!")


# Divider
st.write("---")

# List PDF files in 'data' directory
st.header("PDF Files in Data Directory")

pdf_files = [file for file in os.listdir("data") if file.endswith(".pdf")]
if pdf_files:
    for file_name in pdf_files:
        file_path = os.path.join("data", file_name)
        # Display each PDF in a bordered container
        with st.container(border=True):
            st.text(file_name)
            st.link_button(
                "Open File", f"file:///Users/pb/Developer/ai-lfred/data/{file_name}"
            )

else:
    st.write("No PDF files found in the data directory.")
