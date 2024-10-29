import streamlit as st


ailfred = st.Page(
    "ailfred.py", title="Ask AI-lfred", icon=":material/chat:", default=True
)
upload = st.Page(
    "upload_file.py",
    title="Import New Document",
    icon=":material/upload:",
    default=False,
)

pg = st.navigation(
    {
        "Admin Buddy": [ailfred],
        "Tools": [upload],
    }
)

pg.run()
