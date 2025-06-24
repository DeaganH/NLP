import streamlit as st
from chat_client.openai import OpenAIChatClient
from chat_client.chat_context import manage_history
import os
import docx
import PyPDF2

from rag_tools.vector_store import DocumentVectorStore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAIChatClient(api_key=OPENAI_API_KEY)
context_length = 10 # Limit to last n messages for performance and cost management

st.title("ChatGPT-like clone")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "docx"],
        help="Supported formats: .txt, .pdf, .docx"
    )

    # Use session state to avoid reprocessing and to detect new uploads
    if "doc_uploaded" not in st.session_state:
        st.session_state["doc_uploaded"] = False
    if "last_file_id" not in st.session_state:
        st.session_state["last_file_id"] = None

    loaded_text = ""
    if uploaded_file is not None:
        # Create a unique identifier for the uploaded file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        # Detect if a new file is uploaded
        if st.session_state["last_file_id"] != file_id:
            # New file detected, clear previous state
            st.session_state["doc_uploaded"] = False
            st.session_state["doc_store"] = None
            st.session_state["messages"] = []
            st.session_state["last_file_id"] = file_id

        if not st.session_state["doc_uploaded"]:
            try:
                file_type = uploaded_file.type
                if file_type == "text/plain":
                    loaded_text = uploaded_file.read().decode("utf-8")
                elif file_type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    loaded_text = ""
                    for page in pdf_reader.pages:
                        loaded_text += page.extract_text() or ""
                elif file_type in [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/msword"
                ]:
                    doc = docx.Document(uploaded_file)
                    loaded_text = "\n".join([para.text for para in doc.paragraphs])

                if loaded_text.strip() == "":
                    st.warning("The document is empty or could not be read.")
                    loaded_text = ""
                else:
                    doc_store = DocumentVectorStore(chunk_size=500)
                    doc_store.vectorize(loaded_text)
                    st.session_state["doc_store"] = doc_store
                    st.session_state["loaded_text"] = loaded_text
                    doc_description = client.get_document_description(loaded_text)

                    # Add document description as a system message in chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Document description:\n{doc_description}"
                    })
                    st.session_state["doc_uploaded"] = True  # Mark as processed

            except Exception as e:
                st.error(f"Failed to load document: {e}")
                loaded_text = ""

            st.success("Document loaded!")
            st.text_area("Loaded Text", st.session_state["loaded_text"], height=200)
        else:
            st.text_area("Loaded Text", st.session_state["loaded_text"], height=200)
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    message, function_call = client.chat_with_function_calling(
            messages=manage_history(st.session_state.messages, context_length),
            doc_store=st.session_state.get("doc_store", None)
        )
    if function_call:
        # If a function call is made, handle it
        st.session_state.messages.extend(message)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        stream = client.chat_stream(
            messages=manage_history(st.session_state.messages, context_length) 
        )

        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})