# Streamlit RAG Chat App

A simple, extensible ChatGPT-like application built with [Streamlit](https://streamlit.io/) that supports Retrieval-Augmented Generation (RAG) over uploaded documents. Users can upload `.txt`, `.pdf`, or `.docx` files, which are then vectorized for semantic search and context-aware chat using OpenAI's API.

---

## Features

- **ChatGPT-like interface**: Chat with an AI assistant powered by OpenAI.
- **Document upload**: Upload and process `.txt`, `.pdf`, or `.docx` files.
- **Vector store**: Uploaded documents are chunked and vectorized for similarity search.
- **Retrieval-Augmented Generation (RAG)**: The assistant can search your uploaded document for relevant information as needed.
- **Automatic document description**: The AI summarizes your uploaded document for context.
- **Session management**: Detects new uploads and resets state as needed.

---

## Getting Started

### Prerequisites

- Python 3.8+
- [OpenAI API key](https://platform.openai.com/account/api-keys)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/streamlit-rag-chat.git
    cd streamlit-rag-chat
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    Make sure your `requirements.txt` includes:
    ```
    streamlit
    openai
    python-docx
    PyPDF2
    scikit-learn
    numpy
    ```

3. **Set your OpenAI API key:**
    - You can set it as an environment variable:
      ```bash
      set OPENAI_API_KEY=sk-...
      ```
    - Or use a `.env` file and [python-dotenv](https://pypi.org/project/python-dotenv/).

---

## Usage

1. **Run the app:**
    ```bash
    streamlit run app/chat_app.py
    ```

2. **Upload a document** using the sidebar.

3. **Chat** with the assistant in the main window. The assistant will use the uploaded document for context and can perform semantic search when needed.

---

## Project Structure

```
NLP/streamlit_rag_chat/
├── app/
│   └── chat_app.py
├── chat_client/
│   ├── openai.py
│   └── chat_context.py
├── rag_tools/
│   └── vector_store.py
└── requirements.txt
```

---

## Customization

- **Vector Store**: Modify `rag_tools/vector_store.py` to change chunking or vectorization logic.
- **OpenAI Client**: Extend `chat_client/openai.py` to add more advanced function-calling or prompt engineering.
- **Chat Context**: Adjust `chat_client/chat_context.py` for custom chat history management.

---

## License

MIT License

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [scikit-learn](https://scikit-learn.org/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)