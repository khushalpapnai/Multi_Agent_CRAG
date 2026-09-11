# app.py
import streamlit as st
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.agents.gemini_client import GeminiClient
from src.agents.workflow import AgenticWorkflow

st.set_page_config(
    page_title="Multi-Agent RAG System",
    page_icon="assets/logo.png",
    layout="wide"
)
st.logo("assets/logo.png")

# Initialize persistent memory objects within the ephemeral Streamlit runtime
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# Secure API Key extraction via Streamlit secrets manager
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("FATAL ERROR: GEMINI_API_KEY is missing from the secrets vault. Deployment halted.")
    st.stop()

# Instantiate backend microservices
gemini_client = GeminiClient(api_key=API_KEY)
doc_processor = DocumentProcessor()
workflow = AgenticWorkflow(
    vector_store=st.session_state.vector_store,
    gemini_client=gemini_client
)

with st.sidebar:
    st.title("📄 Document Ingestion Node")
    st.markdown("Upload complex PDF architectures to compile the localized vector graph.")

    uploaded_files = st.file_uploader(
        "Secure PDF Upload",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Initialize Vectorization"):
        if uploaded_files:
            with st.spinner("Executing dense vector embedding and chunking algorithms..."):
                for uploaded_file in uploaded_files:
                    chunks = doc_processor.process_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name
                    )
                    st.session_state.vector_store.add_documents(chunks)
                st.session_state.docs_processed = True
            st.success("HNSW Graph compiled successfully.")
        else:
            st.warning("Upload payload missing.")

    st.divider()
    st.markdown("### System Telemetry & Stack")
    st.markdown("- **Cognitive Core**: Gemini 2.5 Flash")
    st.markdown("- **Vector Math**: all-MiniLM-L6-v2")
    st.markdown("- **Storage Layer**: ChromaDB (In-Memory)")
    st.markdown("- **Architecture**: CRAG State Machine")

st.title(" Autonomous Multi-Agent Document Research Engine")
st.markdown("Submit complex analytical queries. The Critic Agent will verify all retrieved context before synthesis to mathematically eliminate hallucination anomalies.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Input research parameters..."):
    if not st.session_state.docs_processed:
        st.error("Vector database is empty. Ingest documents prior to querying.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_generator = workflow.execute(prompt)
            # st.write_stream seamlessly consumes the Python generator
            full_response = st.write_stream(stream_generator)

        st.session_state.messages.append({"role": "assistant", "content": full_response})