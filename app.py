import streamlit as st
import os
import time
import tempfile
from pathlib import Path

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #0f172a;
    text-align: center;
    margin-bottom: 0.25rem;
}
.sub-header {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.status-ok {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
.status-info {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "vectors": None,
        "embeddings": None,
        "docs_loaded": False,
        "groq_api_key": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ─── Cached Resources ────────────────────────────────────────────────────────
@st.cache_resource
def initialize_llm(groq_api_key: str):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )


@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ─── Vector Store Creation from Uploaded Files ───────────────────────────────
def create_vector_embedding(uploaded_files):
    try:
        all_docs = []

        with st.spinner("Loading embeddings model…"):
            embeddings = initialize_embeddings()
            st.session_state.embeddings = embeddings

        with st.spinner("Reading uploaded PDFs…"):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)
                os.unlink(tmp_path)

            if not all_docs:
                st.error("No content extracted from PDFs. Check your files.")
                return False

            st.info(f"Loaded {len(all_docs)} page(s) from {len(uploaded_files)} file(s).")

        with st.spinner("Splitting into chunks…"):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)
            st.info(f"Created {len(chunks)} text chunks.")

        with st.spinner("Building vector index…"):
            st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
            st.session_state.docs_loaded = True

        return True

    except Exception as e:
        st.error(f"Error processing documents: {e}")
        st.session_state.docs_loaded = False
        return False


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    init_session_state()

    st.markdown('<h1 class="main-header">📚 RAG Document Q&A</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Groq · Llama 3.1 · FAISS</p>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ── Step 1: API Key ──
        st.subheader("Step 1 · API Key")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            value=st.session_state.groq_api_key,
            help="Get your free key at https://console.groq.com",
        )

        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
            st.markdown('<div class="status-ok">✅ API key entered</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-info">🔑 Enter your Groq API key above</div>', unsafe_allow_html=True)
            st.markdown("Get a free key at [console.groq.com](https://console.groq.com)")

        st.markdown("---")

        # ── Step 2: Upload PDFs ──
        st.subheader("Step 2 · Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Your PDFs are processed in-memory and never stored permanently.",
        )

        if uploaded_files:
            st.markdown(f'<div class="status-ok">✅ {len(uploaded_files)} file(s) ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-info">📄 Upload PDF files above</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Step 3: Process ──
        st.subheader("Step 3 · Process Documents")

        process_disabled = not groq_api_key or not uploaded_files
        if st.button("🔄 Build Vector Index", use_container_width=True, disabled=process_disabled):
            if create_vector_embedding(uploaded_files):
                st.success("✅ Index built! You can now ask questions.")
            else:
                st.error("❌ Processing failed.")

        if process_disabled:
            st.caption("Complete Steps 1 & 2 to enable processing.")

        if st.session_state.docs_loaded:
            st.markdown('<div class="status-ok">✅ Documents indexed & ready</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Stack: Groq · Llama 3.1 · FAISS · HuggingFace · LangChain · Streamlit")

    # ── Main: Query Interface ─────────────────────────────────────────────────
    if not st.session_state.groq_api_key:
        st.info("👈 Enter your Groq API key in the sidebar to get started.")
        return

    if not st.session_state.docs_loaded:
        st.info("👈 Upload PDFs and click **Build Vector Index** in the sidebar.")
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        user_prompt = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings of the research?",
        )
    with col2:
        st.write("")
        st.write("")
        search = st.button("🔍 Search", use_container_width=True)

    if search and user_prompt:
        try:
            llm = initialize_llm(st.session_state.groq_api_key)

            prompt_template = ChatPromptTemplate.from_template(
                """Answer the question based only on the provided context.
If the answer is not in the context, say "I cannot find this information in the provided documents."

Context:
{context}

Question: {input}

Answer:"""
            )

            with st.spinner("Searching documents…"):
                t0 = time.perf_counter()
                relevant_docs = st.session_state.vectors.similarity_search(user_prompt, k=5)
                context = "\n\n".join(doc.page_content for doc in relevant_docs)
                formatted = prompt_template.format(context=context, input=user_prompt)
                response = llm.invoke(formatted)
                elapsed = time.perf_counter() - t0

            st.markdown("---")
            st.subheader("💡 Answer")
            st.write(response.content)
            st.caption(f"⏱ Response time: {elapsed:.2f}s")

            with st.expander("📄 Source Chunks", expanded=False):
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Chunk {i + 1}**")
                    st.text(doc.page_content)
                    if doc.metadata:
                        st.caption(
                            f"Source: {doc.metadata.get('source', 'Uploaded file')} "
                            f"| Page: {doc.metadata.get('page', '?')}"
                        )
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Query failed: {e}")
            st.exception(e)

    elif search and not user_prompt:
        st.warning("⚠️ Please enter a question.")


if __name__ == "__main__":
    main()
