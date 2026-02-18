import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

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
}
.status-info {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────
def init_session_state():
    defaults = {"vectors": None, "embeddings": None, "docs_loaded": False}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ─── Environment Validation ──────────────────────────────────────────────────
def validate_environment():
    errors = []
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        errors.append("GROQ_API_KEY is missing from environment variables.")

    papers_dir = Path("research_papers")
    if not papers_dir.exists():
        errors.append("Directory 'research_papers/' not found. Create it and add PDF files.")
    elif not list(papers_dir.glob("*.pdf")):
        errors.append("No PDF files found in 'research_papers/'. Add at least one PDF.")

    return errors, groq_api_key


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


# ─── Vector Store Creation ───────────────────────────────────────────────────
def create_vector_embedding():
    try:
        with st.spinner("Loading embeddings model…"):
            st.session_state.embeddings = initialize_embeddings()

        with st.spinner("Loading PDF documents…"):
            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()
            if not docs:
                st.error("No documents loaded. Check your PDF files.")
                return False
            st.info(f"Loaded {len(docs)} page(s) from PDFs.")

        with st.spinner("Splitting documents into chunks…"):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs[:50])
            st.info(f"Created {len(chunks)} text chunks.")

        with st.spinner("Building vector index (may take a moment)…"):
            st.session_state.vectors = FAISS.from_documents(chunks, st.session_state.embeddings)
            st.session_state.docs_loaded = True

        return True

    except Exception as e:
        st.error(f"Error creating vector embeddings: {e}")
        st.session_state.docs_loaded = False
        return False


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Q&A</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Groq · Llama 3.1 · FAISS</p>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Setup")
        errors, groq_api_key = validate_environment()

        if errors:
            st.error("**Configuration issues found:**")
            for err in errors:
                st.write(f"❌ {err}")
            st.markdown("""
**Quick Setup:**
1. Create a `.env` file:
```
GROQ_API_KEY=your_key_here
HF_TOKEN=your_hf_token
```
2. Create `research_papers/` folder
3. Add PDF files to the folder
4. Restart the app
""")
            return

        st.markdown('<div class="status-ok">✅ Environment configured</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📄 Document Processing")

        if st.button("🔄 Load & Process Documents", use_container_width=True):
            success = create_vector_embedding()
            if success:
                st.success("✅ Vector database ready!")
            else:
                st.error("❌ Processing failed. See errors above.")

        if st.session_state.docs_loaded:
            st.markdown('<div class="status-ok">✅ Documents loaded & indexed</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-info">ℹ️ Click above to load documents</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Stack: Groq · Llama 3.1 · FAISS · HuggingFace · LangChain · Streamlit")

    # ── Query Interface ───────────────────────────────────────────────────────
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

    if search or user_prompt:
        if not st.session_state.docs_loaded:
            st.warning("⚠️ Please load and process documents first (sidebar).")
            return
        if not user_prompt:
            st.warning("⚠️ Please enter a query.")
            return

        try:
            llm = initialize_llm(groq_api_key)

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
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', '?')}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Query failed: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
