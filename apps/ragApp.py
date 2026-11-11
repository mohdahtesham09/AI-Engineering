from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from typing import TypedDict, Annotated
import os
import streamlit as st

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

load_dotenv()


# =====================================================================
# BACKEND — your original logic, unchanged.
# Only differences: heavy objects are cached with st.cache_resource so
# Streamlit doesn't rebuild them on every rerun, and the CLI loop
# (Step 6) is replaced by the UI below.
# =====================================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")


embeddings = load_embeddings()


def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100)

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner="Indexing college documents...")
def load_retrievers():
    return (
        build_retriever("../data/academic_hand_book.pdf"),
        build_retriever("../data/Fees_Rule_Book.pdf"),
    )


academic_retriever, fee_retriever = load_retrievers()


@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


llm = load_llm()

# Step - 2 make state


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str

# Step 3 - Nodes generation


def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'. \n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic. \n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state['messages'][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answer directly using the LLM's own general knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generate the final answer, personalized using the student's programme."""
    query = state['messages'][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student."
            f"Answer this question using your own general knowledge: \n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )
    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# step 4 - router function

def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == 'fee':
        return "fee_rag"
    else:
        return "general"

# step 5 - Building the graph


@st.cache_resource
def build_app():
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    # edges

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_query,
        {
            "academic_rag": "academic_rag",
            "fee_rag": "fee_rag",
            "general": "general"
        }
    )

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


app = build_app()


# =====================================================================
# UI
# =====================================================================

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        color: #fff;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 700; color: #fff; }
    .hero p  { margin: .35rem 0 0 0; opacity: .92; font-size: .98rem; }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 600;
        letter-spacing: .02em;
        margin-bottom: .45rem;
    }
    .badge-academic { background: #e0e7ff; color: #3730a3; }
    .badge-fee      { background: #dcfce7; color: #166534; }
    .badge-general  { background: #f3f4f6; color: #374151; }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        border-radius: 10px;
        font-size: .85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BADGES = {
    "academic": ("📘 Academic Handbook", "badge-academic"),
    "fee": ("💰 Fee Rule Book", "badge-fee"),
    "general": ("💬 General", "badge-general"),
}

EXAMPLES = [
    "What is the minimum attendance required to sit in exams?",
    "How is the final grade calculated?",
    "What are the late fee charges?",
    "Is there any refund policy if I withdraw?",
    "What are the criteria for promotion to the next year?",
]

# ---- session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "programme" not in st.session_state:
    st.session_state.programme = "BCA"

# ---- sidebar ----
with st.sidebar:
    st.markdown("## 🎓 College Assistant")
    st.caption(
        "Ask about academic rules and fees, answered from the official documents.")

    st.markdown("### Your programme")
    programme = st.radio(
        "Programme",
        ["BCA", "BBA", "B.Com (H)"],
        index=["BCA", "BBA", "B.Com (H)"].index(st.session_state.programme),
        label_visibility="collapsed",
    )
    if programme != st.session_state.programme:
        st.session_state.programme = programme
        st.session_state.messages = []
        st.toast(f"Switched to {programme}", icon="🎓")

    st.markdown("### Try asking")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"ex_{i}"):
            st.session_state.pending_prompt = q

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- header ----
st.markdown(
    f"""
    <div class="hero">
        <h1>🎓 College Assistant</h1>
        <p>Hi! You're chatting as a <b>{st.session_state.programme}</b> student.
        Ask me anything about academics, rules or fees.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_message(msg: dict):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "assistant" and msg.get("category"):
            label, css = BADGES[msg["category"]]
            st.markdown(
                f'<span class="badge {css}">{label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        ctx = msg.get("context")
        if ctx and ctx != "NO_RETRIEVAL_NEEDED":
            with st.expander("📄 View source context"):
                st.text(ctx)


# ---- input handling ----
pending = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Ask about attendance, exams, fees, refunds...")
prompt = user_input or pending

# ---- chat history ----
if not st.session_state.messages and not prompt:
    st.info("👋 Pick a sample question from the sidebar or type your own below.")

for m in st.session_state.messages:
    render_message(m)

# ---- new message ----
if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({
                    "programme": st.session_state.programme,
                    "messages": [("human", prompt)]
                })
                answer = result["messages"][-1].content
                category = result.get("query_type", "general")
                context = result.get("retrieved_context", "")
            except Exception as e:
                answer = f"⚠️ Something went wrong: `{e}`"
                category, context = None, ""

    bot_msg = {
        "role": "assistant",
        "content": answer,
        "category": category,
        "context": context,
    }
    st.session_state.messages.append(bot_msg)
    st.rerun()
