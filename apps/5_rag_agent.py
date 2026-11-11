
##! RAG Agent

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()


# ? Data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):

    # ? Load the Documents
    loader = PyPDFLoader(path)
    docs = loader.load()

    # ? Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents=docs)

    # ? Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # ? Create Agent - Tool, LLM, Prompt
    LLM = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    @tool
    def retrieve_context(query_str: str) -> str:
        """Retrieve relevant information from the health report."""

        docs = vector_db.similarity_search(query_str, k=3)

        return "\n\n".join(
            doc.page_content for doc in docs
        )

    system_prompt = """You are a helpful assistant that answers questions using retrieved context.
    My knowledge base consists of the details from the uploaded document.
    Always use the `retrieve_context` tool for questions requiring external knowledge.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=LLM,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.vector_store = vector_db
    st.session_state.document_uploaded = True


# ? Upload UI
st.title("RAG Agent")

if not st.session_state.document_uploaded:

    uploaded = st.file_uploader(
        label="Select PDF File",
        type=["pdf"],
        accept_multiple_files=False
    )

    if uploaded:

        with st.spinner("Processing..."):

            with open(uploaded.name, "wb") as f:
                f.write(uploaded.getvalue())

            process_document(uploaded.name)

        st.rerun()


# ? Chat UI
if st.session_state.document_uploaded and st.session_state.agent:

    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything to uploaded documents...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        st.chat_message("user").markdown(query)

        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}
        )

        answer = response["messages"][-1].content

        st.chat_message("assistant").markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
