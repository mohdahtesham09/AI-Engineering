

#! Project Structure:
"""
?LLM 
?Tool - Google Search Tool
?Agent
?Memory
?Streaming
?Web Interface
"""
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b", streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]


if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are a amazing ai agent can search on google as well"

)

print(st.session_state.memory)
# * Building Web Interface
st.subheader("💬 QuickAnswer - Answer at the speed of throught ")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything ?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": "query"})

    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages"
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)

    st.session_state.history.append({"role": "ai", "content": message})
