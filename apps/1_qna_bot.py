import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

st.title("🤖 AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Gemini!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything?")

if query:
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)

    # AI response
    res = llm.invoke(query)

    answer = res.content[0]["text"]

    # Display only actual text
    st.chat_message("ai").markdown(answer)

    # Save only actual text
    st.session_state.messages.append({
        "role": "ai",
        "content": answer
    })

# while True:
#     query = input("User: ")

#     if query.lower() in ["quit", "exit", "bye"]:
#         print("GoodBye 👋")
#         break

#     res = llm.invoke(query)
#     print("AI: ", res.content, "\n")
