from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="Search the web for current and factual information.",
    checkpointer=memory
)

while True:
    query = input("user: ")
    if query.lower() == "quit":
        print("Good Bye 👋")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "fsd123"}},
    )

    print("AI:", response["messages"][-1].content)
