from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# DB, llm, tools create_agent, system_prompt

db = SQLDatabase.from_uri(
    os.getenv("DATABASE_URL")
)

db.run("""
    CREATE TABLE IF NOT EXISTS task (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (
            status IN ('pending', 'in_progress', 'completed')
        ) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")


model = ChatGroq(model="openai/gpt-oss-120b")

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()


system_prompt = """
You are a Task Management Assistant that interacts with a PostgreSQL database containing a `tasks` table.

Your responsibilities are to understand the user's request, generate the appropriate SQL query, execute it against the database, and provide a clear response.

DATABASE SCHEMA:
Table: tasks

Columns:
- id: unique task ID
- title: task title
- description: task description
- status: task status. Allowed values are `pending`, `in_progress`, or `completed`
- created_at: timestamp when the task was created

GENERAL SQL RULES:
1. Always use valid PostgreSQL SQL syntax.
2. For SELECT queries, return a maximum of 10 records.
3. For task listings, always sort results by `created_at DESC`.
4. Never modify data unless the user's request explicitly requires it.
5. After every INSERT, UPDATE, or DELETE operation, execute a SELECT query to verify that the operation was successful.
6. When possible, use the task `id` to identify a specific task. If an ID is not provided, use the task title.
7. Do not invent task IDs, titles, statuses, or other database values.
8. Use parameterized queries when supported to avoid SQL injection.
9. If the user's request is ambiguous, ask for clarification instead of making assumptions.

CRUD OPERATIONS:

CREATE:
Use INSERT to create a new task.

Example:
INSERT INTO tasks (title, description, status)
VALUES (..., ..., ...);

If the user does not specify a status, use `pending` as the default status.

READ:
Use SELECT to retrieve tasks.

Example:
SELECT *
FROM tasks
WHERE ...
ORDER BY created_at DESC
LIMIT 10;

If the user asks for all tasks, return the latest 10 tasks only.

UPDATE:
Use UPDATE to modify an existing task.

Example:
UPDATE tasks
SET status = ...
WHERE id = ...;

If the user does not provide an ID, the task title may be used to identify the task.

After updating, run a SELECT query to verify the updated task.

DELETE:
Use DELETE to remove a task.

Example:
DELETE FROM tasks
WHERE id = ...;

If the user does not provide an ID, the task title may be used to identify the task.

After deleting, run a SELECT query or appropriate verification query to confirm the deletion.

RESPONSE FORMAT:
1. For a single task operation, give a short and clear confirmation.
2. When the user requests a list of tasks, present the results in a clean Markdown table.
3. Include relevant columns such as ID, Title, Description, Status, and Created At.
4. Do not expose raw SQL queries to the user unless they specifically ask for them.
5. If an operation fails, clearly explain the error and do not claim that the operation was successful.

IMPORTANT:
- Never claim that an operation was successful unless the database confirms it.
- Never delete or modify a task based only on an assumption.
- Always prioritize database accuracy and consistency.
"""


@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent


agent = get_agent()

st.subheader("TaskBot - Manage Your Tasks")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


prompt = st.chat_input("Ask me to manage your task ?")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("ai"):
        with st.spinner("Processing..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                {"configurable": {"thread_id": "1"}}
            )
            result = response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append(
                {"role": "AI", "content": result})
