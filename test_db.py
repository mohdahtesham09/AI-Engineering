from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
import os

load_dotenv()

db = SQLDatabase.from_uri(
    os.getenv("DATABASE_URL")
)

print("✅ PostgreSQL connected successfully!")

print("Databases tables:")
print(db.get_usable_table_names())
