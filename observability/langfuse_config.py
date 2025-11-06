import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler 
from dotenv import load_dotenv

load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    # Add an explicit timeout (in seconds)
    timeout=30
)

# Note: Renamed to `LANGFUSE_HANDLER` for clarity on its purpose.
LANGFUSE_HANDLER = CallbackHandler() 

def get_langfuse_handler():
    """Returns the globally initialized Langfuse callback handler."""
    return LANGFUSE_HANDLER