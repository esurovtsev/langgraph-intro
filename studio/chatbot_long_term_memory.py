from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import SystemMessage
from langgraph.store.memory import InMemoryStore
import configuration


model = ChatOpenAI(model="gpt-4o-mini")

### Nodes

def chat(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Retrieve memory from the store
    user_details = store.get(("memory", user_id), "user_details")

    # Extract the actual memory content if it exists and add a prefix
    if user_details:
        # Value is a dictionary with a memory key
        user_details_content = user_details.value.get('memory')
    else:
        user_details_content = "No existing details found."

    # Format the memory in the system prompt
    system_msg = f"""
    You are a helpful assistant with memory capabilities.
    If user-specific memory is available, use it to personalize 
    your responses based on what you know about the user.
    
    Your goal is to provide relevant, friendly, and tailored 
    assistance that reflects the user’s preferences, context, and past interactions.

    If the user’s name or relevant personal context is available, always personalize your responses by:
        – Addressing the user by name (e.g., "Sure, Bob...") when appropriate
        – Referencing known projects, tools, or preferences (e.g., "your MCP  server typescript based project")
        – Adjusting the tone to feel friendly, natural, and directly aimed at the user

    Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..." say "Since your project is built with TypeScript..."

    Use personalization especially in:
        – Greetings and transitions
        – Help or guidance tailored to tools and frameworks the user uses
        – Follow-up messages that continue from past context

    Always ensure that personalization is based only on known user details and not assumed.
    
    The user’s memory (which may be empty) is provided as: {user_details_content}
    """
    
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": response}


def update_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    namespace = ("memory", user_id)
    key = "user_details"
    user_details = store.get(namespace, key)
        
    if user_details:
        user_details_content = user_details.value.get('memory')
    else:
        user_details_content = "No existing details found."

    # Format the memory in the system prompt
    system_msg = f"""
    You are responsible for updating and maintaining accurate user memory to enable personalized responses.

    CURRENT USER DETAILS:
    {user_details_content}

    INSTRUCTIONS:
    1. Carefully review the chat history below.
    2. Identify any new, explicitly stated user information, such as:
        - Personal details (e.g., name, location)
        - Preferences (likes, dislikes)
        - Interests and hobbies
        - Experiences or background
        - Goals and future plans
    3. If no new information is present, do not output anything.
    4. If new information is found:
        - Merge it with the existing memory
        - Format the updated memory as a clear, bulleted list
        - Include only factual, user-stated details
    5. If new information contradicts existing memory, keep the most recent version stated by the user.

    Important:
    - Do NOT include summaries like "no update needed".
    - ONLY return output when actual new user information is added.

    Your final output should either be a clean, updated bulleted list — or nothing at all.
    """
    
    new_memory = model.invoke([SystemMessage(content=system_msg)] + state['messages'])

    if new_memory.content.strip():
        store.put(namespace, key, {"memory": new_memory.content})


# Define the graph
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_node("update_memory", update_memory)
builder.add_edge(START, "chat")
builder.add_edge("chat", "update_memory")
builder.add_edge("update_memory", END)

long_term_memory = InMemoryStore()
short_term_memory = MemorySaver()

graph = builder.compile()