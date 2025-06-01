from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import SystemMessage
from langgraph.store.memory import InMemoryStore
import configuration
from trustcall import create_extractor
import uuid


# Memory schema
class InteractionDirective(BaseModel):
    directive: str = Field(
        description=(
            "An independent, atomic instruction that describes one specific way the assistant should adapt its interaction to the user. "
            "Each directive should focus on a single behavior or style — do not combine multiple preferences into one. "
            "Directives must be phrased as imperatives. "
            "Examples: "
            "'Explain technical topics using a step-by-step format.', "
            "'Use concise code examples.', "
            "'Avoid adding explanations unless explicitly requested.', "
            "'Include inline comments in code when requested.', "
            "'Ask if the user wants a deeper explanation after giving an answer.'"
        )
    )

model = ChatOpenAI(model="gpt-4o-mini")

trustcall_extractor = create_extractor(
    model,
    tools=[InteractionDirective],
    tool_choice="InteractionDirective",
    enable_inserts=True
)

### Nodes

def chat(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace_for_directives = (user_id, "directives")

    # Retrieve directives from the store
    directives = store.search(namespace_for_directives)
    if directives:
        directive_lines = "\n".join(f"– {d.value['directive']}" for d in directives)
        directives_section = f"When answering, use the following guiding directives:\n{directive_lines}"
    else:
        directives_section = ""

    # Format the memory in the system prompt
    system_msg = f"""
    You are a helpful assistant. Provide clear, relevant, and user-centered answers.

    {directives_section}
    """.strip()
    print(f"[CHAT NODE] system_msg: {system_msg}")
    
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": response}


def update_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace_for_directives = (user_id, "directives")

    # prepare existing facts
    schema = "InteractionDirective"
    existing_directives = store.search(namespace_for_directives)
    memories = (
        [(existing_directive.key, schema, existing_directive.value) for existing_directive in existing_directives] 
        if existing_directives else None
    )

    system_msg = """
    Update existing directives about the user and create new ones based on the following conversation, 
    ensuring each directive reflects one specific communication preference or behavior the assistant should follow.
    Avoid simple rephrasing of existing directives.
    """

    user_messages = [msg for msg in state["messages"] if msg.type == "human"][-3:]

    result = trustcall_extractor.invoke({
        "messages": [system_msg] + user_messages, 
        "existing": memories
    })
    print(f"[UPDATE_MEMORY NODE] response_metadata: {result['response_metadata']}")

    # update directives
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace_for_directives, 
            rmeta.get("json_doc_id", str(uuid.uuid4())), 
            r.model_dump(mode="json")
        )


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