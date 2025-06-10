import uuid
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_openai import ChatOpenAI
from trustcall import create_extractor
import configuration

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Developer Profile
class DevProfile(BaseModel):
    name: Optional[str] = Field(description="Developer's name")
    language: Optional[str] = Field(description="Primary programming language")
    framework: Optional[str] = Field(description="Main framework or library used")
    experience_level: Optional[str] = Field(description="Developer seniority or self-perceived level")
    prefers: Optional[str] = Field(description="Short note about preferred explanation style or coding habits")

# Architectural Decision Record (ADR)
class DecisionRecord(BaseModel):
    decision: str = Field(description="The technical or architectural decision made, stated clearly and unambiguously.")
    date: datetime = Field(description="When this decision was made or last updated.")
    rationale: str = Field(description="The reasoning or motivation behind this decision, including any trade-offs considered.")
    consequences: str = Field(description="The consequences or guidelines resulting from this decision—what should or should not be done because of it.")
    status: Literal["active", "obsolete", "rejected", "archived"] = Field(
        default="active",
        description="Current relevance of this decision for future development."
    )

# Instruction (previously InteractionDirective)
class Instruction(BaseModel):
    instruction: str = Field(
        description=(
            "An independent, atomic instruction that describes one specific way the assistant should adapt its interaction to the user. "
            "Each instruction should focus on a single behavior or style — do not combine multiple preferences into one. "
            "Instructions must be phrased as imperatives. "
            "Examples: "
            "'Explain technical topics using a step-by-step format.', "
            "'Use concise code examples.', "
            "'Avoid adding explanations unless explicitly requested.', "
            "'Include inline comments in code when requested.', "
            "'Ask if the user wants a deeper explanation after giving an answer.'"
        )
    )

# Memory update signal tool
class UpdateMemory(BaseModel):
    update_type: Literal["user", "adr", "instructions"] = Field(description="Which memory to update")

# Extractors
profile_extractor = create_extractor(model, tools=[DevProfile], tool_choice="DevProfile")
instruction_extractor = create_extractor(model, tools=[Instruction], tool_choice="Instruction", enable_inserts=True)

# Prompt templates
MODEL_SYSTEM_MESSAGE = """You are DevMentor, a helpful coding companion.
You assist a developer by tracking and respecting three types of long-term memory:
1. DevProfile — info about the developer and their work style
2. DecisionRecord — architectural or design decisions and their implications
3. Instructions — preferences for how to interact and help in the future

When responding, you must:
- Follow all active DecisionRecords (ADRs). Do not suggest actions that contradict them unless the user explicitly says the ADR is obsolete.
- Respect all Instructions. They describe how the user wants you to communicate and assist.
- Adapt your tone and suggestions to the DevProfile.

<dev_profile>
{dev_profile}
</dev_profile>

<adrs>
{adrs}
</adrs>

<preferences>
{preferences}
</preferences>

Use tool calls to update:
- DevProfile when the user mentions anything about their work style or habits
- DecisionRecord when architectural or policy decisions are discussed
- Instructions when they give feedback about how you should behave

Only confirm updates about DecisionRecord.
"""

TRUSTCALL_INSTRUCTION = """Reflect on this developer conversation. Update memory accordingly.
System Time: {time}"""

# Listener to capture tool calls
class ToolsListener:
    def __init__(self):
        self.tools = []

    def __call__(self, execution):
        runs = [execution]
        while runs:
            run = runs.pop()
            if run.child_runs:
                runs.extend(run.child_runs)
            if run.run_type == "chat_model":
                self.tools.append(
                    run.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Extract tool info from listener

def extract_tool_info(tool_calls, tool_name):
    summaries = []
    for call_batch in tool_calls:
        for call in call_batch:
            if call["name"] == tool_name:
                summaries.append(f"- {tool_name} update: {call['args']}")
    return "\n".join(summaries)

# Node: Main reasoning
def dev_mentor(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    profile = store.search(("dev_profile", user_id))
    dev_profile = profile[0].value if profile else None

    adrs = store.search(("adrs", user_id))
    adr_dump = "\n".join(f"{f.value}" for f in adrs)

    prefs = store.search(("preferences", user_id))
    preferences = "\n".join(f"- {p.value['instruction']}" for p in prefs) if prefs else ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(dev_profile=dev_profile, adrs=adr_dump, preferences=preferences)
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": [response]}

# Node: Update profile
def update_dev_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    ns = ("dev_profile", user_id)
    existing = store.search(ns)
    tool_name = "DevProfile"
    existing_mem = [(item.key, tool_name, item.value) for item in existing] if existing else None

    sys_msg = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = merge_message_runs([SystemMessage(content=sys_msg)] + state["messages"][:-1])

    result = profile_extractor.invoke({"messages": updated_messages, "existing": existing_mem})
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(ns, meta.get("json_doc_id", str(uuid.uuid4())), r.model_dump(mode="json"))

    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_calls[0]['id']}]}  

# Node: Update ADRs
def update_decision_records(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    ns = ("adrs", user_id)
    existing = store.search(ns)
    existing_mem = [(item.key, "DecisionRecord", item.value) for item in existing] if existing else None

    sys_msg = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = merge_message_runs([SystemMessage(content=sys_msg)] + state["messages"][:-1])

    listener = ToolsListener()
    extractor = create_extractor(model, tools=[DecisionRecord], tool_choice="DecisionRecord", enable_inserts=True).with_listeners(on_end=listener)
    result = extractor.invoke({"messages": updated_messages, "existing": existing_mem})

    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(ns, meta.get("json_doc_id", str(uuid.uuid4())), r.model_dump(mode="json"))

    tool_calls = state["messages"][-1].tool_calls
    summary = extract_tool_info(listener.tools, "DecisionRecord")
    return {"messages": [{"role": "tool", "content": summary, "tool_call_id": tool_calls[0]['id']}]}  

# Node: Update instructions
def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    ns = ("preferences", user_id)
    existing = store.search(ns)
    existing_mem = [(item.key, "Instruction", item.value) for item in existing] if existing else None

    sys_msg = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = merge_message_runs([SystemMessage(content=sys_msg)] + state["messages"][:-1])

    result = instruction_extractor.invoke({"messages": updated_messages, "existing": existing_mem})
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(ns, meta.get("json_doc_id", str(uuid.uuid4())), r.model_dump(mode="json"))

    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id": tool_calls[0]['id']}]}  

# Router
def route(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_decision_records", "update_instructions", "update_dev_profile"]:
    calls = state["messages"][-1].tool_calls
    if not calls:
        return END
    t = calls[0]['args']['update_type']
    if t == "user":
        return "update_dev_profile"
    elif t == "adr":
        return "update_decision_records"
    elif t == "instructions":
        return "update_instructions"
    else:
        raise ValueError

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("dev_mentor", dev_mentor)
builder.add_node("update_decision_records", update_decision_records)
builder.add_node("update_dev_profile", update_dev_profile)
builder.add_node("update_instructions", update_instructions)
builder.add_edge(START, "dev_mentor")
builder.add_conditional_edges("dev_mentor", route)
builder.add_edge("update_decision_records", "dev_mentor")
builder.add_edge("update_dev_profile", "dev_mentor")
builder.add_edge("update_instructions", "dev_mentor")

# Memory
long_term = InMemoryStore()
short_term = MemorySaver()
graph = builder.compile()