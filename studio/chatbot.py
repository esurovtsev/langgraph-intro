from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage

# Define LLM
# OpenAI API key configured in .env file
llm = ChatOpenAI(model="gpt-4o-mini")

# Defining Schema
##################################################################################
class SummaryState(MessagesState):
    question: str
    answer: str
    summary: str


# System message
chatbot_system_message = SystemMessage(content=("""
You are a helpful and knowledgeable chatbot assistant. 
Your goal is to provide clear and accurate answers to user questions based on the information they provide. 
Stay focused, concise, and ensure your responses are relevant to the context of the conversation. 
If you don’t have enough information, ask for clarification.”
"""))


# Nodes
def chatbot(state: SummaryState) -> SummaryState:
    summary = state.get("summary", "") # getting summary if it exists

    # If there is summary, then we add it
    if summary:
        # define summary as SystemMessage
        summary_message = SystemMessage(content=(f"""
        Summary of Conversation:

        {summary}
        """))

        messages_with_summary = [summary_message] + state["messages"]
    
    else:
        messages_with_summary = state["messages"]


    question = HumanMessage(content=state.get("question", ""))

    response = llm.invoke([chatbot_system_message] + messages_with_summary + [question]);

    return SummaryState(
        messages = [question, response],
        question = state.get("question", None),
        answer = response.content,
        summary = state.get("summary", None)
    )


def summarize(state: SummaryState) -> SummaryState:
    summary = state.get("summary", "")
    # no system message
    # the order of components is important

    if summary:
        summary_message = HumanMessage(content=(f"""
            Expand the summary below by incorporating the above conversation while preserving context, key points, and 
            user intent. Rework the summary if needed. Ensure that no critical information is lost and that the 
            conversation can continue naturally without gaps. Keep the summary concise yet informative, removing 
            unnecessary repetition while maintaining clarity.
            
            Only return the updated summary. Do not add explanations, section headers, or extra commentary.

            Existing summary:

            {summary}
            """)
        )
        
    else:
        summary_message = HumanMessage(content="""
        Summarize the above conversation while preserving full context, key points, and user intent. Your response 
        should be concise yet detailed enough to ensure seamless continuation of the discussion. Avoid redundancy, 
        maintain clarity, and retain all necessary details for future exchanges.

        Only return the summarized content. Do not add explanations, section headers, or extra commentary.
        """)

    # Add prompt to our history
    messages = state["messages"] + [summary_message]
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    return SummaryState(
        messages = delete_messages,
        question = state.get("question", None),
        answer = state.get("answer", None),
        summary = response.content
    )


# Edges

# Determine whether to end or summarize the conversation
def should_summarize(state: SummaryState):
    messages = state["messages"]
    
    if len(messages) > 2:
        return "summarize"
    
    return END


# Graph
workflow = StateGraph(SummaryState)
workflow.add_node(chatbot)
workflow.add_node(summarize)

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", should_summarize)
workflow.add_edge("summarize", END)

graph = workflow.compile()