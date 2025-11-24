
from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from inference_auth_token import get_access_token
from tools import count_words, split_sentences, uppercase


# ============================================================
# 1. State definition
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. Routing logic (same as original)
# ============================================================
def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messages found.")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


# ============================================================
# 3. Agent 1 — decides whether to call tools
# ============================================================
def text_agent(
    state: State,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str = (
        "You are an assistant that uses tools to analyze text. "
        "Use tools when needed."
    ),
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]

    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


# ============================================================
# 3b. Agent 2 — formats structured JSON output
# ============================================================
def structured_output_agent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str = (
        "You are an assistant that MUST return valid JSON only. "
        "No explanations, no prose — only a JSON object."
    ),
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]

    result = llm.invoke(messages)
    return {"messages": [result]}


# ============================================================
# 4. LLM setup
# ============================================================
access_token = get_access_token()

llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# The tools this agent can use
tools = [count_words, split_sentences, uppercase]


# ============================================================
# 5. Build the multi-agent graph
# ============================================================
graph_builder = StateGraph(State)

graph_builder.add_node(
    "text_agent",
    lambda state: text_agent(state, llm=llm, tools=tools),
)
graph_builder.add_node(
    "structured_output_agent",
    lambda state: structured_output_agent(state, llm=llm),
)

tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "text_agent")

graph_builder.add_conditional_edges(
    "text_agent", route_tools, {"tools": "tools", "done": "structured_output_agent"}
)

graph_builder.add_edge("tools", "text_agent")
graph_builder.add_edge("structured_output_agent", END)

graph = graph_builder.compile()


# ============================================================
# 6. Run the graph (homework prompt)
# ============================================================
prompt = (
    "Analyze the following text using your tools, then return a JSON summary:\n"
    "\"AI is evolving rapidly. Students are learning AI. Many tools exist.\""
)

for chunk in graph.stream({"messages": prompt}, stream_mode="values"):
    new_message = chunk["messages"][-1]
    new_message.pretty_print()

