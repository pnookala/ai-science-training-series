from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Import your tools
from tools import find_word

# Import token helper
from inference_auth_token import get_access_token

# Get ALCF access token
access_token = get_access_token()

# Initialize LLM from ALCF inference endpoint
llm = ChatOpenAI(
        model_name="openai/gpt-oss-120b",
        api_key=access_token,
        base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
        temperature=0,
    )

# Build ReAct agent with your custom tools
tools = [find_word]
agent = create_agent(llm, tools=tools)

# PROMPT: tell agent to use tools!
prompt = """Use your tools to solve this:
    1. Count how many times the word 'AI' appears in this text:
        "AI is amazing. Many students are learning AI because AI will shape the future."
    """

print("=== Agent Output ===")
for chunk in agent.stream(
            {"messages": prompt},
            stream_mode="values",
):
    message = chunk["messages"][-1]
    message.pretty_print()

