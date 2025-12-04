from langchain.agents import create_agent
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = MultiServerMCPClient(
    {
        "playwright": {
            "transport": "streamable_http",
            "url": "http://localhost:8931/mcp"
    }
    }
)

# Lazy model initialization
_model = None

def get_model():
    """Lazily initialize the chat model only when needed."""
    global _model
    if _model is None:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        _model = init_chat_model("google_genai:gemini-2.5-flash")
    return _model


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def multiply_tool(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

def addition_tool(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def get_all_users(msg: str):
    # Logic to fetch users from the database
    # ...
    all_messages = []
    # Run the agent
    async with client.session("playwright") as session:
        tools = await load_mcp_tools(session)
    # tools = await client.get_tools()
        agent = create_agent(
        get_model(),
        tools= tools,
        system_prompt = """You are a helpful assistant with expertise in web automation.

CRITICAL RULES for web page interactions:
1. ALWAYS start by taking a fresh page snapshot using bmp_screenshot
2. Before ANY click, type, or interaction - ALWAYS take a new screenshot first
3. Always use the MOST RECENT screenshot references for interactions
4. If you receive a 'Ref not found' error:
   - Take a new screenshot immediately
   - Re-evaluate the current page state
   - Get fresh element references
   - Retry the action with new references
5. Never reuse old element references - always get fresh ones from the latest snapshot
6. Take screenshots between major page state changes
7. When navigating to new pages, take a screenshot before interacting
8. If any tool fails, take a fresh screenshot and analyze what changed""",
)

        res = await agent.ainvoke(
        {"messages": [{"role": "user", "content": msg}]}
        )
    
    # extract messages list
        msgs = res.get("messages", []) if isinstance(res, dict) else getattr(res, "messages", [])
    
    # collect all messages with necessary data
       
    
        for m in msgs:
        # Extract content and role from message objects
            if hasattr(m, "content"):
                content = m.content
                role = getattr(m, "type", "unknown")
            else:
                content = m.get("content", "")
                role = m.get("role", "unknown")
        
        # Only include messages with content
            if content:
                all_messages.append({
                "role": role,
                "content": content
                })
    
    return [{
        "id": 1,
        "messages": all_messages
    }]

def get_user_by_id(user_id: int):
    # Logic to fetch a specific user
    # 
    return {"id": user_id, "name": "Unkown"}
