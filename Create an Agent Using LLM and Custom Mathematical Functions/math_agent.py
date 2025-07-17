
# from langgraph.graph import StateGraph, END
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from typing import TypedDict, List, Optional
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.tools import tool
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Initialize Gemini LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

# class AgentState(TypedDict):
#     input: str
#     chat_history: List[dict]
#     agent_outcome: Optional[dict]
#     intermediate_steps: List[dict]

# # Define math tools
# @tool
# def plus(a: float, b: float) -> float:
#     """Add two numbers together. Use for addition problems."""
#     return a + b

# @tool
# def sub(a: float, b: float) -> float:
#     """Subtract b from a. Use for subtraction problems."""
#     return a - b

# @tool
# def mul(a: float, b: float) -> float:
#     """Multiply two numbers. Use for multiplication problems."""
#     return a * b

# @tool
# def divide(a: float, b: float) -> float:
#     """Divide a by b. Use for division problems. Returns error if dividing by zero."""
#     if b == 0:
#         return "Error: Cannot divide by zero"
#     return a / b

# # List of all available tools
# tools = [plus, sub, mul, divide]

# # System prompt to guide the agent's behavior
# system_prompt = """You are a helpful assistant that can:
# - Answer general knowledge questions
# - Perform math calculations when requested

# For math operations, always use the appropriate tools.
# For all other questions, respond using your knowledge.

# Format all responses clearly and helpfully."""

# # Create the agent with tools
# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=tools,
#     prompt=ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#         MessagesPlaceholder("agent_scratchpad"),
#     ]),
# )

# # Create a simple agent executor instead of using LangGraph
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# def run_agent(query: str, chat_history: List = []):
#     """Execute the agent with a user query"""
#     try:
#         response = agent_executor.invoke({
#             "input": query,
#             "chat_history": chat_history
#         })
#         return response["output"]
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Interactive chat interface
# if __name__ == "__main__":
#     print("Math-Q&A Agent ready! Type 'exit' to quit.")
#     chat_history = []
    
#     while True:
#         try:
#             query = input("\nYou: ")
#             if query.lower() in ['exit', 'quit']:
#                 break
#             if not query.strip():
#                 continue
                
#             response = run_agent(query, chat_history)
#             chat_history.extend([
#                 {"role": "user", "content": query},
#                 {"role": "assistant", "content": str(response)}
#             ])
#             print(f"Agent: {response}")
            
#         except KeyboardInterrupt:
#             break
#         except Exception as e:
#             print(f"Error: {str(e)}")
from langgraph.graph import StateGraph, END
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, List, Optional, Annotated, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import operator

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

# Define the State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    agent_outcome: Optional[Union[BaseMessage, List[tuple]]]
    intermediate_steps: List

# Define math tools
@tool
def plus(a: float, b: float) -> float:
    """Add two numbers together. Use for addition problems."""
    return a + b

@tool
def sub(a: float, b: float) -> float:
    """Subtract b from a. Use for subtraction problems."""
    return a - b

@tool
def mul(a: float, b: float) -> float:
    """Multiply two numbers. Use for multiplication problems."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Use for division problems. Returns error if dividing by zero."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return a / b

# List of all available tools
tools = [plus, sub, mul, divide]

# System prompt to guide the agent's behavior
system_prompt = """You are a helpful assistant that can:
- Answer general knowledge questions
- Perform math calculations when requested

For math operations, always use the appropriate tools.
For all other questions, respond using your knowledge.

Format all responses clearly and helpfully."""

# Create the agent with tools
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]),
)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Define nodes for the graph
def agent_node(state: AgentState):
    result = agent_executor.invoke({
        "input": state["user_input"],
        "chat_history": state["messages"]
    })
    return {
        "messages": [AIMessage(content=result["output"])],
        "agent_outcome": result["output"]
    }

def tool_node(state: AgentState):
    # This would be used if we wanted to explicitly model tool execution as a separate step
    # For tool_calling_agent, tools are handled automatically by the agent
    pass

# Define the graph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
# workflow.add_node("tools", tool_node)  # Not needed for tool_calling_agent

# Set entry point
workflow.set_entry_point("agent")

# Define edges
# workflow.add_edge("tools", "agent")  # Not needed for tool_calling_agent
workflow.add_edge("agent", END)

# Compile the graph
app = workflow.compile()

def run_agent(query: str, chat_history: List[BaseMessage] = []):
    """Execute the agent with a user query using LangGraph"""
    try:
        inputs = {
            "messages": chat_history,
            "user_input": query
        }
        response = app.invoke(inputs)
        return response["agent_outcome"]
    except Exception as e:
        return f"Error: {str(e)}"

# Interactive chat interface
if __name__ == "__main__":
    print("Math-Q&A Agent ready! Type 'exit' to quit.")
    chat_history = []
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            # Convert chat history to BaseMessage format
            messages = []
            for item in chat_history:
                if item["role"] == "user":
                    messages.append(HumanMessage(content=item["content"]))
                else:
                    messages.append(AIMessage(content=item["content"]))
            
            response = run_agent(query, messages)
            
            # Update chat history
            chat_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": str(response)}
            ])
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")