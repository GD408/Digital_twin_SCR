import os
from typing import Annotated, List, TypedDict, Literal
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# ============================================================================
# 1. INITIALIZE LOCAL MODEL
# ============================================================================
# Using Llama 3.2 3B for local speed and tool-handling precision
llm = ChatOllama(model="llama3.2", temperature=0)

# ============================================================================
# 2. IE TOOLS (The 'Act' in ReAct)
# ============================================================================
@tool
def check_global_risk(location: str):
    """Checks for weather or labor-related disruptions in a specific industrial hub."""
    if "Taiwan" in location or "Hsinchu" in location:
        return "CRITICAL: Typhoon approaching Hsinchu Hub. Port closure expected for 72 hours."
    return "Status: Normal operations. No major threats detected."

@tool
def get_inventory_status(sku_id: str):
    """Queries ERP for stock levels, lead times, and safety stock thresholds."""
    db = {
        "SKU-001": {"part": "Semiconductors", "stock": 120, "safety": 500, "daily_use": 50},
        "SKU-002": {"part": "Lithium Cells", "stock": 2000, "safety": 1500, "daily_use": 100}
    }
    # Ensure tool handles variations in SKU format
    data = db.get(sku_id.upper().replace(" ", ""), "SKU not found in database.")
    return f"ERP DATA: {data}. Note: SKU-001 is currently below safety stock thresholds."

@tool
def logistics_cost_analyzer(mode: str):
    """Calculates the cost impact of switching logistics modes (Air vs Sea)."""
    costs = {"air": "$18,500 (2 days)", "sea": "$4,200 (15 days)"}
    return f"Quote for {mode} freight: {costs.get(mode.lower(), 'Invalid mode')}"

# ============================================================================
# 3. INITIALIZING ReAct AGENT EXECUTORS
# ============================================================================
risk_scout_executor = create_react_agent(llm, tools=[check_global_risk])
impact_analyst_executor = create_react_agent(llm, tools=[get_inventory_status])
planner_executor = create_react_agent(llm, tools=[logistics_cost_analyzer])

# ============================================================================
# 4. STATE & NODES
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: str

class Router(BaseModel):
    """Decide the next steps in the supply chain workflow."""
    next_nodes: List[Literal["RiskScout", "ImpactAnalyst", "LogisticsPlanner", "FINISH"]] = Field(
        description="The specialists to call. Use 'FINISH' if you are answering the user directly or the task is done."
    )
    reasoning: str = Field(description="Brief explanation of why these nodes were chosen.")
    direct_response: str = Field(description="If no specialists are needed, write the response to the user here.")

def risk_scout_node(state: AgentState):
    print("🔍 [RISK SCOUT] Running...")
    response = risk_scout_executor.invoke(state)
    return {"messages": [AIMessage(content=f"RISK REPORT: {response['messages'][-1].content}")]}

def impact_analyst_node(state: AgentState):
    print("📊 [IMPACT ANALYST] Running...")
    response = impact_analyst_executor.invoke(state)
    return {"messages": [AIMessage(content=f"IMPACT ANALYSIS: {response['messages'][-1].content}")]}

def logistics_planner_node(state: AgentState):
    print("🚛 [LOGISTICS PLANNER] Running...")
    # Human-in-the-Loop Interrupt
    proposal = "IE PROPOSAL: Switch SKU-001 to Air Freight. Please authorize."
    decision = interrupt(f"\n{proposal}\nAuthorize? (yes/no): ")
    
    response = planner_executor.invoke({"messages": [HumanMessage(content=f"User approved: {decision}. Use tool for Air cost.")]})
    return {"messages": [AIMessage(content=f"PLAN FINALIZED: {response['messages'][-1].content}")]}
# ============================================================================
# 5. SUPERVISOR & GRAPH
# ============================================================================
def supervisor_node(state: AgentState):
    system_prompt = (
        "You are the Supply Chain Orchestrator. Your job is to manage specialists or answer the user.\n"
        "1. If the user greets you or asks a general question, set next_nodes to ['FINISH'] and write a 'direct_response'.\n"
        "2. If specialized work is needed (Risk, Inventory, Logistics), list the 'next_nodes'.\n"
        "3. Once specialists have provided reports in the history, summarize their findings and set next_nodes to ['FINISH'].\n"
        "Specialists: RiskScout (Weather/Ports), ImpactAnalyst (ERP/Inventory), LogisticsPlanner (Shipping costs)."
    )
    
    structured_llm = llm.with_structured_output(Router)
    # Provide the full conversation history so it knows what has been done
    decision = structured_llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    
    print(f"\n🧠 [SUPERVISOR] Reasoning: {decision.reasoning}")
    
    # If the supervisor is finishing, we add its direct response to the message history
    if "FINISH" in decision.next_nodes or not decision.next_nodes:
        return {
            "next_node": ["FINISH"], 
            "messages": [AIMessage(content=decision.direct_response)]
        }
    
    return {"next_node": decision.next_nodes}

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("RiskScout", risk_scout_node)
builder.add_node("ImpactAnalyst", impact_analyst_node)
builder.add_node("LogisticsPlanner", logistics_planner_node)

builder.add_edge(START, "supervisor")

# This helper handles the list of nodes returned by the supervisor
def route_decision(state: AgentState):
    return state["next_node"]

builder.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "RiskScout": "RiskScout",
        "ImpactAnalyst": "ImpactAnalyst",
        "LogisticsPlanner": "LogisticsPlanner",
        "FINISH": END
    }
)

builder.add_edge("RiskScout", "supervisor")
builder.add_edge("ImpactAnalyst", "supervisor")
builder.add_edge("LogisticsPlanner", "supervisor")

app = builder.compile(checkpointer=InMemorySaver())

# ============================================================================
# 7. EXECUTION LOOP
# ============================================================================
config = {"configurable": {"thread_id": "IE_DYNAMIC_ROUTING"}, "recursion_limit": 30}

while True:
    state = app.get_state(config)
    
    # CASE A: The graph is waiting for Human-in-the-loop (interrupt)
    if state.next:
        decision = input("\n[MANAGER APPROVAL / Type 'exit' to stop]: ")
        
        # FIX: Check for exit here!
        if decision.lower() in ["quit", "exit"]:
            print("Shutting down Command Center...")
            break
            
        for event in app.stream(Command(resume=decision), config):
            for node, val in event.items():
                if "messages" in val: 
                    print(f"[{node.upper()}]: {val['messages'][-1].content}")
    
    # CASE B: The graph is idle, waiting for a new user query
    else:
        query = input("\nYou: ")
        if query.lower() in ["quit", "exit"]: 
            break
        for event in app.stream({"messages": [HumanMessage(content=query)]}, config):
            for node, val in event.items():
                if "messages" in val: 
                    print(f"[{node.upper()}]: {val['messages'][-1].content}")
