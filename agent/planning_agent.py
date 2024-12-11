import os
from typing import Dict, Annotated, TypedDict
import logging
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import MemorySaver
from langgraph.graph.message import add_messages
from agent.router_agent import route_query
from agent.generic_agent import process_generic_query
from agent.product_review_agent import setup_product_review_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an efficient and helpful AI planning agent. Do not assume anything. Always use route_query to determine the router_response. The router_response is the variable which has the user query type. Always refer to this variable router_response.
If router_response is 'generic', use handle_generic_query to process the user query and then use compose_response to format the response. 
If router_response is 'product_review', use get_product_info to retrieve product-related data and then use compose_response to format the response.

Example:

    User: Okay , i want to buy a phone. What buying options do you have ?
    Thought: I will use route_query to understand if query is product_review or generic
    Action: route_query
    Observation: product_review
    Thought: This query is actually a product review request, not a generic query. I will use get_product_info to get appropriate response.
    Action: get_product_info
    Action Input: User query: Okay , i want to buy a phone . What options do you have ?
    Observation: ok
    Thought:I have got the final answer. I will use compose_responses to format the response.
    Action: compose_responses
    Final Answer: ok
    """

class State(TypedDict):
    messages: Annotated[list, add_messages]


def initialize_state() -> State:
    return {
        "messages": []
    }

def get_product_info(state: State, config: dict) -> Dict:
    """Handle product-related queries using ProductReviewAgent"""
    try:
        product_agent = setup_product_review_agent()
        response = product_agent.process_review_query(state, config)
        
        if "error" in response:
            return {"error": response["error"]}
            
        return {"product_info": response["review_response"]}
        
    except Exception as e:
        logger.error(f"Error in get_product_info: {e}")
        return {"error": str(e)}

def compose_response(state: State, config: dict) -> Dict:
    """Compose final response based on processed information"""
    thread_id = config["configurable"]["thread_id"]
    
    logger.info(f"Composing response for thread {thread_id}")
    
    try:
        if "product_info" in state:
            response = state["product_info"]
        elif "generic_response" in state:
            response = state["generic_response"]
        else:
            response = "I apologize, but I couldn't process your request properly. Please try again."
        
        return {"final_response": response}
        
    except Exception as e:
        logger.error(f"Error in compose_response: {e}")
        return {"final_response": "I apologize, but I encountered an error. Please try again."}

def should_process_product_review(state: Dict) -> bool:
    """Determine if query should be routed to product review handler"""
    return state.get("router_response") == "product_review"

def should_process_generic(state: Dict) -> bool:
    """Determine if query should be routed to generic handler"""
    return state.get("router_response") == "generic"

def setup_agent_graph() -> tuple[StateGraph, MemorySaver]:
    """Setup and return the agent workflow graph"""
    memory = MemorySaver()
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("get_product_info", get_product_info)
    workflow.add_node("handle_generic_query", process_generic_query)
    workflow.add_node("compose_response", compose_response)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "route_query",
        {
            should_process_product_review: "get_product_info",
            should_process_generic: "handle_generic_query"
        }
    )
    
    # Add regular edges
    workflow.add_edge("get_product_info", "compose_response")
    workflow.add_edge("handle_generic_query", "compose_response")
    workflow.add_edge("compose_response", "end")
    
    # Set entry point
    workflow.add_edge(START, "route_query")
    
    compiled_workflow = workflow.compile(checkpointer=memory)

    return compiled_workflow, memory