# router_agent.py
from typing import Dict
import logging
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"))
load_dotenv()

class RouterResponse:
    PRODUCT_REVIEW = "product_review"
    GENERIC = "generic"


def should_process_product_review(state: Dict) -> bool:
    """Determine if query should be routed to product review handler"""
    return state.get("router_response") == RouterResponse.PRODUCT_REVIEW


def should_process_generic(state: Dict) -> bool:
    """Determine if query should be routed to generic handler"""
    return state.get("router_response") == RouterResponse.GENERIC


def planning_route_query(message: str, thread_id: str) -> str:
    """Route the query based on content analysis"""
    try:
        logger.info(f"Planning route query for thread {thread_id}")
        prompt = f"""Analyze the following query and determine if it's related to product review or a generic query.
        
        Product Review queries include:
        - Questions about product features, specifications, or capabilities
        - Product prices and availability inquiries
        - Requests for product reviews or comparisons
        - Product warranty or guarantee questions
        - Product shipping or delivery inquiries
        - Product compatibility or dimension questions
        - Product recommendations
        
        Generic queries include:
        - Customer service inquiries
        - Account-related questions
        - Technical support issues
        - Website navigation help
        - Payment or billing queries
        - Return policy questions
        - Company information requests
        
        Query: {message}
        
        Return ONLY 'product_review' or 'generic' as response."""
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages).content.lower().strip()
        
        if RouterResponse.PRODUCT_REVIEW in response:
            return RouterResponse.PRODUCT_REVIEW
        return RouterResponse.GENERIC
    
    except Exception as e:
        logger.error(f"Error in route_query for thread {thread_id}: {e}")
        return RouterResponse.GENERIC


def get_routing_metadata(thread_id: str, category: str) -> Dict:
    """Generate metadata for the routing decision"""
    return {
        "thread_id": thread_id,
        "routing_category": category,
    }


def validate_routing_category(category: str) -> str:
    """Validate and normalize routing category"""
    valid_categories = {RouterResponse.PRODUCT_REVIEW, RouterResponse.GENERIC}
    normalized = category.lower().strip()
    return normalized if normalized in valid_categories else RouterResponse.GENERIC


def log_routing_decision(message: str, category: str, thread_id: str):
    """Log routing decision with relevant context"""
    logger.info(
        f"Thread {thread_id} - Routed message: '{message[:50]}...' to category: {category}"
    )


class RouterAgent:
    def __init__(self, llm_instance=None):
        self.llm = llm_instance or llm
        
    def route_query(self, message: str, thread_id: str) -> Dict:
        """Main routing method that orchestrates the routing process"""
        try:
            # Get initial routing category
            category = planning_route_query(message, thread_id)
            
            # Validate category
            category = validate_routing_category(category)
            
            # Generate metadata
            metadata = get_routing_metadata(thread_id, category)
            
            # Log decision
            log_routing_decision(message, category, thread_id)
            
            return {
                "category": category,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in router agent for thread {thread_id}: {e}")
            return {
                "category": RouterResponse.GENERIC,
                "metadata": get_routing_metadata(thread_id, RouterResponse.GENERIC)
            }