# composer_agent.py

import logging
from typing import Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compose_response(response_data: Union[Dict, str]) -> str:
    """
    Process and enhance the final response
    """
    try:
        logger.info("*********** in composer agent *************")
        logger.info(f"response input received : {response_data}")

        # Handle dict response format
        if isinstance(response_data, dict):
            if response_data.get("status") == "error":
                return response_data.get("response", "An error occurred")
            response_text = response_data.get("response", "")
        else:
            response_text = str(response_data)

        # Remove any system artifacts or unwanted patterns
        response_text = remove_system_artifacts(response_text)
        
        # Apply standard formatting
        response_text = format_response(response_text)
        
        # Ensure that the response does not have surrounding quotes
        return response_text.strip('"').strip("'")
        
    except Exception as e:
        logger.error(f"Error in composition: {str(e)}")
        return str(response_data)  # Fallback to original


def remove_system_artifacts(text: str) -> str:
    """Remove any system artifacts or unwanted patterns"""
    artifacts = ["Assistant:", "AI:", "Human:", "User:"]
    cleaned = text
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    # Remove double quotes
    cleaned = cleaned.replace('"', '').replace("'", "")  # Removes both double and single quotes
    
    return cleaned.strip()


def format_response(text: str) -> str:
    """Apply standard formatting"""
    # Add proper spacing
    formatted = text.replace("\n\n\n", "\n\n")
    
    # Ensure proper capitalization
    formatted = ". ".join(s.strip().capitalize() for s in formatted.split(". "))
    
    # Ensure proper ending punctuation
    if formatted and not formatted[-1] in ['.', '!', '?']:
        formatted += '.'
        
    return formatted