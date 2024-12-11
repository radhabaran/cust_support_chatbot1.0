# composer_agent.py

import logging
from typing import Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compose_response(response_text: str) -> str:
    """
    Process and enhance the final response
    
    Args:
        response_text: String containing the response to format
        
    Returns:
        str: Formatted response text
    """
    try:
        logger.debug("Starting response composition")

        if not response_text:
            return "I apologize, but I couldn't find any response data to process."

        # Process the response
        return process_response(response_text)        
        
    except Exception as e:
        logger.error(f"Error in composition: {str(e)}")
        return "I apologize, but I encountered an error processing the response. Please try again."


def process_response(text: str) -> str:
    """Process and format the response text"""
    response_text = preprocess_response(text)
    response_text = remove_system_artifacts(response_text)
    response_text = format_response(response_text)
    
    return response_text.strip()


def preprocess_response(text: str) -> str:
    """Initial preprocessing of the response text"""
    text = " ".join(text.split())
    
    if not text or text.lower() in ['none', 'null', '']:
        text = "I apologize, but I don't have enough information to provide a response."
    
    return text


def remove_system_artifacts(text: str) -> str:
    """Remove any system artifacts or unwanted patterns"""
    artifacts = [
        "Assistant:", "AI:", "Human:", "User:",
        "System:", "Response:", "Output:",
        "Final Answer:", "Answer:"
    ]
    
    cleaned = text
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    
    cleaned = cleaned.strip('"').strip("'")
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    
    return cleaned.strip()


def format_response(text: str) -> str:
    """Apply standard formatting to the response"""
    if not text:
        return text
        
    sentences = [s.strip() for s in text.replace('\n', '. ').split('. ')]
    
    formatted_sentences = []
    for sentence in sentences:
        if sentence:
            formatted = sentence[0].upper() + sentence[1:] if sentence else sentence
            if not formatted[-1] in ['.', '!', '?']:
                formatted += '.'
            formatted_sentences.append(formatted)
    
    formatted = ' '.join(formatted_sentences)
    formatted = formatted.replace('..', '.').replace('  ', ' ')
    
    return formatted