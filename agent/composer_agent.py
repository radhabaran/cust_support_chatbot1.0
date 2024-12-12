# composer_agent.py

import logging
from typing import Dict, Union
from langchain_core.messages import AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compose_response(state: Dict, config: dict) -> Dict:
    """
    Process and enhance the final response
    
    Args:
        state: Current state dictionary containing messages and response_text
        config: Configuration dictionary
        
    Returns:
        Dict: Updated state with formatted response
    """
    try:
        logger.debug("Starting response composition")
        response_text = state.get("response_text", "")

        if not response_text:
            formatted_response = "I apologize, but I couldn't find any response data to process."
        else:
            # Process the response
            formatted_response = process_response(response_text)
            
        # Update state with formatted response
        state["final_response"] = formatted_response
        state["messages"].append(AIMessage(content=formatted_response))
        
        return state
        
    except Exception as e:
        logger.error(f"Error in composition: {str(e)}")
        error_msg = "I apologize, but I encountered an error processing the response. Please try again."
        state["final_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        return state


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
    
    # Handle different types of sentence separators
    text = text.replace('\n', ' ')
    
    # Split sentences more carefully
    sentences = []
    current_sentence = []
    
    # Split by spaces first to handle word by word
    words = text.split()
    
    for word in words:
        current_sentence.append(word)
        # Check for sentence endings
        if word and word[-1] in ['.', '!', '?'] and len(word) > 1:
            sentences.append(' '.join(current_sentence))
            current_sentence = []
    
    # Handle any remaining text
    if current_sentence:
        sentences.append(' '.join(current_sentence))
    
    # Format each sentence
    formatted_sentences = []
    for sentence in sentences:
        if sentence.strip():
            # Capitalize first letter
            formatted = sentence.strip()
            formatted = formatted[0].upper() + formatted[1:] if formatted else formatted
            
            # Ensure proper ending punctuation
            if not formatted[-1] in ['.', '!', '?']:
                formatted += '.'
                
            formatted_sentences.append(formatted)
    
    # Join sentences with proper spacing
    formatted = ' '.join(formatted_sentences)
    
    # Clean up any double spaces or double periods
    formatted = ' '.join(formatted.split())
    formatted = formatted.replace('..', '.')
    
    return formatted