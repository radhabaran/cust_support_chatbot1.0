from dotenv import load_dotenv
import logging
import uuid
from typing import Dict, Annotated, TypedDict, List, Tuple, NotRequired
from interface import create_interface
from agent.planning_agent import setup_agent_graph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    generic_response: NotRequired[str]

class AgentManager:
    def __init__(self):
        self.graph, self.memory = setup_agent_graph(State)

    def process_query(self, query: str, history: List[Tuple[str, str]], session_id: str) -> str:
        try:
            new_message = HumanMessage(content=query)
            
            state = {
                "messages": [new_message]
            }
            
            config = {"configurable": {"thread_id": session_id}}
            result = self.graph.invoke(state, config=config)
            
            return result["final_response"]
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error: {str(e)}"


    def clear_context(self, session_id: str) -> tuple[List, str]:
        """Clear the conversation context for a session"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Cleared context for session {session_id}")
            return [], ""
        except Exception as e:
            logger.error(f"Error clearing context: {e}")
            return [], ""
              

def main():
    try:
        load_dotenv()
        session_id = str(uuid.uuid4())
        
        agent_manager = AgentManager()
        
        logger.info(f"Starting Gradio app with session_id: {session_id}")
        app = create_interface(
            process_query=agent_manager.process_query,
            clear_context=agent_manager.clear_context,
            session_id=session_id
        )
        app.queue()
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()