from dotenv import load_dotenv
import logging
import uuid
from interface import create_interface
from agent.planning_agent import setup_agent_graph
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.graph, self.memory = setup_agent_graph()

    def process_query(self, query: str, history: list, session_id: str) -> str:
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

def main():
    try:
        load_dotenv()
        session_id = str(uuid.uuid4())
        
        agent_manager = AgentManager()
        
        logger.info(f"Starting Gradio app with session_id: {session_id}")
        app = create_interface(
            agent_manager.process_query, 
            session_id
        )
        app.queue()
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()