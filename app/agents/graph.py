import logging
from typing import Optional
from uuid import uuid4



from app.agents.nodes.chat import ChatNode
from langgraph.graph import StateGraph, START, END

from app.agents.state.state import AgentState 
from app.agents.nodes.translator import TranslatorNode
from app.agents.nodes.router import RouterNode
from app.agents.nodes.synthesizer import SynthesizerNode
from app.agents.nodes.judge import JudgeNode
from app.agents.nodes.db_tools import DBToolsNode
from app.agents.nodes.nedrex_tools import NeDRexToolsNode
logger = logging.getLogger(__name__)


class MedicalRAGGraph:
    """Builds and manages the simplified medical RAG workflow."""
    
    def __init__(self):
        """Initialize nodes and build graph."""
        logger.info("Initializing MedicalRAGGraph")
        
        # Initialize nodes
        self.translator = TranslatorNode()
        self.router = RouterNode()
        self.synthesizer = SynthesizerNode()
        self.judge = JudgeNode()
        self.db_tools = DBToolsNode()
        self.nedrex_tools = NeDRexToolsNode()
        self.chat = ChatNode()
        # Build graph
        self.graph = self._build_graph()
        logger.info("MedicalRAGGraph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Simple sequential flow:
        START → Translate → Router → (DB/NeDRex/Both) → Synthesize → Judge → END
        
        Returns:
            Compiled LangGraph
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("translate", self._translate_step)
        workflow.add_node("router", self._router_step)
        workflow.add_node("db_tools", self._db_tools_step)
        workflow.add_node("nedrex_tools", self._nedrex_tools_step)
        workflow.add_node("synthesize", self._synthesize_step)
        workflow.add_node("judge", self._judge_step)
        workflow.add_node("chat", self._chat_step)
        
        # Add edges (simple linear flow)
        workflow.add_edge(START, "translate")
        workflow.add_edge("translate", "router")
        workflow.add_conditional_edges("router", self.choose_path, {
            "db": "db_tools",
            "nedrex": "nedrex_tools",
            "both": "db_tools",
            "chitchat": "chat" 
        })
        
        # After db_tools, decide whether to go to synthesize or nedrex_tools
        workflow.add_conditional_edges("db_tools", self.choose_next_tool, {
            "synthesize": "synthesize",
            "nedrex_tools": "nedrex_tools",
        })
        
        # nedrex_tools always goes to synthesize
        workflow.add_edge("nedrex_tools", "synthesize")
        
        workflow.add_edge("chat", END)  # After chitchat response, end
        workflow.add_edge("synthesize", "judge")
        workflow.add_edge("judge", END)  
        # Compile
        compiled = workflow.compile()
        logger.info("Graph compiled and ready")
        
        return compiled
    
    def _translate_step(self, state: AgentState) -> AgentState:
        """Run translation node."""
        logger.info("Running TRANSLATE step")
        state = self.translator.translate(state)
        state.setdefault("steps_completed", []).append("translate")
        return state
    
    def _router_step(self, state: AgentState) -> AgentState:
        """Run router node."""
        logger.info("Running ROUTER step")
        state = self.router.route(state)
        state.setdefault("steps_completed", []).append("router")
        return state
    
    def _synthesize_step(self, state: AgentState) -> AgentState:
        """Run synthesizer node."""
        logger.info("Running SYNTHESIZE step")
        state = self.synthesizer.synthesize(state)
        state.setdefault("steps_completed", []).append("synthesize")
        return state
    
    def _judge_step(self, state: AgentState) -> AgentState:
        """Run judge node."""
        logger.info("Running JUDGE step")
        state = self.judge.judge(state)
        state.setdefault("steps_completed", []).append("judge")
        return state
    def _db_tools_step(self, state: AgentState) -> AgentState:
        """Run DB tools node to retrieve information from French database."""
        logger.info("Running DB_TOOLS step")
        state = self.db_tools.query_db(state)
        state.setdefault("steps_completed", []).append("db_tools")
        return state
    
    def _nedrex_tools_step(self, state: AgentState) -> AgentState:
        """Run NeDRex tools node to retrieve information from NeDRex API."""
        logger.info("Running NEDREX_TOOLS step")
        state = self.nedrex_tools.query_nedrex(state)
        state.setdefault("steps_completed", []).append("nedrex_tools")
        return state
    def _chat_step(self, state: AgentState) -> AgentState:
        """Run chat node to generate a response."""
        logger.info("Running CHAT step")
        state = self.chat.chat(state)
        state.setdefault("steps_completed", []).append("chat")
        return state
    def choose_path(self, state: AgentState) -> str:
        """Determine next node based on router's decision."""
        tool_choice = state.get("tool_choice", "chitchat")
        logger.info(f"Router decision: {tool_choice}")
        return tool_choice
    
    def choose_next_tool(self, state: AgentState) -> str:
        """Determine whether to go to nedrex_tools or synthesize after db_tools."""
        tool_choice = state.get("tool_choice", "db")
        logger.info(f"After db_tools, tool_choice: {tool_choice}")
        
        # If tool_choice was "both", we need to query nedrex next
        if tool_choice == "both":
            return "nedrex_tools"
        else:
            return "synthesize"

    
    def visualize_ascii(self) -> str:
        """
        Generate ASCII representation of the graph.
        
        Returns:
            ASCII art representation of the workflow
        """
        try:
            return self.graph.get_graph().draw_ascii()
        except Exception as e:
            logger.error(f"Error generating ASCII visualization: {e}")
            return ""
    
    def visualize_mermaid(self) -> str:
        """
        Generate Mermaid diagram representation of the graph.
        
        Returns:
            Mermaid markdown representation of the workflow
        """
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            logger.error(f"Error generating Mermaid visualization: {e}")
            return ""
    
    def save_visualization(self, output_dir: str = "./visualizations") -> dict:
        """
        Save graph visualizations to files.
        
        Args:
            output_dir: Directory to save visualizations
        
        Returns:
            Dict with file paths of saved visualizations
        """
        import os
        from pathlib import Path
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Save ASCII visualization
            ascii_viz = self.visualize_ascii()
            if ascii_viz:
                ascii_path = os.path.join(output_dir, "graph_ascii.txt")
                with open(ascii_path, "w") as f:
                    f.write(ascii_viz)
                saved_files["ascii"] = ascii_path
                logger.info(f"Saved ASCII visualization to {ascii_path}")
            
            # Save Mermaid visualization
            mermaid_viz = self.visualize_mermaid()
            if mermaid_viz:
                mermaid_path = os.path.join(output_dir, "graph_mermaid.md")
                with open(mermaid_path, "w") as f:
                    f.write("```mermaid\n")
                    f.write(mermaid_viz)
                    f.write("\n```")
                saved_files["mermaid"] = mermaid_path
                logger.info(f"Saved Mermaid visualization to {mermaid_path}")
            
            # Try to save PNG if pyvis is available
            try:
                png_path = os.path.join(output_dir, "graph.png")
                self.graph.get_graph().draw_mermaid_png(output_file_path=png_path)
                saved_files["png"] = png_path
                logger.info(f"Saved PNG visualization to {png_path}")
            except Exception as e:
                logger.warning(f"Could not generate PNG: {e}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")
            return {}

    def invoke(self, query: str, conversation_id: Optional[str] = None) -> dict:
        """
        Run the workflow with a user query.
        
        Args:
            query: User question (any language)
            conversation_id: Optional conversation ID for tracking
        
        Returns:
            Final state with response and validation result
        """
        conversation_id = conversation_id or str(uuid4())
        
        logger.info(f"Starting workflow for query: {query[:80]}...")
        
        # Initial state
        initial_state: AgentState = {
            "question": query,
            "original_query": query,
            "conversation_id": conversation_id,
            "steps_completed": [],
        }
        
        # Run workflow
        final_state = self.graph.invoke(initial_state)
        
        logger.info(f"Workflow completed. Steps: {final_state.get('steps_completed', [])}")
        
        return final_state
    
    def get_response(self, query: str) -> dict:
        """
        Convenience method to get just the response.
        
        Args:
            query: User question
        
        Returns:
            Dict with 'response' and 'is_valid' fields
        """
        state = self.invoke(query)
        return {
            "response": state.get("response", ""),
            "is_valid": state.get("is_valid", False),
            "validation_notes": state.get("validation_notes", ""),
            "tool_choice": state.get("tool_choice", ""),
        }


# Global graph instance
_graph_instance = None


def get_graph() -> MedicalRAGGraph:
    """Get or create global graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = MedicalRAGGraph()
    return _graph_instance

if __name__ == "__main__":
    # Quick test
    graph = get_graph()
    
