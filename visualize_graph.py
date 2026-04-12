"""
Graph Visualization Script

Generates visualizations of the Medical RAG workflow graph.
Saves in multiple formats: ASCII, Mermaid, and PNG.
"""

import logging
from pathlib import Path
from app.agents.graph import get_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate and save graph visualizations."""
    
    logger.info("Initializing Medical RAG Graph...")
    graph = get_graph()
    
    # Create visualizations directory
    vis_dir = Path("./visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    logger.info("Generating graph visualizations...")
    
    # Save all visualizations
    saved_files = graph.save_visualization(str(vis_dir))
    
    if saved_files:
        logger.info(f"✓ Visualizations saved successfully!")
        for format_type, file_path in saved_files.items():
            logger.info(f"  - {format_type.upper()}: {file_path}")
    else:
        logger.warning("No visualizations were generated")
        return
    
    # Print ASCII visualization to console
    logger.info("\n" + "="*80)
    logger.info("ASCII Visualization of the Graph:")
    logger.info("="*80 + "\n")
    
    ascii_viz = graph.visualize_ascii()
    if ascii_viz:
        print(ascii_viz)
    else:
        logger.warning("Could not generate ASCII visualization")
    
    logger.info("\n" + "="*80)
    logger.info("Mermaid Visualization:")
    logger.info("="*80 + "\n")
    
    mermaid_viz = graph.visualize_mermaid()
    if mermaid_viz:
        print("```mermaid")
        print(mermaid_viz)
        print("```")
    else:
        logger.warning("Could not generate Mermaid visualization")
    
    logger.info("\nVisualization generation complete!")


if __name__ == "__main__":
    main()
