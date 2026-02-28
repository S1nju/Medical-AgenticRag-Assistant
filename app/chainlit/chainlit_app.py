"""
Chainlit UI for Medical RAG Assistant.

Features:
- Multi-language input (Algerian Darija, French, Arabic, English)
- Real-time streaming responses
- Memory management (conversation history)
- Token tracking
- UI customization
- LangSmith tracing for observability
"""

import logging
import chainlit as cl
from app.agents.graph import get_graph
from app.core.tracing import setup_langsmith_tracing, get_langsmith_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith tracing
setup_langsmith_tracing()
langsmith_config = get_langsmith_config()
if langsmith_config["enabled"]:
    logger.info(f"🔍 LangSmith tracing active: {langsmith_config['project']}")

# Get graph instance
graph = get_graph()


# ============================================================================
# CHAINLIT SETUP
# ============================================================================

@cl.set_starters
async def set_starters():
    """Set suggested starter prompts for users."""
    return [
        cl.Starter(
            label="Dosage de l'aspirine",
            message="Quel est le dosage recommandé de l'aspirine pour les adultes?",
            icon="pill",
        ),
        cl.Starter(
            label="Symptômes de la grippe",
            message="Quels sont les symptômes de la grippe?",
            icon="fever",
        ),
        cl.Starter(
            label="Interactions médicamenteuses",
            message="Y a-t-il des interactions entre l'ibuprofène et l'aspirine?",
            icon="warning",
        ),
        cl.Starter(
            label="Effets secondaires",
            message="Quels sont les effets secondaires de l'ibuprofène?",
            icon="exclamation",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session."""
    # Create user session
    cl.user_session.set("conversation_id", cl.user_session.get("id"))
    cl.user_session.set("message_count", 0)
    cl.user_session.set("token_count", 0)
    
    # Welcome message
    await cl.Message(
        content="""
🏥 **Medical RAG Assistant**

Bienvenue! Je peux répondre à vos questions sur:
- 💊 Médicaments et dosages
- 🤒 Maladies et symptômes
- ⚠️ Effets secondaires et interactions

**Languages supported:** 
- Français (French)
- Darija (Algerian Arabic)
- English
- العربية (Arabic)

Posez votre question! 👇
        """
    ).send()
    
    logger.info(f"Chat session started: {cl.user_session.get('id')}")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle user messages with streaming response.
    """
    user_query = message.content
    conversation_id = cl.user_session.get("conversation_id")
    message_count = cl.user_session.get("message_count", 0) + 1
    
    cl.user_session.set("message_count", message_count)
    
    logger.info(f"Processing message #{message_count}: {user_query[:80]}...")
    
    # Create a message to stream the response
    response_message = cl.Message(content="")
    await response_message.send()
    
    try:
        # Show progress indicator
        async with cl.Step(
            name="Processing query...",
            type="run",
        ) as step:
            step.input = user_query
            
            # Run the workflow
            result = graph.invoke(user_query, conversation_id)
            
            # Extract info
            response_text = result.get("response", "")
            is_valid = result.get("is_valid", True)
            validation_notes = result.get("validation_notes", "")
            tool_choice = result.get("tool_choice", "")
            
            step.output = response_text[:200] + "..." if len(response_text) > 200 else response_text
        
        # Add validation indicator if needed
        if not is_valid:
            response_text += f"\n\n⚠️ **Note:** {validation_notes}"
        
        # Add source info
        response_text += f"\n\n📚 **Sources:** {tool_choice.upper()}"
        
        # Update message with full response
        response_message.content = response_text
        await response_message.update()
        
        # Add to session memory
        cl.user_session.set(
            "conversation_history",
            cl.user_session.get("conversation_history", [])
            + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": response_text},
            ],
        )
        
        logger.info(f"Message processed successfully. Valid: {is_valid}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        error_message = f"❌ **Error:** {str(e)}\n\nPlease try again."
        response_message.content = error_message
        await response_message.update()


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends."""
    message_count = cl.user_session.get("message_count", 0)
    logger.info(f"Chat session ended. Total messages: {message_count}")


# ============================================================================
# OPTIONAL: File Upload Handler (for documents)
# ============================================================================

# ============================================================================
# CUSTOM ACTIONS
# ============================================================================

@cl.action_callback("feedback")
async def handle_feedback(action):
    """Log user feedback."""
    await cl.Message(
        content=f"✅ Merci pour votre retour!",
        disable_feedback=True,
    ).send()


if __name__ == "__main__":
    # This runs the Chainlit server
    # Use: chainlit run app/api/chainlit_app.py
    logger.info("Starting Chainlit app")
