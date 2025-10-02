from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

from models.schemas import LLMRequest, LLMResponse, DebugContext
from llm.provider import LocalLLMProvider, ModelRegistry
from llm.prompt_builder import PromptBuilder
from context.context_manager import ContextManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
llm_provider = None
prompt_builder = None
context_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global llm_provider, prompt_builder, context_manager

    logger.info("🚀 Starting Smart Debug Companion Backend...")

    # Initialize components
    try:
        # Use free local LLM (deepseek-coder recommended for debugging)
        model = ModelRegistry.get_recommended()
        logger.info(f"Loading model: {model}")

        llm_provider = LocalLLMProvider(model=model)
        prompt_builder = PromptBuilder(max_tokens=4000)
        context_manager = ContextManager()

        logger.info("✓ All components initialized successfully")
        logger.info("✓ Backend ready at http://localhost:8000")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        logger.error("Please ensure Ollama is installed and running:")
        logger.error("  1. Install: https://ollama.ai")
        logger.error("  2. Run: ollama serve")
        logger.error(f"  3. Pull model: ollama pull {ModelRegistry.get_recommended()}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(
    title="Smart Debug Companion",
    description="Free, local LLM-powered debugging assistant for VS Code",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # VS Code webviews
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Smart Debug Companion",
        "model": llm_provider.model if llm_provider else "not loaded",
        "version": "1.0.0"
    }


@app.get("/models")
async def list_models():
    """List available free models."""
    return {
        "current": llm_provider.model if llm_provider else None,
        "available": ModelRegistry.list_models(),
        "recommended": ModelRegistry.get_recommended()
    }


@app.post("/debug/ask")
async def ask_debug_question(request: LLMRequest):
    """
    Main endpoint: Ask debugging question with context.

    Request body:
    {
        "user_message": "Why is this failing?",
        "context": { ... full debug context ... },
        "stream": true
    }

    Returns:
        Streaming response (SSE) if stream=True
        JSON response if stream=False
    """
    try:
        # Build prompt from context
        messages = prompt_builder.build_chat_messages(
            user_message=request.user_message,
            context=request.context
        )

        # Update context manager
        context_manager.add_user_message(request.user_message, request.context)

        if request.stream:
            # Stream response
            async def generate():
                try:
                    for chunk in llm_provider.chat(messages, stream=True):
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: Error: {str(e)}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        else:
            # Non-streaming response
            response_text = ""
            for chunk in llm_provider.chat(messages, stream=False):
                response_text += chunk

            return LLMResponse(
                response=response_text,
                context_used=["current_error", "recent_changes"],
                confidence=0.85
            )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/error")
async def report_error(context: DebugContext):
    """
    Report a new error to the system.
    Automatically updates context and suggests solutions.
    """
    try:
        # Update context manager
        context_manager.update_error(context.current_error)

        # Check for similar patterns
        suggestions = context_manager.find_similar_patterns(context.current_error)

        return {
            "error_received": True,
            "suggestions": suggestions,
            "session_id": context.session_id
        }

    except Exception as e:
        logger.error(f"Error reporting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/attempt")
async def record_attempt(session_id: str, attempt: dict):
    """
    Record a debugging attempt (what the user tried).
    """
    try:
        context_manager.add_attempt(session_id, attempt)

        return {
            "recorded": True,
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Error recording attempt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/session/{session_id}")
async def get_session(session_id: str):
    """
    Retrieve full debugging session history.
    """
    try:
        session = context_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/debug/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a debugging session (start fresh).
    """
    try:
        context_manager.clear_session(session_id)

        return {
            "cleared": True,
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
