from app.utils.logger import get_logger

logger = get_logger(__name__)


async def query_brain(text: str) -> str:
    # LLM integration is intentionally deferred for this milestone.
    logger.info("Brain client bypassed (LLM not enabled yet)")
    return text
