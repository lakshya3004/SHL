from typing import List, Dict, Any
from loguru import logger
from app.models.orchestration_models import ConversationState, UserRequirements
from app.services.llm.llm_service import LLMService
from app.services.llm import prompts


class ConversationAnalyzer:
    """
    Analyzes conversation history to detect intent and extract requirements.
    Uses LLM-assisted analysis with deterministic heuristic fallbacks for robustness.
    """

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def analyze(
        self, history: List[Dict[str, str]], latest_message: str, turn_count: int = 1
    ) -> ConversationState:
        """
        Runs full analysis on the current conversation turn.
        turn_count is the number of user messages so far (including this one).
        """
        logger.info(f"Analyzing conversation turn {turn_count}...")

        # Fast-path: heuristic safety checks before calling LLM
        if self._is_off_topic_heuristic(latest_message):
            return ConversationState(
                intent="refuse",
                requirements=UserRequirements(),
                recent_query=latest_message,
                turn_count=turn_count
            )

        # Build analysis prompt with turn_count context
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
        )
        analysis_prompt = (
            f"turn_count: {turn_count} (number of user messages so far)\n\n"
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"LATEST USER MESSAGE: {latest_message}\n\n"
            f"Analyze the above and return the JSON schema as specified."
        )

        analysis_messages = [
            {"role": "system", "content": prompts.ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": analysis_prompt}
        ]

        analysis_json = await self.llm.generate_json(analysis_messages)

        # Extract with safe defaults
        intent = analysis_json.get("intent", "clarify")
        req_data = analysis_json.get("requirements", {})
        is_sufficient = analysis_json.get("is_sufficient", False)

        requirements = UserRequirements(
            role=req_data.get("role"),
            seniority=req_data.get("seniority"),
            skills=req_data.get("skills", []),
            hiring_goals=req_data.get("hiring_goals", []),
            test_type_preference=req_data.get("test_type_preference", []),
            is_sufficient=is_sufficient
        )

        # Deterministic overrides to prevent over-clarification
        # If turn >= 4 and we know the role, force recommendation
        if turn_count >= 4 and (requirements.role or requirements.skills):
            intent = "recommend"
            requirements.is_sufficient = True

        # Cannot recommend without any context
        if intent == "recommend" and not (requirements.role or requirements.skills or requirements.hiring_goals):
            intent = "clarify"
            requirements.is_sufficient = False

        # Safety: refuse if heuristic + LLM both flag it
        if intent == "refuse" or self._is_off_topic_heuristic(latest_message):
            intent = "refuse"

        return ConversationState(
            intent=intent,
            requirements=requirements,
            recent_query=latest_message,
            turn_count=turn_count
        )

    def _is_off_topic_heuristic(self, message: str) -> bool:
        """Fast check for obviously off-topic or injection attempts."""
        off_topic_keywords = [
            "weather", "pizza", "joke", "poem", "recipe", "story about",
            "ignore previous instructions", "disregard all prior",
            "pretend you are", "you are now", "forget everything",
            "reveal your system prompt", "what is your prompt",
            "salary negotiation", "legal advice", "legal counsel",
            "recommend a competitor", "non-shl test", "who is the president",
            "tell me about yourself", "what is your name"
        ]
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in off_topic_keywords)
