import time
from typing import List, Dict, Any
from loguru import logger

from app.models.request_models import ChatRequest, ChatMessage
from app.models.response_models import ChatResponse, AssessmentRecommendation
from app.models.orchestration_models import ConversationState
from app.core.config import settings

from app.services.llm.llm_service import LLMService
from app.services.retrieval.hybrid_search import HybridRetriever
from app.services.retrieval.context_builder import RetrievalContextBuilder
from app.services.orchestration.conversation_analyzer import ConversationAnalyzer
from app.services.orchestration.recommendation_engine import RecommendationEngine
from app.services.orchestration.comparison_engine import ComparisonEngine
from app.services.orchestration.refusal_engine import RefusalEngine
from app.services.orchestration.response_generator import ResponseGenerator

from app.services.evaluation.evaluator_guardrails import EvaluatorGuardrails
from app.services.evaluation.retrieval_validator import RetrievalValidator
from app.services.evaluation.response_validator import ResponseValidator
from app.services.evaluation.metrics import PerformanceMonitor


class ChatOrchestrator:
    """
    Hardened controller for the recommendation flow.
    Handles the stateless messages[] API format, enforces the 8-turn cap,
    and sets end_of_conversation appropriately.
    """

    def __init__(
        self,
        llm: LLMService,
        retriever: HybridRetriever,
        context_builder: RetrievalContextBuilder
    ):
        self.analyzer = ConversationAnalyzer(llm)
        self.recommender = RecommendationEngine(retriever)
        self.comparer = ComparisonEngine(llm)
        self.refuser = RefusalEngine(llm)
        self.generator = ResponseGenerator(llm)
        self.context_builder = context_builder

        self.guardrails = EvaluatorGuardrails()
        self.retrieval_validator = RetrievalValidator()
        self.response_validator = ResponseValidator()

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Processes a stateless chat request from the evaluator.

        The request.messages array contains the FULL conversation including
        the latest user message as the last element. We extract:
          - latest_message: the last user-role message
          - history: all prior messages (for context)
          - turn_count: number of user messages so far

        Turn cap: evaluator allows max 8 turns (user+assistant combined).
        We enforce recommendation by turn 6 at the latest.
        """
        start_time = time.time()

        messages = request.messages

        # Extract the latest user message and history
        latest_message, history, turn_count = self._parse_messages(messages)

        if not latest_message:
            return ChatResponse(
                reply="I didn't receive a message. Could you describe the role you're hiring for?",
                recommendations=[],
                end_of_conversation=False
            )

        # Sanitize input
        query = self.guardrails.sanitize_input(latest_message)

        # --- 1. Safety Guardrails ---
        if self.guardrails.is_injection_attempt(query) or self.guardrails.is_malicious_intent(query):
            reply = await self.refuser.handle_refusal(query)
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

        # --- 2. Turn Cap Enforcement ---
        # If we're at turn 7+ (user messages), force a recommendation to stay within cap
        force_recommend = turn_count >= settings.MAX_TURNS // 2

        # --- 3. Analyze Intent ---
        state: ConversationState = await self.analyzer.analyze(
            history, query, turn_count=turn_count
        )

        # Apply force-recommend override if near turn limit
        if force_recommend and state.intent in ("clarify",):
            if state.requirements.role or state.requirements.skills or state.requirements.hiring_goals:
                state.intent = "recommend"
                state.requirements.is_sufficient = True
                logger.info(f"Turn cap override at turn {turn_count}: forcing recommend.")

        # --- 4. Route & Generate ---
        final_response = await self._route(state, query, history, turn_count)

        # --- 5. Final Validation ---
        final_response = self.response_validator.validate(final_response)

        # Record timing
        PerformanceMonitor.record_request(
            time.time() - start_time, 0,
            len(final_response.recommendations) == 0,
            len(final_response.recommendations)
        )

        return final_response

    async def _route(
        self,
        state: ConversationState,
        query: str,
        history: List[Dict[str, str]],
        turn_count: int
    ) -> ChatResponse:
        """Routes to the correct handler based on detected intent."""

        if state.intent == "refuse":
            reply = await self.refuser.handle_refusal(query)
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

        elif state.intent == "compare":
            candidates = await self.recommender.get_candidates(state, k=6)
            candidates = self.retrieval_validator.validate_results(candidates)
            reply = await self.comparer.compare(query, candidates)
            return self._build_response(reply, candidates, end_of_conversation=False)

        else:
            decision = self.recommender.evaluate_readiness(state)

            if not decision.should_recommend and state.intent not in ("recommend", "refine"):
                # Generate a clarification response
                context = self.context_builder.build_context(query, [])
                reply = await self.generator.generate_response(query, context, "clarify", history)
                return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

            else:
                # Retrieve and recommend
                candidates = await self.recommender.get_candidates(state, k=10)
                candidates = self.retrieval_validator.validate_results(candidates)
                # Cap at 10 per spec
                candidates = candidates[:10]

                context = self.context_builder.build_context(query, candidates)
                reply = await self.generator.generate_response(
                    query, context, state.intent, history
                )

                # Determine end_of_conversation: true when we've committed to a shortlist
                # and either we are near the turn cap (turn_count >= 4) or the user has confirmed/accepted.
                confirmation_keywords = ["perfect", "confirm", "thank", "done", "great", "accept", "correct", "yes", "sure", "ok"]
                has_confirmed = any(w in query.lower() for w in confirmation_keywords)
                eoc = len(candidates) >= 1 and (turn_count >= 4 or (turn_count >= 2 and has_confirmed))

                return self._build_response(reply, candidates, end_of_conversation=eoc)

    def _parse_messages(self, messages: List[ChatMessage]):
        """
        Extract latest_message, history list, and user turn_count from
        the full messages array. Returns (latest_message, history, turn_count).
        """
        if not messages:
            return "", [], 0

        # Count user turns
        user_messages = [m for m in messages if m.role == "user"]
        turn_count = len(user_messages)

        # Latest message must be from user (last user message in the array)
        latest_message = ""
        for m in reversed(messages):
            if m.role == "user":
                latest_message = m.content
                break

        # History = everything before the last user message
        # Find index of last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_idx = i
                break

        history = []
        if last_user_idx is not None and last_user_idx > 0:
            for m in messages[:last_user_idx]:
                history.append({"role": m.role, "content": m.content})

        return latest_message, history, turn_count

    def _build_response(
        self,
        reply: str,
        candidates: List[Any],
        end_of_conversation: bool
    ) -> ChatResponse:
        """Converts candidate list to the exact schema response."""
        type_to_code = {
            "ability": "A",
            "behavioral": "B",
            "competency": "C",
            "development": "D",
            "exercise": "E",
            "knowledge": "K",
            "motivation": "M",
            "personality": "P",
            "simulation": "S",
        }
        
        recommendations = []
        for c in candidates:
            if not c.url or not c.url.startswith("http"):
                continue
            
            # Map full name to single-letter code if possible to match example spec
            t_type = c.test_type
            code = None
            t_type_lower = t_type.lower()
            for key, val in type_to_code.items():
                if key in t_type_lower:
                    code = val
                    break
            if not code:
                code = t_type[0].upper() if t_type else "K"
                
            recommendations.append(AssessmentRecommendation(
                name=c.name,
                url=c.url,
                test_type=code
            ))

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation
        )
