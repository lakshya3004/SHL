from typing import List
from loguru import logger
from app.models.orchestration_models import ConversationState, RecommendationDecision
from app.services.retrieval.hybrid_search import HybridRetriever
from app.models.retrieval_models import HybridSearchResult


class RecommendationEngine:
    """
    Decides if we should recommend yet and executes retrieval.
    Enforces minimum context requirements, builds rich queries for better Recall@10.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def evaluate_readiness(self, state: ConversationState) -> RecommendationDecision:
        """
        Determines if there is enough information to make a valid recommendation.
        We are intentionally generous — any of (role, skills, hiring_goals) is enough.
        """
        reqs = state.requirements

        if not (reqs.role or reqs.skills or reqs.hiring_goals):
            return RecommendationDecision(
                should_recommend=False,
                confidence_score=0.2,
                clarification_questions=[
                    "What role or job title are you hiring for?",
                    "What are the key skills or competencies you need to evaluate?"
                ],
                reasoning="No role, skills, or hiring goals provided."
            )

        return RecommendationDecision(
            should_recommend=True,
            confidence_score=0.85,
            reasoning="Sufficient context: role/skills/goals available."
        )

    async def get_candidates(self, state: ConversationState, k: int = 10) -> List[HybridSearchResult]:
        """
        Builds a rich search query from conversation state and retrieves candidates.
        Uses k=10 for maximum Recall@10 coverage.
        """
        query_parts = []

        # Primary signals
        if state.requirements.role:
            query_parts.append(state.requirements.role)

        if state.requirements.seniority:
            query_parts.append(state.requirements.seniority)

        # Skill signals — most important for retrieval quality
        query_parts.extend(state.requirements.skills)

        # Hiring goal signals
        query_parts.extend(state.requirements.hiring_goals)

        # Test type preference signals
        for pref in state.requirements.test_type_preference:
            query_parts.append(pref)

        # For refine intent, include the recent query verbatim for refinement signal
        if state.intent in ("refine", "compare"):
            query_parts.append(state.recent_query)

        # Fallback: use the raw query if we have nothing else
        if not query_parts:
            query_parts.append(state.recent_query)

        # Deduplicate and clean
        seen = set()
        clean_parts = []
        for p in query_parts:
            if p and p.lower() not in seen:
                seen.add(p.lower())
                clean_parts.append(p)

        search_query = " ".join(clean_parts)
        logger.info(f"Retrieval query: '{search_query}' | intent={state.intent}")

        results = self.retriever.search(search_query, k=k)
        logger.info(f"Retrieved {len(results)} candidates.")
        return results
