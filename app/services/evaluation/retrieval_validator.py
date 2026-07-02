from typing import List
from loguru import logger
from app.models.retrieval_models import HybridSearchResult


class RetrievalValidator:
    """
    Ensures retrieved recommendations are valid, unique, and grounded in the SHL catalog.
    Enforces the spec requirement: 1 to 10 recommendations max.
    """

    MAX_RECOMMENDATIONS = 10  # Per evaluator spec: max 10
    MIN_DESCRIPTION_LENGTH = 10  # Relaxed to avoid over-filtering

    def validate_results(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """
        Filters and validates the retrieved list of assessments.
        Returns up to MAX_RECOMMENDATIONS valid, deduplicated results.
        """
        validated = []
        seen_ids = set()
        seen_names = set()

        for res in results:
            # 1. Deduplication by ID and name
            if res.assessment_id in seen_ids:
                continue
            if res.name.lower() in seen_names:
                continue

            # 2. Must have a valid SHL URL
            if not res.url or not res.url.startswith("http"):
                logger.warning(f"Skipping result with invalid URL: {res.name}")
                continue

            # 3. Soft grounding check — must reference SHL domain or be from catalog
            if "shl.com" not in res.url:
                logger.warning(f"Skipping non-SHL URL for: {res.name}")
                continue

            validated.append(res)
            seen_ids.add(res.assessment_id)
            seen_names.add(res.name.lower())

            if len(validated) >= self.MAX_RECOMMENDATIONS:
                break

        logger.info(
            f"Retrieval validation: {len(results)} raw → {len(validated)} valid"
        )
        return validated
