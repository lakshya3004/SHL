from loguru import logger
from app.models.response_models import ChatResponse


class ResponseValidator:
    """
    Ensures the final API response is strictly schema-compliant.
    Runs post-generation safety checks without being too aggressive.
    """

    def validate(self, response: ChatResponse) -> ChatResponse:
        """
        Runs post-generation checks on the final response object.
        """
        # 1. URL validation — must start with https://
        valid_recs = []
        for rec in response.recommendations:
            if not rec.url or not rec.url.startswith("http"):
                logger.warning(f"Dropping recommendation with bad URL: {rec.name}")
                continue
            if "shl.com" not in rec.url:
                logger.warning(f"Dropping non-SHL URL recommendation: {rec.name}")
                continue
            valid_recs.append(rec)
        response.recommendations = valid_recs

        # 2. Recommendation count enforcement: 1 to 10 per spec
        if len(response.recommendations) > 10:
            logger.warning("Capping recommendations at 10 per spec.")
            response.recommendations = response.recommendations[:10]

        # 3. Reply must never be empty
        if not response.reply or not response.reply.strip():
            response.reply = (
                "I can help you find the right SHL assessments. "
                "Could you tell me more about the role you're hiring for?"
            )

        # 4. Truncate excessively long replies
        if len(response.reply) > 2500:
            response.reply = response.reply[:2500] + "..."

        # 5. end_of_conversation consistency: only true if recommendations provided
        if response.end_of_conversation and not response.recommendations:
            response.end_of_conversation = False

        return response
