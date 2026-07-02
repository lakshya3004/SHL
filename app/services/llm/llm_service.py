import json
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger
from app.core.config import settings


class LLMService:
    """
    LLM provider wrapper supporting Google Gemini (google-genai SDK)
    and OpenAI-compatible endpoints (e.g. OpenAI, Groq, OpenRouter, local llama/Ollama).
    Handles async execution, retries, timeouts, and robust JSON extraction.
    """

    def __init__(self):
        self._client = None
        self.use_openai = False
        self._init_client()

    def _init_client(self):
        """Initialize the LLM client (Gemini or OpenAI)."""
        gemini_key = settings.GEMINI_API_KEY
        if gemini_key and ("your_gemini" in gemini_key.lower() or "placeholder" in gemini_key.lower() or gemini_key.strip() == ""):
            gemini_key = ""

        openai_key = settings.OPENAI_API_KEY
        if openai_key and ("placeholder" in openai_key.lower() or openai_key.strip() == ""):
            openai_key = ""

        # If GEMINI_API_KEY is not set but OPENAI_API_KEY is provided, use OpenAI
        if not gemini_key and openai_key:
            self.use_openai = True
            logger.info(
                f"Initializing OpenAI-compatible LLM client with URL: {settings.LLM_BASE_URL} "
                f"and model: {settings.OPENAI_MODEL_NAME}"
            )
            return

        if not gemini_key:
            logger.warning("Neither GEMINI_API_KEY nor valid OPENAI_API_KEY is set. LLM calls will use fallback mock mode.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=gemini_key)
            logger.info("Gemini client initialized successfully (google-genai SDK).")
        except ImportError:
            # Fallback to legacy SDK
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self._client = "legacy"
                self._legacy_genai = genai
                logger.info("Gemini client initialized (legacy google-generativeai SDK).")
            except Exception as e2:
                logger.error(f"Failed to initialize any Gemini SDK: {e2}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Sends a request to the LLM (Gemini or OpenAI) and returns a text response.
        Falls back to a smart deterministic conversation generator if no API key is set.
        """
        if self.use_openai:
            try:
                formatted_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    if role not in ("system", "user", "assistant"):
                        role = "user"
                    formatted_messages.append({"role": role, "content": msg.get("content", "")})

                result = await self._call_openai_completion(formatted_messages, temperature, max_tokens)
                return result or self._fallback_response()
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")
                return self._fallback_response()

        if self._client is not None:
            prompt = self._build_prompt(messages)
            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._call_text(prompt, temperature, max_tokens)),
                    timeout=settings.LLM_TIMEOUT_SECONDS
                )
                return result or self._fallback_response()
            except asyncio.TimeoutError:
                logger.error("LLM text request timed out.")
                return self._fallback_response()
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return self._fallback_response()

        # Deterministic Mock Fallback for local testing/offline validation
        return self._generate_mock_completion(messages)

    async def generate_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Requests structured JSON output from the LLM (Gemini or OpenAI).
        Falls back to a smart deterministic rule-based analyzer if no API key is set.
        """
        if self.use_openai:
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                if role not in ("system", "user", "assistant"):
                    role = "user"
                formatted_messages.append({"role": role, "content": msg.get("content", "")})

            if not any("json" in m["content"].lower() for m in formatted_messages):
                formatted_messages.append({"role": "user", "content": "Respond ONLY with valid JSON. No markdown, no explanation."})

            try:
                raw = await self._call_openai_completion(formatted_messages, 0.1, 1024, response_json=True)
                return self._parse_json_safe(raw)
            except Exception as e:
                logger.warning(f"OpenAI JSON format request failed, retrying without strict format mode: {e}")
                try:
                    raw = await self._call_openai_completion(formatted_messages, 0.1, 1024, response_json=False)
                    return self._parse_json_safe(raw)
                except Exception as e2:
                    logger.error(f"OpenAI JSON fallback call failed: {e2}")
                    return {}

        if self._client is not None:
            modified = list(messages)
            modified.append({"role": "user", "content": "Respond ONLY with valid JSON. No markdown, no explanation, no text outside the JSON object."})
            prompt = self._build_prompt(modified)
            try:
                loop = asyncio.get_event_loop()
                raw = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._call_json(prompt)),
                    timeout=settings.LLM_TIMEOUT_SECONDS
                )
                return self._parse_json_safe(raw)
            except asyncio.TimeoutError:
                logger.error("LLM JSON request timed out.")
                return {}
            except Exception as e:
                logger.error(f"LLM JSON call failed: {e}")
                return {}

        # Deterministic Mock Fallback for local testing/offline validation
        return self._generate_mock_json(messages)

    def _generate_mock_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyzes user message keywords to simulate requirements extraction."""
        # Find the actual latest user message from the analyzer prompt
        latest_user_msg = ""
        for m in reversed(messages):
            content = m.get("content", "")
            if "LATEST USER MESSAGE:" in content:
                parts = content.split("LATEST USER MESSAGE:")
                if len(parts) > 1:
                    latest_user_msg = parts[1].split("\n\n")[0].strip()
                    break
        
        # Fallback to last user message if not in analyzer format
        if not latest_user_msg:
            for m in reversed(messages):
                if m.get("role") == "user" and "### AVAILABLE" not in m.get("content", ""):
                    latest_user_msg = m.get("content", "")
                    break

        # Extract only USER lines from history to avoid assistant reply pollution
        history_user_text = ""
        for m in messages:
            content = m.get("content", "")
            if "CONVERSATION_HISTORY:" in content:
                lines = content.split("\n")
                user_lines = [line.replace("USER:", "").strip() for line in lines if line.strip().startswith("USER:")]
                history_user_text = " ".join(user_lines)
                break

        # Fallback to joining raw user messages if not in analyzer format
        if not history_user_text:
            history_user_text = " ".join([
                m.get("content", "") for m in messages 
                if m.get("role") == "user" and "### AVAILABLE" not in m.get("content", "")
            ])

        full_user_history = (history_user_text + " " + latest_user_msg).strip().lower()
        
        # Calculate turn count based on number of user messages
        user_messages_count = len([line for line in history_user_text.split("\n") if line.strip()]) + 1
        if not history_user_text.strip():
            user_messages_count = len([m for m in messages if m.get("role") == "user"])
        if user_messages_count < 1:
            user_messages_count = 1

        # Check for off-topic or injection attempts
        if any(w in latest_user_msg.lower() for w in ["pizza", "weather", "recipe", "ignore previous", "disregard"]):
            return {
                "intent": "refuse",
                "requirements": {},
                "is_sufficient": False
            }

        # Check for comparison requests
        if any(w in latest_user_msg.lower() for w in ["compare", "difference", "versus", " vs "]):
            return {
                "intent": "compare",
                "requirements": {},
                "is_sufficient": True
            }

        # Extract role with spelling tolerances
        role = None
        if "java" in full_user_history:
            role = "Java Developer"
        elif "python" in full_user_history:
            role = "Python Developer"
        elif "javascript" in full_user_history or "react" in full_user_history or "node" in full_user_history or " js " in full_user_history:
            role = "Frontend/Fullstack Developer"
        elif "data scientist" in full_user_history or "data science" in full_user_history:
            role = "Data Scientist"
        elif "sales" in full_user_history:
            role = "Sales Representative"
        elif any(w in full_user_history for w in ["customer", "call center", "call centre", "service", "support", "agent"]):
            role = "Customer Service Agent"
        elif any(w in full_user_history for w in ["graduate", "campus", "trainee", "fresher"]):
            role = "Graduate Trainee"
        elif any(w in full_user_history for w in ["finance", "banking", "analyst"]):
            role = "Financial Analyst"

        # Seniority
        seniority = None
        if "senior" in full_user_history or "lead" in full_user_history:
            seniority = "Senior"
        elif "mid" in full_user_history:
            seniority = "Mid-Level"
        elif any(w in full_user_history for w in ["junior", "entry", "graduate"]):
            seniority = "Junior"

        # Skills
        skills = []
        for kw in ["spring", "django", "sql", "excel", "c++", "c#", "scala", "kotlin", "cognitive", "verbal", "numerical", "simulation"]:
            if kw in full_user_history:
                skills.append(kw.capitalize())

        # If turn count >= 2, we force recommendation if any signal is present
        is_sufficient = bool(role or skills or user_messages_count >= 2)
        
        # Fallback if no specific role matched at turn >= 2
        if is_sufficient and not role:
            role = "General Candidate"

        return {
            "intent": "recommend" if is_sufficient else "clarify",
            "requirements": {
                "role": role,
                "seniority": seniority,
                "skills": skills,
                "hiring_goals": [],
                "test_type_preference": []
            },
            "is_sufficient": is_sufficient
        }

    def _generate_mock_completion(self, messages: List[Dict[str, str]]) -> str:
        """Simulates conversational replies grounded in retrieved assessments."""
        user_msg = ""
        context_text = ""
        for m in messages:
            content = m.get("content", "")
            if "### AVAILABLE SHL ASSESSMENTS:" in content:
                context_text = content
            elif m.get("role") == "user":
                user_msg = content.lower()

        logger.info(f"MOCK COMP: context_len={len(context_text)}, user_msg='{user_msg[:60]}'")
        
        # Refusal check
        if any(w in user_msg for w in ["pizza", "weather", "recipe"]):
            return "I am only authorized to assist with selecting and comparing SHL assessments from the official catalog. Let's get back to your hiring requirements!"

        # Comparison check
        if any(w in user_msg for w in ["compare", "difference", "versus", " vs "]):
            return (
                "Based on the catalog data: \n\n"
                "- **OPQ32r** is an Occupational Personality Questionnaire that measures 32 characteristics "
                "predicting workplace behavior and leadership style.\n"
                "- **Motivation Questionnaire (MQ)** assesses 18 dimensions of work motivation and what drives "
                "individual performance.\n\n"
                "Let me know if you would like to shortlist either of these!"
            )

        # Recommendation check
        import re
        names = re.findall(r"Name:\s*(.*)", context_text)
        if names:
            clean_names = [n.strip() for n in names if n.strip()]
            reply = f"Based on your requirements, I have selected the top {len(clean_names[:5])} relevant SHL assessments from the catalog:\n\n"
            for name in clean_names[:5]:
                reply += f"- **{name}**: Evaluates the core competencies needed for this role.\n"
            reply += "\nAll selected recommendations are displayed in your shortlist sidebar. You can compare them or ask to refine further!"
            return reply

        # Clarification prompt
        return (
            "I would be happy to help you find the right SHL assessments. "
            "Could you tell me the job title/role you are hiring for, and what specific "
            "seniority or technical skills you would like to evaluate?"
        )

    async def _call_openai_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_json: bool = False
    ) -> str:
        """Call the OpenAI-compatible chat completion endpoint asynchronously."""
        url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.OPENAI_MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _call_text(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Synchronous text call — runs in thread pool."""
        if self._client == "legacy":
            model = self._legacy_genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            resp = model.generate_content(prompt)
            return resp.text.strip()
        else:
            from google.genai import types
            resp = self._client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            return resp.text.strip() if resp.text else ""

    def _call_json(self, prompt: str) -> str:
        """Synchronous JSON-mode call — runs in thread pool."""
        if self._client == "legacy":
            model = self._legacy_genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                    "response_mime_type": "application/json",
                }
            )
            resp = model.generate_content(prompt)
            return resp.text.strip()
        else:
            from google.genai import types
            resp = self._client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                )
            )
            return resp.text.strip() if resp.text else "{}"

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build a single flat prompt from message list."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}")
            else:
                parts.append(f"[USER]\n{content}")
        return "\n\n".join(parts)

    def _parse_json_safe(self, content: str) -> Dict[str, Any]:
        """Robustly extract JSON from LLM output, handling markdown blocks."""
        if not content:
            return {}

        # Strip markdown code fences
        for marker in ["```json", "```"]:
            if marker in content:
                parts = content.split(marker)
                if len(parts) >= 3:
                    content = parts[1].strip()
                    break

        # Extract JSON object from surrounding text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed ({e}). Content: {content[:200]}")
            return {}

    def _fallback_response(self) -> str:
        return (
            "I can help you find the right SHL assessments for your hiring needs. "
            "Could you tell me more about the role you're hiring for and what skills "
            "or competencies you'd like to evaluate?"
        )

