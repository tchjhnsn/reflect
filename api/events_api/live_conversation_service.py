import json
import logging
import uuid

from django.conf import settings

from .context_assembly import ContextAssembler, assemble_context
from .graph_sync import get_journey_state_from_graph
from .journey_context import format_journey_context
from .journey_scoring import compute_soul_profile, compute_value_profile
from .llm_client import generate_conversation_reply
from .llm_prompts import build_system_prompt
from .persona_config import DEFAULT_PERSONA, get_persona
from .signal_engine import SignalGenerator

logger = logging.getLogger(__name__)

GRAPH_CONTEXT_PREAMBLE = (
    "## How to read the Context from Graph section\n"
    "You may receive a section called 'Context from Graph' containing structured "
    "data from the user's personal emotional knowledge graph.\n\n"
    "Signal history lines use the format "
    "SA(context, person, action, temporal): emotions - preview.\n"
    "Translate those into natural observations rather than reading the addresses aloud.\n\n"
    "When a dimension is '*', treat it as missing information to explore with one focused "
    "follow-up question.\n\n"
    "Observation bias patterns, clusters, and pending insights should be treated as "
    "hypotheses to explore rather than facts.\n"
)

DEFAULT_BASE_PROMPT = (
    "You are ThriveSight, a warm and perceptive AI companion that helps users "
    "understand themselves better through conversation. You have access to the user's "
    "personal emotional knowledge graph built from their conversation history.\n\n"
    "## Core guidelines\n"
    "- Be conversational, insightful, and concise (2-4 sentences).\n"
    "- End each response with one thoughtful follow-up question.\n"
    "- Never diagnose, prescribe, or label. Guide and illuminate.\n"
    "- Mirror the user's language and validate before probing.\n"
    "- When the user mentions specific people, places, or events, ask questions "
    "grounded in those details rather than generic prompts.\n\n"
    + GRAPH_CONTEXT_PREAMBLE
)

FALLBACK_SYSTEM_PROMPT = (
    "You are ThriveSight, a warm and perceptive AI companion that helps users "
    "understand themselves better through conversation. Be conversational, insightful, "
    "and concise (2-4 sentences). Ask thoughtful follow-up questions. Help users see "
    "patterns, connections, and blind spots in their thinking. Never diagnose or "
    "prescribe — guide and illuminate."
)

GENERATION_FAILURE_REPLY = (
    "I'm having trouble connecting to my thinking engine right now. Could you try again "
    "in a moment? In the meantime, I'd love to hear more about what's on your mind."
)

THERAPEUTIC_TEMPLATES = {
    "reflection": (
        "You are a warm, perceptive counseling-aware AI guide. Help the user reflect on "
        "their emotional state and recent experiences. Ask open-ended questions, validate "
        "feelings, and guide them toward deeper self-awareness. Keep responses concise "
        "(2-4 sentences). Never diagnose.\n\n"
        "When graph context includes emotion history, use it to deepen the reflection — "
        "connect current feelings to past patterns without lecturing.\n\n"
        + GRAPH_CONTEXT_PREAMBLE
    ),
    "decision": (
        "You are a structured thinking partner helping the user work through a decision. "
        "Use the framework of options, priorities, trade-offs, and gut sense. Keep "
        "responses concise and ask one question at a time.\n\n"
        "When graph context includes past signal data, ground choices in what the user "
        "has expressed before — 'Last time you faced a similar decision, you felt...' \n\n"
        + GRAPH_CONTEXT_PREAMBLE
    ),
    "patterns": (
        "You are a perceptive thinking partner who helps users notice patterns in their "
        "thoughts and behavior. Present observations gently as questions, not verdicts, "
        "and invite the user to confirm or correct them.\n\n"
        "Explicitly leverage cluster data and bias flags when present. If a cluster "
        "shows recurring conflict with the same person, name it. If bias flags appear, "
        "probe the underlying assumption gently.\n\n"
        + GRAPH_CONTEXT_PREAMBLE
    ),
}


class LiveConversationService:
    def __init__(self, *, workspace, user):
        self.workspace = workspace
        self.user = user
        self.workspace_id = str(workspace.id)

    def run(
        self,
        *,
        message: str,
        conversation_id: str | None,
        template_id: str | None,
        history: list[dict] | None,
        persona_id: str | None,
    ) -> dict:
        normalized_history = history or []
        active_conversation_id = conversation_id or str(uuid.uuid4())[:8]

        try:
            prompt_context = self.build_prompt_context(
                message=message,
                template_id=template_id,
                conversation_id=active_conversation_id,
                persona_id=persona_id,
            )
        except Exception as exc:
            logger.warning("Context-enriched prompt assembly failed, using fallback: %s", exc)
            prompt_context = {
                "system_prompt": FALLBACK_SYSTEM_PROMPT,
                "entities": {},
                "persona_id": persona_id or DEFAULT_PERSONA,
                "context_packet": "",
            }

        ai_error = False
        try:
            ai_response = self.generate_reply(
                system_prompt=prompt_context["system_prompt"],
                message=message,
                history=normalized_history,
            )
        except Exception as exc:
            logger.error("Anthropic API call failed: %s", exc)
            ai_response = GENERATION_FAILURE_REPLY
            ai_error = True

        try:
            persistence = self.persist_conversation(
                conversation_id=active_conversation_id,
                message=message,
                ai_response=ai_response,
            )
        except Exception as exc:
            logger.warning("Graph write for live conversation failed (non-fatal): %s", exc)
            persistence = {
                "graph_updated": False,
                "graph_summary": None,
                "conversation_title": None,
            }

        try:
            signal_metadata = self.generate_and_link_signals(
                conversation_id=active_conversation_id,
                message=message,
                history=normalized_history,
                graph_updated=persistence["graph_updated"],
            )
        except Exception as exc:
            logger.warning("Signal generation failed (non-fatal): %s", exc)
            signal_metadata = {
                "signal_count": 0,
                "addresses": [],
                "emotions": [],
                "observation_biases": [],
                "error": str(exc),
            }

        self.write_pipeline_trace(
            conversation_id=active_conversation_id,
            context_metadata=prompt_context,
            signal_metadata=signal_metadata,
        )

        return {
            "response": ai_response,
            "error": ai_error,
            "conversation_id": active_conversation_id,
            "conversation_title": persistence["conversation_title"],
            "graph_updated": persistence["graph_updated"],
            "graph_summary": persistence["graph_summary"],
            "template": template_id,
            "signals": signal_metadata,
            "context_assembly": {
                "entities": prompt_context.get("entities", {}),
                "persona": prompt_context.get("persona_id"),
                "context_packet_length": len(prompt_context.get("context_packet", "")),
            },
        }

    def build_prompt_context(
        self,
        *,
        message: str,
        template_id: str | None,
        conversation_id: str,
        persona_id: str | None,
    ) -> dict:
        assembler = ContextAssembler()
        entities = assembler.extract_entities(message)

        resolved_persona_id = persona_id or DEFAULT_PERSONA
        try:
            persona = get_persona(resolved_persona_id)
        except KeyError:
            resolved_persona_id = DEFAULT_PERSONA
            persona = get_persona(DEFAULT_PERSONA)

        # ── Full pipeline: graph reads + context assembly ──
        graph_context = assembler.enrich_from_graph(
            entities,
            self.workspace_id,
            persona,
            conversation_id=conversation_id,
        )
        context_packet = assemble_context(
            entities,
            signals=graph_context.get("signals"),
            clusters=graph_context.get("clusters"),
            insights=graph_context.get("insights"),
            bias_flags=graph_context.get("bias_flags"),
            token_budget=persona.context_token_budget,
        )

        # ── Enrich with Journey civic identity (best-effort) ──
        context_packet = self._inject_journey_context(context_packet)

        base_prompt = THERAPEUTIC_TEMPLATES.get(template_id or "", DEFAULT_BASE_PROMPT)
        system_prompt = build_system_prompt(
            base_prompt=base_prompt,
            persona_modifier=persona.system_prompt_modifier,
            context_packet=context_packet,
        )
        return {
            "system_prompt": system_prompt,
            "entities": entities,
            "persona_id": resolved_persona_id,
            "context_packet": context_packet,
        }

    def _inject_journey_context(self, context_packet: str) -> str:
        """
        Best-effort: append Journey civic identity to the context packet.

        If the user has completed any part of their Journey, this adds a
        compact summary of their path, soul ordering, value hierarchy,
        and regime type. Never blocks or raises on failure.
        """
        try:
            journey_state = get_journey_state_from_graph(
                workspace_id=self.workspace_id,
                owner_user_id=self.user.id,
            )
            value_profile = compute_value_profile(
                workspace_id=self.workspace_id,
                owner_user_id=self.user.id,
            )
            soul_profile = compute_soul_profile(
                workspace_id=self.workspace_id,
                owner_user_id=self.user.id,
            )
            journey_block = format_journey_context(
                journey_state, value_profile, soul_profile,
            )
            if journey_block:
                return f"{context_packet}\n\n{journey_block}" if context_packet else journey_block
        except Exception as exc:
            logger.debug("Journey context injection skipped: %s", exc)

        return context_packet

    def generate_reply(self, *, system_prompt: str, message: str, history: list[dict]) -> str:
        api_messages = []
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                api_messages.append({"role": role, "content": content})
        api_messages.append({"role": "user", "content": message})
        from .llm_client import DEFAULT_MODEL
        return generate_conversation_reply(
            system_prompt=system_prompt,
            messages=api_messages,
            model=settings.LLM_MODEL or DEFAULT_MODEL,
            max_tokens=800,
        )

    def persist_conversation(self, *, conversation_id: str, message: str, ai_response: str) -> dict:
        from .live_graph import build_live_conversation_title, write_live_conversation_to_graph

        conversation_title = build_live_conversation_title(message)
        graph_summary = write_live_conversation_to_graph(
            conversation_id=conversation_id,
            message=message,
            ai_response=ai_response,
            workspace_id=self.workspace_id,
            owner_user_id=self.user.id,
            conversation_title=conversation_title,
            username=self.user.username,
            email=self.user.email or "",
        )
        return {
            "graph_updated": bool(graph_summary.get("updated")),
            "graph_summary": graph_summary,
            "conversation_title": conversation_title,
        }

    def generate_and_link_signals(
        self,
        *,
        conversation_id: str,
        message: str,
        history: list[dict],
        graph_updated: bool,
    ) -> dict:
        conversation_context = self._build_conversation_context(history)
        generator = SignalGenerator(
            use_llm=True,
            llm_client=True,
            workspace_id=self.workspace_id,
            current_username=self.user.username,
            owner_user_id=self.user.id,
        )
        result = generator.generate_from_message(
            message=message,
            conversation_context=conversation_context,
            participants=[],
        )
        signals = result.get("signals", [])

        if signals and graph_updated:
            self._link_signals_to_conversation(conversation_id=conversation_id, signals=signals)

        return {
            "signal_count": len(signals),
            "addresses": [signal.get("signal_address", "") for signal in signals],
            "emotions": self._extract_detected_emotions(signals),
            "observation_biases": self._extract_bias_flags(signals),
        }

    def write_pipeline_trace(self, *, conversation_id: str, context_metadata: dict, signal_metadata: dict) -> None:
        try:
            from .live_graph import write_pipeline_trace

            context_packet = context_metadata.get("context_packet", "")
            write_pipeline_trace(
                workspace_id=self.workspace_id,
                conversation_id=conversation_id,
                entities=context_metadata.get("entities", {}),
                persona=context_metadata.get("persona_id"),
                context_summary=str(context_packet)[:500] if context_packet else "",
                signals_referenced=signal_metadata.get("signal_count", 0),
                clusters_referenced=0,
                token_count=len(str(context_packet)) if context_packet else 0,
            )
        except Exception as exc:
            logger.warning("PipelineTrace write failed (non-fatal): %s", exc)

    def _build_conversation_context(self, history: list[dict]) -> str | None:
        lines = []
        for item in history[-6:]:
            role = item.get("role", "user")
            content = (item.get("content", "") or "")[:200]
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else None

    def _link_signals_to_conversation(self, *, conversation_id: str, signals: list[dict]) -> None:
        from neomodel import db as neo_db

        for signal in signals:
            address = signal.get("signal_address", "")
            if not address:
                continue
            neo_db.cypher_query(
                """
                MATCH (c:Conversation {conversation_id: $conv_id, workspace_id: $ws})
                MATCH (s:Signal {signal_address: $addr, workspace_id: $ws})
                WHERE NOT (c)-[:CONTAINS_SIGNAL]->(s)
                MERGE (c)-[:CONTAINS_SIGNAL]->(s)
                """,
                {
                    "conv_id": conversation_id,
                    "ws": self.workspace_id,
                    "addr": address,
                },
            )

    def _extract_detected_emotions(self, signals: list[dict]) -> list[dict]:
        detected = []
        seen_names = set()
        for signal in signals:
            raw_emotions = signal.get("emotions", [])
            if isinstance(raw_emotions, str):
                try:
                    raw_emotions = json.loads(raw_emotions)
                except (json.JSONDecodeError, TypeError):
                    raw_emotions = []
            for emotion in raw_emotions if isinstance(raw_emotions, list) else []:
                if isinstance(emotion, dict):
                    name = (emotion.get("emotion") or emotion.get("name") or "").strip()
                    intensity = emotion.get("intensity")
                else:
                    name = str(emotion).strip()
                    intensity = None
                if name and name not in seen_names:
                    seen_names.add(name)
                    detected.append({"name": name, "intensity": intensity})
        return detected

    def _extract_bias_flags(self, signals: list[dict]) -> list[str]:
        flags = []
        seen_flags = set()
        for signal in signals:
            raw_biases = signal.get("observation_bias_flags", [])
            if isinstance(raw_biases, str):
                try:
                    raw_biases = json.loads(raw_biases)
                except (json.JSONDecodeError, TypeError):
                    raw_biases = []
            for flag in raw_biases if isinstance(raw_biases, list) else []:
                flag_text = flag if isinstance(flag, str) else str(flag)
                if flag_text and flag_text not in seen_flags:
                    seen_flags.add(flag_text)
                    flags.append(flag_text)
        return flags
