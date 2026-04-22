import json
import logging
from datetime import datetime, time

from django.db import connection
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from rest_framework import status
from rest_framework.response import Response

from .models import Event, Pattern, PatternRun
from .serializers import (
    AskRequestSerializer,
    EventSerializer,
    PatternRecomputeRequestSerializer,
    PatternSerializer,
)
from .services import answer_question, recompute_pattern_candidates
from .conversation import analyze_conversation
from .pattern_engine import detect_conversation_patterns
from .trajectory import TrajectoryComputer
from .reframe import ReframeGenerator
from .safety import SafetyAwareness
from .neo4j_client import cypher_query
from .graph_tests import build_graph_tests_payload
from .live_conversation_service import LiveConversationService
from .workspaces import WorkspaceAPIView

logger = logging.getLogger(__name__)

EMOTIONAL_NODE_FILTER = (
    "coalesce(trim(n.name), '') <> '' "
    "AND trim(n.name) <> '*' "
    "AND NOT n.name CONTAINS '*'"
)


def _emotional_node_filter(alias: str = "n") -> str:
    return EMOTIONAL_NODE_FILTER.replace("n.", f"{alias}.")


def _normalize_graph_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _normalize_graph_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_graph_value(item) for item in value]

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()

    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return iso_format()

    return str(value)


def _parse_filter_datetime(value: str, *, end_of_day: bool = False) -> datetime | None:
    parsed_datetime = parse_datetime(value)
    if parsed_datetime is not None:
        return timezone.make_aware(parsed_datetime) if timezone.is_naive(parsed_datetime) else parsed_datetime

    parsed_date = parse_date(value)
    if parsed_date is None:
        return None

    base_time = time.max if end_of_day else time.min
    parsed = datetime.combine(parsed_date, base_time)
    return timezone.make_aware(parsed)


def _filter_by_tag(queryset, tag: str):
    if "sqlite" in connection.vendor:
        return [event for event in queryset if tag in (event.context_tags or [])]
    return queryset.filter(context_tags__contains=[tag])


def _graph_scope(request) -> dict[str, str | int]:
    workspace = request.workspace
    return {
        "workspace_id": str(workspace.id),
        "owner_user_id": request.user.id,
    }


def _parse_csv_param(value: str | None) -> list[str]:
    if not value:
        return []
    return [" ".join(part.strip().split()) for part in value.split(",") if part.strip()]


def _parse_bool_param(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class EventListCreateView(WorkspaceAPIView):
    def get(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        queryset = Event.objects.filter(workspace=workspace)

        from_value = request.query_params.get("from")
        if from_value:
            parsed_from = _parse_filter_datetime(from_value)
            if parsed_from is None:
                return Response({"detail": "Invalid 'from' datetime."}, status=status.HTTP_400_BAD_REQUEST)
            queryset = queryset.filter(occurred_at__gte=parsed_from)

        to_value = request.query_params.get("to")
        if to_value:
            parsed_to = _parse_filter_datetime(to_value, end_of_day=True)
            if parsed_to is None:
                return Response({"detail": "Invalid 'to' datetime."}, status=status.HTTP_400_BAD_REQUEST)
            queryset = queryset.filter(occurred_at__lte=parsed_to)

        emotion_value = request.query_params.get("emotion")
        if emotion_value:
            queryset = queryset.filter(emotion__iexact=emotion_value)

        tag_value = request.query_params.get("tag")
        if tag_value:
            queryset = _filter_by_tag(queryset, tag_value)

        serializer = EventSerializer(queryset, many=True)
        return Response(serializer.data)

    def post(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        serializer = EventSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(workspace=workspace, created_by=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def delete(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        deleted_events = Event.objects.filter(workspace=workspace).count()
        PatternRun.objects.filter(workspace=workspace).delete()
        Event.objects.filter(workspace=workspace).delete()
        return Response({"deleted_events": deleted_events})


class EventDetailView(WorkspaceAPIView):
    def get(self, request, pk):
        workspace = self.get_workspace()
        request.workspace = workspace
        event = get_object_or_404(Event, pk=pk, workspace=workspace)
        serializer = EventSerializer(event)
        return Response(serializer.data)


class PatternRecomputeView(WorkspaceAPIView):
    def post(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        params_serializer = PatternRecomputeRequestSerializer(data=request.data or {})
        params_serializer.is_valid(raise_exception=True)
        params = params_serializer.validated_data

        patterns_data, event_count = recompute_pattern_candidates(
            workspace=workspace,
            max_patterns=params["max_patterns"],
            evidence_per_pattern=params["evidence_per_pattern"],
        )

        run = PatternRun.objects.create(
            workspace=workspace,
            created_by=request.user,
            params={
                "max_patterns": params["max_patterns"],
                "evidence_per_pattern": params["evidence_per_pattern"],
            },
            event_count=event_count,
        )

        Pattern.objects.bulk_create(
            [
                Pattern(
                    run=run,
                    key=pattern["key"],
                    name=pattern["name"],
                    hypothesis=pattern["hypothesis"],
                    score=pattern["score"],
                    evidence=pattern["evidence"],
                )
                for pattern in patterns_data
            ]
        )

        patterns = run.patterns.all().order_by("-score", "name")
        serialized_patterns = PatternSerializer(patterns, many=True).data

        return Response({"run_id": str(run.id), "patterns": serialized_patterns})


class PatternListView(WorkspaceAPIView):
    def get(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        latest_run = PatternRun.objects.filter(workspace=workspace).order_by("-created_at").first()
        if not latest_run:
            return Response([])

        patterns = latest_run.patterns.all().order_by("-score", "name")
        serializer = PatternSerializer(patterns, many=True)
        return Response(serializer.data)


class PatternDetailView(WorkspaceAPIView):
    def get(self, request, pk):
        workspace = self.get_workspace()
        request.workspace = workspace
        pattern = get_object_or_404(Pattern, pk=pk, run__workspace=workspace)
        serializer = PatternSerializer(pattern)
        return Response(serializer.data)


class AskView(WorkspaceAPIView):
    def post(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        serializer = AskRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data["question"]
        focus_event = None
        focus_event_id = serializer.validated_data.get("focus_event_id")

        if focus_event_id is not None:
            focus_event = Event.objects.filter(id=focus_event_id, workspace=workspace).first()
            if focus_event is None:
                return Response(
                    {"detail": "focus_event_id does not reference an existing event."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        payload = answer_question(question=question, workspace=workspace, focus_event=focus_event)
        return Response(payload)


class AnalysisView(WorkspaceAPIView):
    """
    POST endpoint for full conversation analysis pipeline.

    Accepts:
    - Raw text in request body, or
    - JSON with {"text": "...conversation..."}

    Returns:
    - Full Analysis Response: conversation + signals + patterns + trajectory + reframe + metadata
    """

    def post(self, request):
        try:
            workspace = self.get_workspace()
            request.workspace = workspace
            # Extract conversation text from request
            conversation_text = self._extract_text(request)
            if not conversation_text:
                return Response(
                    {"detail": "No conversation text provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            logger.info(f"Starting analysis for {len(conversation_text)} characters")

            # Stage 1: Parse and generate signals + trigger actions
            try:
                analysis_result = analyze_conversation(
                    conversation_text,
                    source_type="text",
                    write_to_graph=True,
                    graph_scope=_graph_scope(request),
                )
                conversation = analysis_result.get("conversation", {})
                signals = analysis_result.get("signals", [])
            except Exception as e:
                logger.error(f"Conversation analysis failed: {e}")
                return Response(
                    {"detail": f"Conversation analysis failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            # Get metadata ref for stage tracking
            pipeline_metadata = analysis_result.get("metadata", {})
            stages = pipeline_metadata.setdefault("stages", {})

            # Stage 2: Detect patterns
            try:
                patterns = detect_conversation_patterns(signals)
                # Check if any pattern has LLM-generated names (non-template)
                has_llm_names = any(
                    "–" not in p.get("pattern_name", "") or
                    not p.get("pattern_name", "").endswith("Cycle")
                    for p in patterns
                ) if patterns else False
                stages["patterns"] = {"method": "llm" if has_llm_names else "deterministic", "count": len(patterns)}
                logger.info(f"Detected {len(patterns)} patterns")
            except Exception as e:
                logger.error(f"Pattern detection failed: {e}")
                patterns = []
                stages["patterns"] = {"method": "failed"}

            # Stage 3: Compute trajectory (always deterministic)
            try:
                trajectory_computer = TrajectoryComputer()
                trajectory = trajectory_computer.process(signals)
                stages["trajectory"] = {"method": "deterministic"}
                logger.info("Trajectory computed")
            except Exception as e:
                logger.error(f"Trajectory computation failed: {e}")
                trajectory = None
                stages["trajectory"] = {"method": "failed"}

            # Stage 4: Generate reframe
            try:
                reframe_gen = ReframeGenerator()
                reframe = reframe_gen.process(patterns)
                stages["reframe"] = {"method": "deterministic_fallback" if reframe_gen.did_fallback_apply() else "llm"}
                logger.info("Reframe generated")
            except Exception as e:
                logger.error(f"Reframe generation failed: {e}")
                reframe = None
                stages["reframe"] = {"method": "failed"}

            # Stage 5: Safety awareness check
            safety_result = None
            try:
                safety = SafetyAwareness()
                safety_result = safety.analyze(signals, patterns, conversation)
                if safety_result:
                    logger.info(f"Safety awareness: {safety_result['overall_severity']} severity, {len(safety_result['flags'])} flags")
            except Exception as e:
                logger.error(f"Safety awareness check failed: {e}")

            # Build response
            response_data = {
                "conversation": conversation,
                "signals": signals,
                "patterns": patterns,
                "trajectory": trajectory,
                "reframe": reframe,
                "safety_awareness": safety_result,
                "metadata": analysis_result.get("metadata", {}),
            }

            logger.info("Analysis pipeline complete")
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Unexpected error in analysis: {e}")
            return Response(
                {"detail": f"Unexpected error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _extract_text(self, request) -> str:
        """Extract conversation text from request body or JSON."""
        # Try JSON body first
        if isinstance(request.data, dict):
            text = request.data.get("text")
            if isinstance(text, str):
                return text.strip()

        # Try raw body as text
        try:
            if hasattr(request, "body"):
                body = request.body.decode("utf-8")
                if body.startswith("{"):
                    data = json.loads(body)
                    text = data.get("text")
                    if isinstance(text, str):
                        return text.strip()
                else:
                    return body.strip()
        except Exception:
            pass

        return ""


class AnalysisAskView(WorkspaceAPIView):
    """
    POST endpoint for asking follow-up questions about analysis results.

    Accepts: {"question": "...", "context": {analysis result object}}
    Returns: {"answer": "...", "citations": [...]}
    """

    def post(self, request):
        try:
            question = request.data.get("question", "").strip()
            context = request.data.get("context", {})

            if not question:
                return Response(
                    {"detail": "No question provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if not context:
                return Response(
                    {"detail": "No analysis context provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Build context summary for LLM
            signals = context.get("signals", [])
            patterns = context.get("patterns", [])
            trajectory = context.get("trajectory", {})
            reframe = context.get("reframe", {})

            context_summary = self._build_context_summary(signals, patterns, trajectory, reframe)

            # Call LLM
            from . import llm_client

            system_prompt = (
                "You are a behavioral signal analyst for ThriveSight, a counseling awareness tool. "
                "You have analyzed a conversation and the user is asking a follow-up question about the results. "
                "Answer based ONLY on the analysis data provided. Be specific, cite turn numbers, "
                "and reference detected patterns. Keep answers concise (2-4 sentences). "
                "Never assign blame. Externalize dynamics.\n\n"
                "Respond in JSON: {\"answer\": \"...\", \"citations\": [{\"turn_number\": N, \"speaker\": \"...\", \"text\": \"...\"}]}"
            )

            user_prompt = f"Analysis context:\n{context_summary}\n\nQuestion: {question}"

            try:
                result = llm_client.analyze_with_retry(system_prompt, user_prompt)
                if isinstance(result, dict):
                    return Response(result)
                return Response({"answer": str(result), "citations": []})
            except Exception as e:
                logger.error(f"Analysis ask LLM failed: {e}")
                return Response(
                    {"answer": "I couldn't process that question. Try rephrasing it.", "citations": []},
                )

        except Exception as e:
            logger.error(f"Analysis ask error: {e}")
            return Response(
                {"detail": f"Error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _build_context_summary(self, signals, patterns, trajectory, reframe):
        """Build a concise text summary of analysis results for the LLM."""
        parts = []

        # Signals summary
        if signals:
            parts.append("SIGNALS:")
            for s in signals[:20]:  # Limit to prevent token overflow
                parts.append(
                    f"  Turn {s.get('turn_number', '?')} ({s.get('speaker', '?')}): "
                    f"emotion={s.get('emotion', '?')}, intensity={s.get('intensity', '?')}, "
                    f"reaction={s.get('reaction', '?')}"
                )

        # Patterns summary
        if patterns:
            parts.append("\nPATTERNS:")
            for p in patterns:
                parts.append(
                    f"  {p.get('pattern_name', '?')}: score={p.get('score', '?')}, "
                    f"trigger={p.get('trigger_category', '?')}, "
                    f"response={p.get('response_emotion', '?')}, "
                    f"count={p.get('occurrence_count', '?')}"
                )

        # Trajectory summary
        if trajectory:
            parts.append("\nTRAJECTORY:")
            for speaker, data in trajectory.items():
                if isinstance(data, dict):
                    inflections = data.get("inflection_points", [])
                    if inflections:
                        parts.append(f"  {speaker}: {len(inflections)} inflection point(s)")

        # Reframe summary
        if reframe and isinstance(reframe, dict):
            parts.append(f"\nREFRAME: {reframe.get('text', 'N/A')[:200]}")

        return "\n".join(parts)


class RealtimeTokenView(WorkspaceAPIView):
    """
    POST /api/audio/realtime-token/
    Generate a single-use token for WebSocket-based realtime STT.
    Keeps the ElevenLabs API key server-side.
    """

    def post(self, request):
        try:
            from .elevenlabs_client import ElevenLabsClient

            client = ElevenLabsClient()
            token = client.get_realtime_token(ttl_seconds=900)
            return Response({"token": token})
        except ValueError as e:
            return Response(
                {"detail": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.error(f"Realtime token generation failed: {e}")
            return Response(
                {"detail": f"Token generation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AudioBatchTranscribeView(WorkspaceAPIView):
    """
    POST /api/audio/transcribe/
    Transcribe audio via ElevenLabs Scribe v2 with diarization + entity detection.

    Accepts multipart form data:
        - audio: audio file (wav, mp3, m4a, webm, ogg, etc.)
        - num_speakers: int (optional, default auto-detect)
        - diarize: bool (optional, default true)

    Returns:
        - transcript: formatted "Speaker 0: text\\nSpeaker 1: text" string
        - raw_segments: original ElevenLabs segments with timestamps
        - entities: structured entity detection results
        - speakers: list of detected speaker IDs
        - language: detected language code
    """

    def post(self, request):
        try:
            audio_file = request.FILES.get("audio")
            if not audio_file:
                return Response(
                    {"detail": "No audio file provided. Send as 'audio' in multipart form."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Parse options
            num_speakers_raw = request.data.get("num_speakers")
            num_speakers = int(num_speakers_raw) if num_speakers_raw else None
            diarize = request.data.get("diarize", "true").lower() != "false"

            logger.info(
                f"Audio transcription request: {audio_file.name} "
                f"({audio_file.size / 1024:.0f}KB), diarize={diarize}, "
                f"num_speakers={num_speakers}"
            )

            # Read file bytes
            audio_bytes = audio_file.read()

            # Transcribe via ElevenLabs
            from .elevenlabs_client import ElevenLabsClient

            client = ElevenLabsClient()
            result = client.transcribe_batch(
                file_bytes=audio_bytes,
                filename=audio_file.name or "audio.wav",
                diarize=diarize,
                num_speakers=num_speakers,
                entity_detection="all",
            )

            # Format for frontend
            from .transcript_formatter import (
                format_diarized_to_labeled,
                build_entity_summary,
                extract_speakers_from_segments,
            )

            segments = result.get("segments", [])
            raw_entities = result.get("entities", [])

            transcript = format_diarized_to_labeled(segments)
            entities = build_entity_summary(raw_entities)
            speakers = extract_speakers_from_segments(segments)

            return Response({
                "transcript": transcript,
                "raw_segments": segments,
                "entities": entities,
                "speakers": speakers,
                "language": result.get("language_code"),
                "full_text": result.get("text", ""),
            })

        except ValueError as e:
            return Response(
                {"detail": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.error(f"Audio batch transcription failed: {e}")
            return Response(
                {"detail": f"Transcription failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Character Profile endpoints
# ──────────────────────────────────────────────────────────────────────────────


class CharacterProfileListCreateView(WorkspaceAPIView):
    """
    GET  /api/profiles/           → List all character profiles
    POST /api/profiles/           → Create a new character profile

    Falls back to file-based storage when Memgraph is unavailable.
    """

    FALLBACK_FILE = "character_profiles.json"

    @staticmethod
    def _fallback_path():
        import os
        from django.conf import settings
        return os.path.join(settings.BASE_DIR, "..", "..", "datasets", CharacterProfileListCreateView.FALLBACK_FILE)

    @staticmethod
    def _load_fallback():
        import os
        path = CharacterProfileListCreateView._fallback_path()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    @staticmethod
    def _save_fallback(profiles):
        import os
        path = CharacterProfileListCreateView._fallback_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)

    @staticmethod
    def _try_graph():
        """Return True if Memgraph is reachable."""
        try:
            from .graph_models import CharacterProfile
            CharacterProfile.nodes.all()[:0]  # minimal query
            return True
        except Exception:
            return False

    def get(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        workspace_id = str(workspace.id)
        # Try graph DB first
        if self._try_graph():
            from .graph_models import CharacterProfile
            try:
                profiles = CharacterProfile.nodes.filter(workspace_id=workspace_id)
                data = []
                for p in profiles:
                    conversations = list(p.conversations) if isinstance(p.conversations, list) else list(p.conversations.all())
                    data.append({
                        "uid": p.uid,
                        "name": p.name,
                        "source_work": p.source_work,
                        "description": p.description,
                        "profile_image_url": p.profile_image_url,
                        "created_at": str(p.created_at) if p.created_at else None,
                        "scene_count": len(conversations),
                    })
                return Response(data)
            except Exception as e:
                logger.warning(f"Graph DB query failed, falling back to file: {e}")

        # Fallback: file-based profiles
        logger.info("Using file-based profile storage (Memgraph unavailable)")
        profiles = [
            profile for profile in self._load_fallback()
            if profile.get("workspace_id") == workspace_id
        ]
        return Response(profiles)

    def post(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        workspace_id = str(workspace.id)
        name = request.data.get("name", "").strip()
        source_work = request.data.get("source_work", "").strip()
        description = request.data.get("description", "").strip()

        if not name or not source_work:
            return Response(
                {"detail": "Both 'name' and 'source_work' are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Try graph DB first
        if self._try_graph():
            from .graph_models import CharacterProfile
            try:
                profile = CharacterProfile(
                    workspace_id=workspace_id,
                    owner_user_id=request.user.id,
                    name=name,
                    source_work=source_work,
                    description=description or None,
                )
                profile.save()
                return Response({
                    "uid": profile.uid,
                    "name": profile.name,
                    "source_work": profile.source_work,
                    "description": profile.description,
                    "created_at": str(profile.created_at),
                }, status=status.HTTP_201_CREATED)
            except Exception as e:
                logger.warning(f"Graph create failed, falling back to file: {e}")

        # Fallback: file-based create
        import uuid as _uuid
        from datetime import datetime as _dt
        profiles = self._load_fallback()
        new_profile = {
            "uid": str(_uuid.uuid4()),
            "workspace_id": workspace_id,
            "owner_user_id": request.user.id,
            "name": name,
            "source_work": source_work,
            "description": description or None,
            "profile_image_url": None,
            "created_at": _dt.now().isoformat(),
            "scene_count": 0,
        }
        profiles.append(new_profile)
        self._save_fallback(profiles)
        return Response(new_profile, status=status.HTTP_201_CREATED)


class CharacterProfileDetailView(WorkspaceAPIView):
    """
    GET /api/profiles/<uid>/      → Get profile detail with aggregated scene data.
    Falls back to file-based storage when Memgraph is unavailable.
    """

    def get(self, request, uid):
        workspace = self.get_workspace()
        request.workspace = workspace
        workspace_id = str(workspace.id)
        # Try graph first
        if CharacterProfileListCreateView._try_graph():
            from .graph_models import CharacterProfile
            try:
                profile = CharacterProfile.nodes.get(uid=uid, workspace_id=workspace_id)
            except CharacterProfile.DoesNotExist:
                return Response(
                    {"detail": "Character profile not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            # Fallback: file-based lookup
            profiles = CharacterProfileListCreateView._load_fallback()
            match = [
                p for p in profiles
                if p["uid"] == uid and p.get("workspace_id") == workspace_id
            ]
            if not match:
                return Response(
                    {"detail": "Character profile not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            p = match[0]
            return Response({
                "uid": p["uid"],
                "name": p["name"],
                "source_work": p["source_work"],
                "description": p.get("description"),
                "profile_image_url": p.get("profile_image_url"),
                "created_at": p.get("created_at"),
                "scene_count": p.get("scene_count", 0),
                "scenes": [],
            })

        conversations = list(profile.conversations) if isinstance(profile.conversations, list) else list(profile.conversations.all())
        scenes = []
        for conv in conversations:
            scenes.append({
                "uid": conv.uid,
                "title": conv.title,
                "episode_reference": conv.episode_reference,
                "narrative_order": conv.narrative_order,
                "created_at": str(conv.created_at) if conv.created_at else None,
            })

        scenes.sort(key=lambda s: s.get("narrative_order") or 0)

        return Response({
            "uid": profile.uid,
            "name": profile.name,
            "source_work": profile.source_work,
            "description": profile.description,
            "profile_image_url": profile.profile_image_url,
            "created_at": str(profile.created_at) if profile.created_at else None,
            "scenes": scenes,
            "scene_count": len(scenes),
        })


class CharacterProfileAnalyzeView(WorkspaceAPIView):
    """
    POST /api/profiles/<uid>/analyze/

    Batch-analyze a scene transcript and link it to the character profile.

    Body:
        - text: The scene transcript
        - episode_reference: e.g., "S1E1" (optional)
        - narrative_order: integer for ordering (optional)
        - title: Scene title (optional, auto-generated if missing)
    """

    def post(self, request, uid):
        from .graph_models import CharacterProfile, Conversation
        workspace = self.get_workspace()
        request.workspace = workspace
        workspace_id = str(workspace.id)

        # Look up the profile
        try:
            profile = CharacterProfile.nodes.get(uid=uid, workspace_id=workspace_id)
        except CharacterProfile.DoesNotExist:
            return Response(
                {"detail": "Character profile not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        text = request.data.get("text", "").strip()
        if not text:
            return Response(
                {"detail": "No scene transcript text provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        episode_ref = request.data.get("episode_reference", "")
        narrative_order = request.data.get("narrative_order")
        title = request.data.get("title", "").strip()

        if not title:
            title = f"{profile.name} — {episode_ref or 'Scene'}"

        logger.info(
            f"Analyzing scene for character '{profile.name}': "
            f"{title} ({len(text)} chars)"
        )

        try:
            # Run the full analysis pipeline with graph writes
            analysis_result = analyze_conversation(
                text,
                source_type="text",
                write_to_graph=True,
                graph_scope=_graph_scope(request),
            )

            # Link the conversation to the character profile in the graph
            conversation_data = analysis_result.get("conversation", {})
            conv_uid = conversation_data.get("uid")

            if conv_uid:
                try:
                    conv_node = Conversation.nodes.get(uid=conv_uid, workspace_id=workspace_id)
                    # Set episode metadata
                    if episode_ref:
                        conv_node.episode_reference = episode_ref
                    if narrative_order is not None:
                        conv_node.narrative_order = int(narrative_order)
                    if title:
                        conv_node.title = title
                    conv_node.save()
                    # Link to profile
                    profile.conversations.connect(conv_node)
                    logger.info(f"Linked conversation {conv_uid} to profile {profile.name}")
                except Exception as link_err:
                    logger.warning(f"Could not link conversation to profile: {link_err}")

            # Run subsequent pipeline stages
            signals = analysis_result.get("signals", [])
            metadata = analysis_result.get("metadata", {})
            stages = metadata.setdefault("stages", {})

            # Stage 2: Detect patterns
            try:
                patterns = detect_conversation_patterns(signals)
                has_llm_names = any(
                    "–" not in p.get("pattern_name", "") or
                    not p.get("pattern_name", "").endswith("Cycle")
                    for p in patterns
                ) if patterns else False
                stages["patterns"] = {"method": "llm" if has_llm_names else "deterministic", "count": len(patterns)}
            except Exception as e:
                logger.error(f"Pattern detection failed for scene: {e}")
                patterns = []
                stages["patterns"] = {"method": "failed"}

            # Stage 3: Compute trajectory (always deterministic)
            try:
                trajectory_computer = TrajectoryComputer()
                trajectory = trajectory_computer.process(signals)
                stages["trajectory"] = {"method": "deterministic"}
            except Exception as e:
                logger.error(f"Trajectory computation failed for scene: {e}")
                trajectory = None
                stages["trajectory"] = {"method": "failed"}

            # Stage 4: Generate reframe
            try:
                reframe_gen = ReframeGenerator()
                reframe = reframe_gen.process(patterns)
                stages["reframe"] = {"method": "deterministic_fallback" if reframe_gen.did_fallback_apply() else "llm"}
            except Exception as e:
                logger.error(f"Reframe generation failed for scene: {e}")
                reframe = None
                stages["reframe"] = {"method": "failed"}

            # Stage 5: Safety awareness check
            safety_result = None
            try:
                safety_checker = SafetyAwareness()
                safety_result = safety_checker.analyze(signals, patterns, conversation_data)
                if safety_result:
                    logger.info(f"Safety awareness: {safety_result['overall_severity']} severity")
            except Exception as e:
                logger.error(f"Safety awareness check failed for scene: {e}")

            return Response({
                "profile_uid": profile.uid,
                "character_name": profile.name,
                "scene_title": title,
                "episode_reference": episode_ref,
                "conversation": conversation_data,
                "signals": signals,
                "patterns": patterns,
                "trajectory": trajectory,
                "reframe": reframe,
                "safety_awareness": safety_result,
                "metadata": metadata,
            })

        except Exception as e:
            logger.error(f"Scene analysis failed for profile {profile.name}: {e}")
            return Response(
                {"detail": f"Scene analysis failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2: AI REFLECTION CONVERSATION
# ═══════════════════════════════════════════════════════════════════════


class ReflectionStartView(WorkspaceAPIView):
    """
    POST /api/reflection/start/

    Begin a reflection conversation. Receives the full analysis result
    and generates the opening question targeting the biggest emotional moment.

    Body: {"analysis_result": {full analysis payload from AnalysisView}}
    Returns: {"question": str, "reasoning": str, "target_turn": int | null}
    """

    def post(self, request):
        from .reflection import ReflectionEngine

        analysis_result = request.data.get("analysis_result")
        if not analysis_result or not isinstance(analysis_result, dict):
            return Response(
                {"detail": "No analysis_result provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        signals = analysis_result.get("signals", [])
        if not signals:
            return Response(
                {"detail": "Analysis result contains no signals."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(
            f"Starting reflection session ({len(signals)} signals, "
            f"{len(analysis_result.get('patterns', []))} patterns)"
        )

        try:
            engine = ReflectionEngine()
            result = engine.generate_opening_question(analysis_result)
            return Response(result)
        except Exception as e:
            logger.error(f"Reflection start failed: {e}")
            return Response(
                {"detail": f"Reflection start failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ScreenplayAnalyzeView(WorkspaceAPIView):
    """
    POST /api/screenplay/analyze/

    Full screenplay → character dataset pipeline.

    Accepts:
    - text: Raw screenplay text
    - film_title: Title of the film
    - focal_character: Character to build profile for
    - character_description: Brief description (optional)
    - known_characters: List of character names to aid detection (optional)
    - lenses: Which analytical lenses to apply (optional, default: all)
              Options: ["dialogue", "plot", "psychology", "relational", "context"]

    Returns: Complete character dataset with multi-lens analysis.
    """

    def post(self, request):
        from .character_analyzer import build_character_dataset

        text = request.data.get("text", "").strip()
        film_title = request.data.get("film_title", "Untitled")
        focal_character = request.data.get("focal_character", "").strip()
        character_description = request.data.get("character_description", "")
        known_characters = request.data.get("known_characters", [])
        lenses = request.data.get("lenses")

        if not text:
            return Response(
                {"detail": "No screenplay text provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not focal_character:
            return Response(
                {"detail": "No focal_character specified."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(
            f"Screenplay analysis: {film_title}, focal={focal_character}, "
            f"{len(text)} chars, lenses={lenses or 'all'}"
        )

        try:
            dataset = build_character_dataset(
                raw_screenplay=text,
                film_title=film_title,
                focal_character=focal_character,
                character_description=character_description,
                known_characters=known_characters,
                lenses=lenses,
            )
            return Response(dataset, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Screenplay analysis failed: {e}")
            return Response(
                {"detail": f"Screenplay analysis failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ScreenplayParseView(WorkspaceAPIView):
    """
    POST /api/screenplay/parse/

    Parse a screenplay into structured scenes WITHOUT running analysis.
    Useful for previewing scene extraction before committing to full analysis.

    Accepts:
    - text: Raw screenplay text
    - film_title: Title of the film (optional)
    - known_characters: List of character names (optional)

    Returns: Parsed screenplay with scenes, characters, and metadata.
    """

    def post(self, request):
        from .screenplay_parser import ScreenplayParser

        text = request.data.get("text", "").strip()
        film_title = request.data.get("film_title", "Untitled")
        known_characters = request.data.get("known_characters", [])

        if not text:
            return Response(
                {"detail": "No screenplay text provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(f"Screenplay parse: {film_title}, {len(text)} chars")

        try:
            parser = ScreenplayParser(known_characters=known_characters)
            result = parser.parse(text, film_title=film_title)

            # Return a summary (full scene content can be large)
            scene_summaries = []
            for scene in result.get("scenes", []):
                scene_summaries.append({
                    "scene_number": scene["scene_number"],
                    "slugline": scene["slugline"],
                    "location": scene["location"],
                    "time_of_day": scene["time_of_day"],
                    "characters_present": scene["characters_present"],
                    "witnesses": scene["witnesses"],
                    "dialogue_turn_count": len(scene["dialogue_turns"]),
                    "action_line_count": len(scene["action_lines"]),
                })

            return Response({
                "film_title": result["film_title"],
                "total_scenes": result["total_scenes"],
                "total_dialogue_turns": result["total_dialogue_turns"],
                "characters": result["characters"],
                "scenes": scene_summaries,
                "parse_metadata": result["parse_metadata"],
            })
        except Exception as e:
            logger.error(f"Screenplay parse failed: {e}")
            return Response(
                {"detail": f"Screenplay parse failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ScreenplayDatasetListView(WorkspaceAPIView):
    """
    GET /api/screenplay/datasets/
    List available pre-parsed screenplay dataset files from the datasets/ directory.
    """

    def get(self, request):
        import os
        from django.conf import settings

        datasets_dir = os.path.join(settings.BASE_DIR, "..", "..", "datasets")
        datasets_dir = os.path.normpath(datasets_dir)

        if not os.path.isdir(datasets_dir):
            return Response([])

        results = []
        for fname in sorted(os.listdir(datasets_dir)):
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(datasets_dir, fname)
            try:
                size_kb = round(os.path.getsize(filepath) / 1024, 1)
                # Read just the character metadata from the file
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                character_info = data.get("character", {})
                entry = {
                    "filename": fname,
                    "size_kb": size_kb,
                    "character": character_info.get("name", ""),
                    "source_work": character_info.get("source_work", ""),
                    "description": character_info.get("description", ""),
                    "scene_count": len(data.get("scenes", [])),
                    "total_scenes": data.get("total_scenes", len(data.get("scenes", []))),
                }
                # For full parsed files, include character list
                if "characters" in data:
                    entry["characters"] = data["characters"][:15]
                    entry["total_dialogue_turns"] = data.get("total_dialogue_turns", 0)
                results.append(entry)
            except Exception:
                results.append({"filename": fname, "size_kb": size_kb, "error": "Could not read metadata"})

        return Response(results)


class ScreenplayDatasetDetailView(WorkspaceAPIView):
    """
    GET /api/screenplay/datasets/<filename>/
    Return the full contents of a pre-parsed screenplay dataset JSON file.
    """

    def get(self, request, filename):
        import os
        import re as _re
        from django.conf import settings

        # Validate filename (prevent path traversal)
        if not _re.match(r'^[a-zA-Z0-9_\-]+\.json$', filename):
            return Response(
                {"detail": "Invalid filename."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        datasets_dir = os.path.join(settings.BASE_DIR, "..", "..", "datasets")
        filepath = os.path.normpath(os.path.join(datasets_dir, filename))

        if not filepath.startswith(os.path.normpath(datasets_dir)):
            return Response(
                {"detail": "Invalid filename."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not os.path.isfile(filepath):
            return Response(
                {"detail": f"Dataset file not found: {filename}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Response(data)
        except Exception as e:
            logger.error(f"Failed to read dataset {filename}: {e}")
            return Response(
                {"detail": f"Failed to read dataset: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CharacterNetworkBuildView(WorkspaceAPIView):
    """
    POST /api/screenplay/datasets/<filename>/network/

    Run the allocation engine on a parsed screenplay dataset, write the
    results to Neo4j, and return the character network data for visualization.

    Query params:
        write_graph=false  — skip graph writing (default: true)
    """

    def post(self, request, filename):
        import os
        import re as _re
        from django.conf import settings
        from .character_network import build_network, write_to_graph_with_scenes
        workspace = self.get_workspace()
        request.workspace = workspace

        # Validate filename
        if not _re.match(r'^[a-zA-Z0-9_\-]+\.json$', filename):
            return Response(
                {"detail": "Invalid filename."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        datasets_dir = os.path.join(settings.BASE_DIR, "..", "..", "datasets")
        filepath = os.path.normpath(os.path.join(datasets_dir, filename))

        if not filepath.startswith(os.path.normpath(datasets_dir)):
            return Response({"detail": "Invalid filename."}, status=status.HTTP_400_BAD_REQUEST)

        if not os.path.isfile(filepath):
            return Response({"detail": f"Dataset not found: {filename}"}, status=status.HTTP_404_NOT_FOUND)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                skeleton_data = json.load(f)

            result = build_network(skeleton_data)

            # Write to Neo4j graph (unless explicitly disabled)
            write_graph = request.query_params.get("write_graph", "true").lower() != "false"
            graph_summary = None
            if write_graph:
                try:
                    source_work_data = {
                        "year": skeleton_data.get("year"),
                        "medium": skeleton_data.get("medium", "film"),
                        "ontology_type": skeleton_data.get("ontology_type", "fictional"),
                    }
                    graph_summary = write_to_graph_with_scenes(
                        result,
                        skeleton_data,
                        source_work_data,
                        workspace_id=str(workspace.id),
                        owner_user_id=request.user.id,
                    )
                    logger.info(f"Graph write complete: {graph_summary}")
                except Exception as e:
                    logger.warning(f"Graph write failed (returning results anyway): {e}")
                    graph_summary = {"error": str(e)}

            # Serialize for frontend
            relationships = []
            for rel in result.relationships:
                relationships.append({
                    "source": rel.source_character,
                    "target": rel.target_character,
                    "co_occurrence_count": rel.co_occurrence_count,
                    "shared_dialogue_scenes": rel.shared_dialogue_scenes,
                    "witness_scenes": rel.witness_scenes,
                    "first_observed_scene": rel.first_observed_scene,
                    "last_observed_scene": rel.last_observed_scene,
                    "significance_score": rel.significance_score,
                })

            characters = {}
            for name, char_data in result.characters.items():
                vocab_json = char_data.build_vocabulary_json()
                top_words = sorted(
                    char_data.vocabulary.items(),
                    key=lambda x: x[1].count,
                    reverse=True,
                )[:20]
                characters[name] = {
                    "name": name,
                    "total_words": char_data.total_words,
                    "unique_words": char_data.unique_words,
                    "dialogue_turn_count": char_data.dialogue_turn_count,
                    "scene_count": len(char_data.scene_appearances),
                    "top_words": [
                        {"word": w, "count": e.count, "scenes": len(set(e.scenes))}
                        for w, e in top_words
                    ],
                }

            # DIRECTED_AT method distribution
            from collections import Counter
            method_dist = dict(Counter(da.inference_method for da in result.directed_at_results))

            response_data = {
                "summary": result.to_summary(),
                "relationships": relationships,
                "characters": characters,
                "directed_at_methods": method_dist,
                "scene_roles": {str(k): v for k, v in result.scene_roles.items()},
                "graph_write": graph_summary,
            }

            return Response(response_data)

        except Exception as e:
            logger.error(f"Network build failed for {filename}: {e}")
            return Response(
                {"detail": f"Network build failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ReflectionNextView(WorkspaceAPIView):
    """
    POST /api/reflection/next/

    Generate the next reflection question based on user response + signal data.

    Body: {
        "user_response": str,
        "analysis_result": {full analysis payload},
        "reflection_history": [{"role": "ai"|"user", "text": str}, ...],
        "exchange_count": int
    }
    Returns: {"question": str, "question_type": str, "should_suggest_reveal": bool}
    """

    def post(self, request):
        from .reflection import ReflectionEngine

        user_response = request.data.get("user_response", "").strip()
        analysis_result = request.data.get("analysis_result")
        reflection_history = request.data.get("reflection_history", [])
        exchange_count = request.data.get("exchange_count", 0)

        if not user_response:
            return Response(
                {"detail": "No user_response provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not analysis_result or not isinstance(analysis_result, dict):
            return Response(
                {"detail": "No analysis_result provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(
            f"Reflection followup (exchange {exchange_count}, "
            f"history length {len(reflection_history)})"
        )

        try:
            engine = ReflectionEngine()
            result = engine.generate_followup(
                user_response=user_response,
                analysis_result=analysis_result,
                reflection_history=reflection_history,
                exchange_count=exchange_count,
            )
            return Response(result)
        except Exception as e:
            logger.error(f"Reflection next failed: {e}")
            return Response(
                {"detail": f"Reflection question generation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# ═══════════════════════════════════════════════════════════════════════
# V5.0 — AI Conversation Import
# ═══════════════════════════════════════════════════════════════════════


class ConversationImportView(WorkspaceAPIView):
    """
    POST /api/import/conversations/

    Import AI conversation history from ChatGPT, Claude, or Gemini.
    Accepts the raw JSON export, parses it, extracts topics, and writes
    to the Neo4j knowledge graph.

    Request body: The raw JSON export file contents (as JSON).
    Optional query param: platform=chatgpt|claude (auto-detected if omitted)

    Returns: Import summary with conversation count, topic count, and graph write status.
    """

    def post(self, request):
        try:
            workspace = self.get_workspace()
            request.workspace = workspace
            from .import_parsers import (
                detect_platform,
                parse_chatgpt_export,
                parse_claude_export,
                extract_topics_from_conversations,
                write_import_to_graph,
            )

            raw_json = request.data
            if not isinstance(raw_json, list):
                return Response(
                    {"detail": "Expected a JSON array of conversations."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Detect or use specified platform
            platform = request.query_params.get("platform") or detect_platform(raw_json)

            if platform == "chatgpt":
                normalized = parse_chatgpt_export(raw_json)
            elif platform == "claude":
                normalized = parse_claude_export(raw_json)
            else:
                return Response(
                    {"detail": f"Unsupported or undetected platform: {platform}. Use ?platform=chatgpt or ?platform=claude"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if normalized["metadata"]["total_conversations"] == 0:
                return Response(
                    {"detail": "No conversations found in the export."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Extract topics
            topics = extract_topics_from_conversations(normalized)

            # Write to Neo4j
            graph_summary = write_import_to_graph(
                normalized,
                topics,
                workspace_id=str(workspace.id),
                owner_user_id=request.user.id,
            )

            return Response({
                "status": "success",
                "platform": platform,
                "metadata": normalized["metadata"],
                "topics_extracted": len(topics.get("topics", [])),
                "user_patterns": topics.get("user_patterns", {}),
                "graph_write": graph_summary,
                "sample_topics": [t["topic"] for t in topics.get("topics", [])[:15]],
            })

        except json.JSONDecodeError as e:
            return Response(
                {"detail": f"Invalid JSON: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.error(f"Conversation import failed: {e}")
            return Response(
                {"detail": f"Import failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GraphQueryView(WorkspaceAPIView):
    """
    POST /api/graph/query/

    Execute a pre-built insight query against the knowledge graph.

    Request body:
        { "query_type": "topics" | "patterns" | "mood" | "orphans" | "overview" }

    Returns: Query results formatted for frontend visualization.
    """

    QUERIES = {
        "overview": {
            "description": "What does my knowledge graph look like?",
            "cypher": """
                MATCH (n)
                WHERE n.workspace_id = $workspace_id
                WITH labels(n) AS types, count(*) AS count
                UNWIND types AS type
                RETURN type, sum(count) AS total
                ORDER BY total DESC
            """,
        },
        "topics": {
            "description": "What have I been thinking about most?",
            "cypher": """
                MATCH (t:Topic)
                WHERE t.workspace_id = $workspace_id
                RETURN t.word AS topic, t.total_count AS mentions,
                       t.conversation_count AS conversations
                ORDER BY t.total_count DESC
                LIMIT 20
            """,
        },
        "recent_topics": {
            "description": "What have I discussed recently?",
            "cypher": """
                MATCH (t:Topic)-[:DISCUSSED_IN]->(c:Conversation)
                WHERE c.create_time IS NOT NULL AND t.workspace_id = $workspace_id AND c.workspace_id = $workspace_id
                WITH t, c ORDER BY c.create_time DESC
                WITH t, collect(c.title)[..3] AS recent_convs, max(c.create_time) AS last_discussed
                RETURN t.word AS topic, t.total_count AS mentions,
                       recent_convs, last_discussed
                ORDER BY last_discussed DESC
                LIMIT 15
            """,
        },
        "patterns": {
            "description": "What topics keep coming back?",
            "cypher": """
                MATCH (t:Topic)-[:DISCUSSED_IN]->(c:Conversation)
                WHERE t.workspace_id = $workspace_id AND c.workspace_id = $workspace_id
                WITH t, count(c) AS conv_count
                WHERE conv_count >= 3
                RETURN t.word AS recurring_topic, conv_count AS conversations,
                       t.total_count AS total_mentions
                ORDER BY conv_count DESC
                LIMIT 15
            """,
        },
        "questions": {
            "description": "What questions do I ask most?",
            "cypher": """
                MATCH (c:Conversation)-[:CONTAINS]->(turn:UserTurn)
                WHERE c.workspace_id = $workspace_id
                  AND turn.workspace_id = $workspace_id
                  AND turn.has_question = true
                RETURN c.title AS conversation, turn.content_preview AS question,
                       turn.create_time AS asked_at
                ORDER BY turn.create_time DESC
                LIMIT 20
            """,
        },
        "conversation_topics": {
            "description": "What topics are active in this live reflection?",
            "cypher": """
                MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
                MATCH (t:Topic {workspace_id: $workspace_id})-[:DISCUSSED_IN]->(c)
                RETURN c.title AS conversation,
                       t.word AS topic,
                       t.total_count AS mentions,
                       t.conversation_count AS conversations
                ORDER BY t.total_count DESC
                LIMIT 20
            """,
        },
        "conversation_turns": {
            "description": "What turns have been stored for this live reflection?",
            "cypher": """
                MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})-[:CONTAINS]->(turn)
                WHERE turn:UserTurn OR turn:AssistantTurn
                RETURN c.title AS conversation,
                       CASE
                         WHEN turn:AssistantTurn THEN 'AssistantTurn'
                         ELSE 'UserTurn'
                       END AS turn_type,
                       turn.content_preview AS content,
                       coalesce(turn.has_question, false) AS has_question,
                       turn.create_time AS created_at
                ORDER BY turn.create_time DESC
                LIMIT 12
            """,
        },
        "orphans": {
            "description": "What topics did I only mention once?",
            "cypher": """
                MATCH (t:Topic)
                WHERE t.workspace_id = $workspace_id AND t.conversation_count = 1
                RETURN t.word AS orphan_topic, t.total_count AS mentions
                ORDER BY t.total_count DESC
                LIMIT 20
            """,
        },
        "conversation_list": {
            "description": "All conversations in this workspace",
            "cypher": """
                MATCH (c:Conversation {workspace_id: $workspace_id})
                RETURN c.conversation_id AS conversation_id,
                       c.title AS title,
                       coalesce(c.turn_count, 0) AS turn_count,
                       c.last_active AS last_active,
                       c.create_time AS create_time
                ORDER BY c.last_active DESC
            """,
        },
        "conversation_history": {
            "description": "Full message history for a conversation",
            "cypher": """
                MATCH (c:Conversation {conversation_id: $conversation_id, workspace_id: $workspace_id})
                      -[:CONTAINS]->(turn)
                WHERE turn:UserTurn OR turn:AssistantTurn
                WITH turn,
                     CASE WHEN 'UserTurn' IN labels(turn) THEN 'user' ELSE 'assistant' END AS role
                RETURN role,
                       coalesce(turn.content, turn.content_preview) AS content,
                       turn.create_time AS created_at
                ORDER BY turn.create_time ASC
            """,
        },
        "workspace_graph": {
            "description": "Full workspace knowledge graph for visualization",
            "handler": "_handle_workspace_graph",
        },
        "emotional_graph": {
            "description": "Emotional landscape with relational context",
            "handler": "_handle_emotional_graph",
        },
        "emotional_node_detail": {
            "description": "Additional context for a selected emotional graph node",
            "handler": "_handle_emotional_node_detail",
        },
    }

    # User-friendly label translations for the Insights tab
    LABEL_MAP = {
        "Signal": "Moment",
        "Cluster": "Recurring Pattern",
        "PipelineTrace": "How this was understood",
        "UserTurn": "Your message",
        "AssistantTurn": "AI response",
        "UserProfile": "You",
        "Conversation": "Conversation",
        "Topic": "Theme",
        "Person": "Person",
        "DataSource": "Source",
        "Pattern": "Pattern",
        "Insight": "Insight",
        "Reflection": "Reflection",
        "ContextNode": "Context",
        "ActionNode": "Action",
        "TemporalNode": "Time",
    }

    def post(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        query_type = request.data.get("query_type", "overview")
        conversation_id = request.data.get("conversation_id")

        if query_type not in self.QUERIES:
            return Response(
                {
                    "detail": f"Unknown query type: {query_type}",
                    "available": list(self.QUERIES.keys()),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        needs_conversation_id = query_type in ("conversation_topics", "conversation_turns", "conversation_history")
        if needs_conversation_id and not conversation_id:
            return Response(
                {"detail": "conversation_id is required for conversation-scoped graph queries."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        query_info = self.QUERIES[query_type]

        # ── Custom handler queries (workspace_graph, etc.) ───────────
        if "handler" in query_info:
            handler_name = query_info["handler"]
            handler = getattr(self, handler_name, None)
            if handler is None:
                return Response(
                    {"detail": f"Handler not found: {handler_name}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            try:
                return handler(str(workspace.id), request.data)
            except Exception as e:
                logger.error(f"Graph query handler failed: {e}")
                if query_type == "emotional_node_detail":
                    return Response(
                        {
                            "query_type": query_type,
                            "description": query_info["description"],
                            "node": None,
                            "related_conversations": [],
                            "related_signals": [],
                            "related_nodes": {},
                            "counts": {
                                "connected_signals": 0,
                                "connected_conversations": 0,
                            },
                            "status": "degraded",
                            "message": "Graph backend unavailable.",
                            "detail": str(e),
                        },
                        status=status.HTTP_200_OK,
                    )
                return Response(
                    {
                        "query_type": query_type,
                        "description": query_info["description"],
                        "nodes": [],
                        "edges": [],
                        "counts": {},
                        "status": "degraded",
                        "message": "Graph backend unavailable.",
                        "detail": str(e),
                    },
                    status=status.HTTP_200_OK,
                )

        # ── Standard Cypher queries ──────────────────────────────────
        try:
            query_params = {"workspace_id": str(workspace.id)}
            if conversation_id:
                query_params["conversation_id"] = conversation_id
            results, meta = cypher_query(query_info["cypher"], query_params)

            # Convert to list of dicts using column names
            columns = meta if isinstance(meta, list) else []
            rows = []
            for row in results:
                if columns:
                    rows.append(_normalize_graph_value(dict(zip(columns, row))))
                else:
                    rows.append(_normalize_graph_value(row))

            payload = {
                "query_type": query_type,
                "description": query_info["description"],
                "results": rows,
                "count": len(rows),
            }
            if query_type == "conversation_list":
                payload["conversations"] = rows
            elif query_type == "conversation_history":
                payload["turns"] = rows

            return Response(payload)

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            # Degrade gracefully when graph DB is unavailable so UI can still render
            # placeholder/empty-state visuals instead of hard-failing.
            return Response(
                {
                    "query_type": query_type,
                    "description": query_info["description"],
                    "results": [],
                    "count": 0,
                    "status": "degraded",
                    "message": "Graph backend unavailable. Showing empty results.",
                    "detail": str(e),
                },
                status=status.HTTP_200_OK,
            )

    def _handle_workspace_graph(self, workspace_id, params):
        """
        Return all nodes and edges in the user's workspace for the Insights
        tab knowledge graph visualization.

        Supports optional filters:
            - node_types: list of Neo4j labels to include (e.g. ["Conversation", "Topic"])
            - since: epoch timestamp — only nodes created after this time
            - min_confidence: float — minimum confidence_score for Signal/Cluster nodes
        """
        import time as _time

        node_types = params.get("node_types")  # list or None
        since = params.get("since")  # epoch float or None
        min_confidence = params.get("min_confidence")  # float or None

        # ── Build dynamic WHERE clause ───────────────────────────────
        where_clauses = ["n.workspace_id = $workspace_id"]
        query_params = {"workspace_id": workspace_id}

        if node_types:
            # Filter to requested labels only
            label_checks = " OR ".join(f"n:{label}" for label in node_types if label.isalnum())
            if label_checks:
                where_clauses.append(f"({label_checks})")

        if since is not None:
            try:
                query_params["since"] = float(since)
                where_clauses.append(
                    "(n.create_time >= $since OR n.created_at >= $since OR n.create_time IS NULL)"
                )
            except (ValueError, TypeError):
                pass

        where_str = " AND ".join(where_clauses)

        # ── Query all nodes ──────────────────────────────────────────
        node_cypher = f"""
            MATCH (n)
            WHERE {where_str}
            WITH n, labels(n) AS lbls
            RETURN
                n.uid AS uid,
                lbls[0] AS kind,
                COALESCE(n.title, n.name, n.word, n.content_preview, n.username, n.description, '') AS label,
                n.create_time AS create_time,
                n.created_at AS created_at,
                n.confidence_score AS confidence_score,
                n.status AS status,
                n.persona_used AS persona_used,
                n.entities_extracted AS entities_extracted,
                n.context_packet_summary AS context_packet_summary,
                n.signals_referenced AS signals_referenced,
                n.clusters_referenced AS clusters_referenced,
                n.token_count AS token_count,
                n.emotions AS emotions,
                n.signal_address AS signal_address,
                n.provenance AS provenance,
                n.strength AS strength,
                n.total_count AS total_count,
                n.conversation_count AS conversation_count,
                n.role AS role,
                n.turn_count AS turn_count,
                n.word_count AS word_count,
                n.conversation_id AS conversation_id,
                n.owner_user_id AS owner_user_id
        """
        node_rows, node_cols = cypher_query(node_cypher, query_params)
        columns = node_cols if isinstance(node_cols, list) else []

        nodes = []
        counts = {}
        seen_uids = set()

        for row in node_rows:
            node_dict = dict(zip(columns, row)) if columns else {}
            uid = node_dict.get("uid")
            if not uid or uid in seen_uids:
                continue
            seen_uids.add(uid)

            raw_kind = node_dict.get("kind", "default")

            # Apply confidence filter for Signals and Clusters
            if min_confidence is not None and raw_kind in ("Signal", "Cluster"):
                conf = node_dict.get("confidence_score")
                if conf is not None and conf < float(min_confidence):
                    continue

            friendly_kind = self.LABEL_MAP.get(raw_kind, raw_kind)

            # Build properties dict (non-null values only)
            properties = {}
            for key in (
                "create_time", "created_at", "confidence_score", "status",
                "persona_used", "entities_extracted", "context_packet_summary",
                "signals_referenced", "clusters_referenced", "token_count",
                "emotions", "signal_address", "provenance", "strength",
                "total_count", "conversation_count", "role", "turn_count",
                "word_count", "conversation_id",
            ):
                val = node_dict.get(key)
                if val is not None:
                    properties[key] = _normalize_graph_value(val)

            nodes.append({
                "uid": uid,
                "label": node_dict.get("label", ""),
                "kind": friendly_kind,
                "raw_kind": raw_kind,
                "properties": properties,
            })

            # Count by friendly kind
            counts[friendly_kind] = counts.get(friendly_kind, 0) + 1

        # ── Query all edges between workspace nodes ──────────────────
        edge_cypher = f"""
            MATCH (a)-[r]->(b)
            WHERE a.workspace_id = $workspace_id AND b.workspace_id = $workspace_id
              AND a.uid IS NOT NULL AND b.uid IS NOT NULL
            RETURN DISTINCT a.uid AS source, b.uid AS target, type(r) AS rel_type
        """
        edge_rows, edge_cols = cypher_query(edge_cypher, {"workspace_id": workspace_id})
        edge_columns = edge_cols if isinstance(edge_cols, list) else []

        edges = []
        for row in edge_rows:
            edge_dict = dict(zip(edge_columns, row)) if edge_columns else {}
            source = edge_dict.get("source")
            target = edge_dict.get("target")
            if source in seen_uids and target in seen_uids:
                edges.append({
                    "source": source,
                    "target": target,
                    "type": edge_dict.get("rel_type", "RELATED"),
                })

        return Response({
            "query_type": "workspace_graph",
            "description": self.QUERIES["workspace_graph"]["description"],
            "nodes": nodes,
            "edges": edges,
            "counts": counts,
        })

    # ── Emotional graph category configuration ─────────────────────
    EMOTIONAL_CATEGORIES = {
        "Emotion": {
            "match": "(s:Signal {workspace_id: $ws})-[:EXPRESSES_EMOTION]->(n:Emotion {workspace_id: $ws})",
            "label_field": "n.name",
            "props": ["name", "valence", "description"],
        },
        "Person": {
            "match": "(n:Person {workspace_id: $ws})-[:PARTICIPANT_IN]->(s:Signal {workspace_id: $ws})",
            "label_field": "n.name",
            "props": ["name", "role", "role_type", "description"],
        },
        "ContextNode": {
            "match": "(s:Signal {workspace_id: $ws})-[:IN_CONTEXT]->(n:ContextNode {workspace_id: $ws})",
            "label_field": "n.name",
            "props": ["name", "description", "level"],
        },
        "ActionNode": {
            "match": "(s:Signal {workspace_id: $ws})-[:INVOLVES_ACTION]->(n:ActionNode {workspace_id: $ws})",
            "label_field": "n.name",
            "props": ["name", "description", "level"],
        },
        "TemporalNode": {
            "match": "(s:Signal {workspace_id: $ws})-[:AT_TIME]->(n:TemporalNode {workspace_id: $ws})",
            "label_field": "n.name",
            "props": ["name", "temporal_type", "description"],
        },
    }

    EMOTIONAL_LABEL_MAP = {
        "Emotion": "Emotion",
        "Person": "People",
        "ContextNode": "Context",
        "ActionNode": "Trigger",
        "TemporalNode": "Time",
    }

    EMOTIONAL_DETAIL_MATCHES = {
        "Emotion": """
            MATCH (n:Emotion {uid: $uid, workspace_id: $ws})<-[:EXPRESSES_EMOTION]-(s:Signal {workspace_id: $ws})
        """,
        "Person": """
            MATCH (n:Person {uid: $uid, workspace_id: $ws})-[:PARTICIPANT_IN]->(s:Signal {workspace_id: $ws})
        """,
        "ContextNode": """
            MATCH (n:ContextNode {uid: $uid, workspace_id: $ws})<-[:IN_CONTEXT]-(s:Signal {workspace_id: $ws})
        """,
        "ActionNode": """
            MATCH (n:ActionNode {uid: $uid, workspace_id: $ws})<-[:INVOLVES_ACTION]-(s:Signal {workspace_id: $ws})
        """,
        "TemporalNode": """
            MATCH (n:TemporalNode {uid: $uid, workspace_id: $ws})<-[:AT_TIME]-(s:Signal {workspace_id: $ws})
        """,
    }

    EMOTIONAL_RELATED_NODE_MATCHES = {
        "Emotion": """
            MATCH (s)-[:EXPRESSES_EMOTION]->(candidate:Emotion {workspace_id: $ws})
        """,
        "Person": """
            MATCH (candidate:Person {workspace_id: $ws})-[:PARTICIPANT_IN]->(s)
        """,
        "ContextNode": """
            MATCH (s)-[:IN_CONTEXT]->(candidate:ContextNode {workspace_id: $ws})
        """,
        "ActionNode": """
            MATCH (s)-[:INVOLVES_ACTION]->(candidate:ActionNode {workspace_id: $ws})
        """,
        "TemporalNode": """
            MATCH (s)-[:AT_TIME]->(candidate:TemporalNode {workspace_id: $ws})
        """,
    }

    def _handle_emotional_graph(self, workspace_id, params):
        """
        Return an emotional landscape graph — situation nodes on one side,
        emotion nodes on the other, with Signals as the invisible bridge.

        Accepts 'categories' param: list of node types to include.
        Default: ["Emotion"] (emotional fingerprint).
        Accepts optional 'conversation_id' to scope to a single conversation.
        Always returns counts for ALL categories regardless of selection.
        """
        requested = params.get("categories") or ["Emotion"]
        conversation_id = params.get("conversation_id")

        # Validate categories
        valid_categories = [c for c in requested if c in self.EMOTIONAL_CATEGORIES]
        if not valid_categories:
            valid_categories = ["Emotion"]

        # Build conversation scope clause if needed
        conv_scope_prefix = ""
        conv_params = {"ws": workspace_id}
        if conversation_id:
            conv_scope_prefix = (
                "MATCH (conv:Conversation {conversation_id: $conv_id, workspace_id: $ws})"
                "-[:CONTAINS_SIGNAL]->(s_conv:Signal) WITH collect(DISTINCT s_conv) AS scoped_signals "
            )
            conv_params["conv_id"] = conversation_id

        # ── 1. Collect counts for ALL categories ────────────────────
        counts = {}
        for cat_name, cat_info in self.EMOTIONAL_CATEGORIES.items():
            try:
                if conversation_id:
                    count_cypher = f"""
                        {conv_scope_prefix}
                        MATCH {cat_info['match']}
                        WHERE s IN scoped_signals AND {EMOTIONAL_NODE_FILTER}
                        RETURN count(DISTINCT n) AS cnt
                    """
                else:
                    count_cypher = f"""
                        MATCH {cat_info['match']}
                        WHERE {EMOTIONAL_NODE_FILTER}
                        RETURN count(DISTINCT n) AS cnt
                    """
                rows, _ = cypher_query(count_cypher, conv_params)
                counts[cat_name] = rows[0][0] if rows else 0
            except Exception:
                counts[cat_name] = 0

        # ── 2. Collect nodes for selected categories ────────────────
        nodes = []
        seen_uids = set()
        # Map uid → category for edge building
        uid_to_category = {}

        for cat_name in valid_categories:
            cat_info = self.EMOTIONAL_CATEGORIES[cat_name]
            if conversation_id:
                node_cypher = f"""
                    {conv_scope_prefix}
                    MATCH {cat_info['match']}
                    WHERE s IN scoped_signals AND {EMOTIONAL_NODE_FILTER}
                    WITH n,
                         count(DISTINCT s) AS signal_count,
                         collect(DISTINCT s.signal_address)[0..5] AS signal_addresses
                    RETURN n.uid AS uid,
                           properties(n) AS node_properties,
                           signal_count,
                           signal_addresses
                """
            else:
                node_cypher = f"""
                    MATCH {cat_info['match']}
                    WHERE {EMOTIONAL_NODE_FILTER}
                    WITH n,
                         count(DISTINCT s) AS signal_count,
                         collect(DISTINCT s.signal_address)[0..5] AS signal_addresses
                    RETURN n.uid AS uid,
                           properties(n) AS node_properties,
                           signal_count,
                           signal_addresses
                """
            try:
                rows, cols = cypher_query(node_cypher, conv_params)
                columns = cols if isinstance(cols, list) else []
                for row in rows:
                    d = dict(zip(columns, row)) if columns else {}
                    uid = d.get("uid")
                    if not uid or uid in seen_uids:
                        continue
                    node_properties = _normalize_graph_value(d.get("node_properties") or {})
                    label = (node_properties.get("name") or "").strip()
                    if not label or "*" in label:
                        continue
                    seen_uids.add(uid)
                    uid_to_category[uid] = cat_name

                    friendly = self.EMOTIONAL_LABEL_MAP.get(cat_name, cat_name)
                    properties = {
                        **{k: v for k, v in node_properties.items() if v is not None},
                        "signal_count": d.get("signal_count", 0) or 0,
                    }
                    if d.get("signal_addresses"):
                        properties["signal_addresses"] = _normalize_graph_value(d["signal_addresses"])

                    nodes.append({
                        "uid": uid,
                        "name": label,
                        "label": label,
                        "kind": friendly,
                        "raw_kind": cat_name,
                        "weight": d.get("signal_count", 0) or 0,
                        "properties": properties,
                    })
            except Exception as e:
                logger.warning(f"Emotional graph node query failed for {cat_name}: {e}")

        # ── 3. Build edges (signal-mediated) ──────────────────────────
        edges = []
        edge_set = set()  # (source, target) dedup

        def _add_edge_rows(rows, cols):
            columns = cols if isinstance(cols, list) else []
            for row in rows:
                d = dict(zip(columns, row)) if columns else {}
                src = d.get("source")
                tgt = d.get("target")
                w = d.get("weight", 1)
                if src and tgt and src != tgt and src in seen_uids and tgt in seen_uids and w > 0:
                    pair = (min(src, tgt), max(src, tgt))
                    if pair not in edge_set:
                        edge_set.add(pair)
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "weight": w,
                            "type": "CONNECTED_THROUGH_SIGNALS",
                            "relationship": "CONNECTED_THROUGH_SIGNALS",
                        })

        # 3a. Intra-category edges: nodes of the same type co-occurring
        #     on the same Signal are connected (weight = shared signal count).
        for cat_name in valid_categories:
            info = self.EMOTIONAL_CATEGORIES[cat_name]
            if cat_name == "Person":
                intra_cypher = f"""
                    MATCH (a:Person {{workspace_id: $ws}})-[:PARTICIPANT_IN]->(s:Signal {{workspace_id: $ws}})
                          <-[:PARTICIPANT_IN]-(b:Person {{workspace_id: $ws}})
                    WHERE a.uid IS NOT NULL AND b.uid IS NOT NULL AND a.uid < b.uid
                    RETURN a.uid AS source, b.uid AS target, count(DISTINCT s) AS weight
                """
            else:
                rel_pattern = info["match"]  # e.g. (s:Signal ...)-[:REL]->(n:Type)
                # Extract the relationship from the match pattern
                intra_cypher = f"""
                    MATCH {info['match']}
                    WITH s, n AS node_a
                    MATCH (s)-[]->(node_b:{cat_name} {{workspace_id: $ws}})
                    WHERE node_a.uid IS NOT NULL AND node_b.uid IS NOT NULL
                          AND node_a.uid < node_b.uid
                    RETURN node_a.uid AS source, node_b.uid AS target,
                           count(DISTINCT s) AS weight
                """
            # Scope intra-category edges to conversation if needed
            if conversation_id:
                if cat_name == "Person":
                    intra_cypher = f"""
                        {conv_scope_prefix}
                        MATCH (a:Person {workspace_id: $ws})-[:PARTICIPANT_IN]->(s:Signal {{workspace_id: $ws}})
                              <-[:PARTICIPANT_IN]-(b:Person {workspace_id: $ws})
                        WHERE s IN scoped_signals
                              AND a.uid IS NOT NULL AND b.uid IS NOT NULL AND a.uid < b.uid
                        RETURN a.uid AS source, b.uid AS target, count(DISTINCT s) AS weight
                    """
                else:
                    intra_cypher = f"""
                        {conv_scope_prefix}
                        MATCH {info['match']}
                        WHERE s IN scoped_signals
                        WITH s, n AS node_a
                        MATCH (s)-[]->(node_b:{cat_name} {{workspace_id: $ws}})
                        WHERE node_a.uid IS NOT NULL AND node_b.uid IS NOT NULL
                              AND node_a.uid < node_b.uid
                        RETURN node_a.uid AS source, node_b.uid AS target,
                               count(DISTINCT s) AS weight
                    """
            try:
                rows, cols = cypher_query(intra_cypher, conv_params)
                _add_edge_rows(rows, cols)
            except Exception as e:
                logger.warning(f"Emotional graph intra-edge query failed for {cat_name}: {e}")

        # 3b. Cross-category edges: nodes of different types connected
        #     through the same Signal.
        if len(valid_categories) >= 2:
            for i, cat_a in enumerate(valid_categories):
                for cat_b in valid_categories[i + 1:]:
                    info_a = self.EMOTIONAL_CATEGORIES[cat_a]
                    info_b = self.EMOTIONAL_CATEGORIES[cat_b]

                    # Build edge Cypher based on direction of the relationship
                    if cat_a == "Person" and cat_b == "Person":
                        continue  # handled by intra-category above
                    elif cat_a == "Person":
                        edge_cypher = f"""
                            MATCH (node_a:Person {{workspace_id: $ws}})-[:PARTICIPANT_IN]->(s:Signal {{workspace_id: $ws}})
                            WITH s, node_a
                            MATCH (s)-[]->(node_b:{cat_b} {{workspace_id: $ws}})
                            WHERE node_a.uid IS NOT NULL AND node_b.uid IS NOT NULL
                            RETURN node_a.uid AS source, node_b.uid AS target,
                                   count(DISTINCT s) AS weight
                        """
                    elif cat_b == "Person":
                        edge_cypher = f"""
                            MATCH {info_a['match']}
                            WITH s, n AS node_a
                            MATCH (node_b:Person {{workspace_id: $ws}})-[:PARTICIPANT_IN]->(s)
                            WHERE node_a.uid IS NOT NULL AND node_b.uid IS NOT NULL
                            RETURN node_a.uid AS source, node_b.uid AS target,
                                   count(DISTINCT s) AS weight
                        """
                    else:
                        edge_cypher = f"""
                            MATCH {info_a['match']}
                            WITH s, n AS node_a
                            MATCH (s)-[]->(node_b:{cat_b} {{workspace_id: $ws}})
                            WHERE node_a.uid IS NOT NULL AND node_b.uid IS NOT NULL
                            RETURN node_a.uid AS source, node_b.uid AS target,
                                   count(DISTINCT s) AS weight
                        """

                    try:
                        rows, cols = cypher_query(edge_cypher, conv_params)
                        _add_edge_rows(rows, cols)
                    except Exception as e:
                        logger.warning(f"Emotional graph edge query failed ({cat_a}↔{cat_b}): {e}")

        return Response({
            "query_type": "emotional_graph",
            "description": self.QUERIES["emotional_graph"]["description"],
            "nodes": nodes,
            "edges": edges,
            "counts": counts,
            "selected_categories": valid_categories,
        })

    def _handle_emotional_node_detail(self, workspace_id, params):
        uid = str(params.get("uid") or "").strip()
        raw_kind = str(params.get("raw_kind") or "").strip()

        if not uid:
            return Response({"detail": "uid is required for emotional_node_detail."}, status=status.HTTP_400_BAD_REQUEST)

        if raw_kind not in self.EMOTIONAL_DETAIL_MATCHES:
            return Response(
                {"detail": f"Unsupported emotional node type: {raw_kind}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        query_params = {"ws": workspace_id, "uid": uid}
        base_match = self.EMOTIONAL_DETAIL_MATCHES[raw_kind]

        node_rows, node_cols = cypher_query(
            f"""
                {base_match}
                WHERE {_emotional_node_filter("n")}
                WITH n, count(DISTINCT s) AS signal_count
                RETURN n.uid AS uid,
                       properties(n) AS node_properties,
                       signal_count
                LIMIT 1
            """,
            query_params,
        )
        node_columns = node_cols if isinstance(node_cols, list) else []
        node_dict = dict(zip(node_columns, node_rows[0])) if node_rows and node_columns else {}
        node_properties = _normalize_graph_value(node_dict.get("node_properties") or {})
        label = str(node_properties.get("name") or "").strip()

        if not label or "*" in label:
            return Response(
                {
                    "query_type": "emotional_node_detail",
                    "description": self.QUERIES["emotional_node_detail"]["description"],
                    "node": None,
                    "related_conversations": [],
                    "related_signals": [],
                    "related_nodes": {},
                    "counts": {
                        "connected_signals": 0,
                        "connected_conversations": 0,
                    },
                }
            )

        def _rows_to_dicts(rows, cols):
            columns = cols if isinstance(cols, list) else []
            return [_normalize_graph_value(dict(zip(columns, row))) for row in rows] if columns else []

        aggregate_rows, aggregate_cols = cypher_query(
            f"""
                {base_match}
                WHERE {_emotional_node_filter("n")}
                OPTIONAL MATCH (c:Conversation {{workspace_id: $ws}})-[:CONTAINS_SIGNAL]->(s)
                RETURN count(DISTINCT s) AS connected_signals,
                       count(DISTINCT c) AS connected_conversations
            """,
            query_params,
        )
        aggregate_columns = aggregate_cols if isinstance(aggregate_cols, list) else []
        aggregate = (
            dict(zip(aggregate_columns, aggregate_rows[0]))
            if aggregate_rows and aggregate_columns
            else {}
        )

        conversation_rows, conversation_cols = cypher_query(
            f"""
                {base_match}
                WHERE {_emotional_node_filter("n")}
                OPTIONAL MATCH (c:Conversation {{workspace_id: $ws}})-[:CONTAINS_SIGNAL]->(s)
                WITH c,
                     count(DISTINCT s) AS matched_signals,
                     max(coalesce(s.created_at, s.create_time, 0)) AS last_seen
                WHERE c IS NOT NULL
                RETURN c.conversation_id AS conversation_id,
                       coalesce(c.title, 'Untitled conversation') AS title,
                       coalesce(c.last_active, c.create_time, last_seen) AS last_active,
                       c.create_time AS create_time,
                       c.turn_count AS turn_count,
                       matched_signals
                ORDER BY last_active DESC, matched_signals DESC
                LIMIT 6
            """,
            query_params,
        )
        related_conversations = _rows_to_dicts(conversation_rows, conversation_cols)

        signal_rows, signal_cols = cypher_query(
            f"""
                {base_match}
                WHERE {_emotional_node_filter("n")}
                OPTIONAL MATCH (c:Conversation {{workspace_id: $ws}})-[:CONTAINS_SIGNAL]->(s)
                RETURN DISTINCT s.uid AS uid,
                       s.signal_address AS signal_address,
                       s.content_preview AS content_preview,
                       coalesce(s.created_at, s.create_time) AS created_at,
                       c.conversation_id AS conversation_id,
                       c.title AS conversation_title
                ORDER BY created_at DESC
                LIMIT 8
            """,
            query_params,
        )
        related_signals = _rows_to_dicts(signal_rows, signal_cols)

        related_nodes = {}
        for related_kind, related_match in self.EMOTIONAL_RELATED_NODE_MATCHES.items():
            if related_kind == raw_kind:
                continue
            related_rows, related_cols = cypher_query(
                f"""
                    {base_match}
                    WHERE {_emotional_node_filter("n")}
                    WITH DISTINCT s
                    {related_match}
                    WHERE candidate.uid <> $uid AND {_emotional_node_filter("candidate")}
                    WITH candidate, count(DISTINCT s) AS shared_signal_count
                    RETURN candidate.uid AS uid,
                           candidate.name AS label,
                           shared_signal_count
                    ORDER BY shared_signal_count DESC, label ASC
                    LIMIT 6
                """,
                query_params,
            )
            items = [
                row
                for row in _rows_to_dicts(related_rows, related_cols)
                if row.get("label") and "*" not in str(row.get("label"))
            ]
            if items:
                related_nodes[related_kind] = items

        node_payload = {
            "uid": uid,
            "label": label,
            "kind": self.EMOTIONAL_LABEL_MAP.get(raw_kind, raw_kind),
            "raw_kind": raw_kind,
            "properties": {
                **{k: v for k, v in node_properties.items() if v is not None},
                "signal_count": node_dict.get("signal_count", 0) or 0,
            },
        }

        return Response(
            {
                "query_type": "emotional_node_detail",
                "description": self.QUERIES["emotional_node_detail"]["description"],
                "node": node_payload,
                "related_conversations": related_conversations,
                "related_signals": related_signals,
                "related_nodes": related_nodes,
                "counts": {
                    "connected_signals": aggregate.get("connected_signals", node_dict.get("signal_count", 0) or 0),
                    "connected_conversations": aggregate.get("connected_conversations", 0) or 0,
                },
            }
        )

    def get(self, request):
        """GET returns available query types."""
        return Response({
            "available_queries": {
                k: v["description"] for k, v in self.QUERIES.items()
            }
        })


class GraphTestsView(WorkspaceAPIView):
    """GET /api/graph-tests/ returns card-ready graph test analytics for the current workspace."""

    def get(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace
        filters = {
            "time_range": request.query_params.get("time_range", "30d"),
            "contexts": _parse_csv_param(request.query_params.get("contexts")),
            "people": _parse_csv_param(request.query_params.get("people")),
            "compare_by": request.query_params.get("compare_by", "time_of_day"),
            "pattern_trend": request.query_params.get("pattern_trend")
            or request.query_params.get("pattern_valence", "helping_me_recover"),
            "min_support": max(1, int(request.query_params.get("min_support", "1") or "1")),
            "normalize": _parse_bool_param(request.query_params.get("normalize")),
        }

        payload = build_graph_tests_payload(workspace=workspace, filters=filters)
        return Response(payload)


# ═══════════════════════════════════════════════════════════════════════
# V5.0 — Live Conversation
# ═══════════════════════════════════════════════════════════════════════


class LiveConversationView(WorkspaceAPIView):
    def post(self, request):
        workspace = self.get_workspace()
        request.workspace = workspace

        message = request.data.get("message", "").strip()
        if not message:
            return Response(
                {"detail": "No message provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        service = LiveConversationService(workspace=workspace, user=request.user)
        payload = service.run(
            message=message,
            conversation_id=request.data.get("conversation_id"),
            template_id=request.data.get("template"),
            history=request.data.get("history", []),
            persona_id=request.data.get("persona"),
        )
        return Response(payload)
