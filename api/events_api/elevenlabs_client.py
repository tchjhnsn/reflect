"""
ThriveSight ElevenLabs Client — Speech-to-Text integration.

Provides two transcription modes:
1. Batch (Scribe v2): Multi-speaker diarization + entity detection for conversation capture
2. Realtime token generation: Single-use tokens for WebSocket-based live transcription

Mirrors the centralized pattern of llm_client.py — env-based config, lazy init, error handling.
"""

import json
import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Configuration
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_STT_MODEL = "scribe_v2"

# File size limit (ElevenLabs allows up to 2GB, but we cap for hackathon sanity)
MAX_FILE_SIZE_MB = 100


class ElevenLabsClient:
    """
    Centralized ElevenLabs API client for speech-to-text operations.

    Usage:
        client = ElevenLabsClient()
        result = client.transcribe_batch(audio_bytes, "recording.wav", diarize=True)
        token = client.get_realtime_token()
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ELEVENLABS_API_KEY
        if not self.api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY not set. Add it to your .env file."
            )
        self._headers = {"xi-api-key": self.api_key}

    def transcribe_batch(
        self,
        file_bytes: bytes,
        filename: str = "audio.wav",
        diarize: bool = True,
        num_speakers: Optional[int] = None,
        entity_detection: Optional[str] = "all",
        language_code: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio using Scribe v2 batch API with optional diarization.

        Args:
            file_bytes: Raw audio file bytes
            filename: Original filename (used for MIME type inference)
            diarize: Enable speaker diarization (default True for conversations)
            num_speakers: Expected number of speakers (None = auto-detect)
            entity_detection: Entity detection mode ('all', 'pii', 'phi', etc.)
            language_code: ISO language code (None = auto-detect)

        Returns:
            Dict with keys:
            - text: str (full transcript)
            - language_code: str
            - segments: list[dict] (with speaker, text, start, end)
            - entities: list[dict] (detected entities with type, text, positions)
            - words: list[dict] (word-level timestamps)

        Raises:
            ValueError: If API key missing or file too large
            requests.HTTPError: If API call fails
        """
        # Validate file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Audio file too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB"
            )

        logger.info(
            f"Transcribing {filename} ({size_mb:.1f}MB) with Scribe v2 "
            f"(diarize={diarize}, num_speakers={num_speakers}, entity_detection={entity_detection})"
        )

        url = f"{ELEVENLABS_BASE_URL}/speech-to-text"

        # Build multipart form data
        files = {"file": (filename, file_bytes)}
        data = {"model_id": DEFAULT_STT_MODEL}

        if diarize:
            data["diarize"] = "true"
        if num_speakers is not None:
            data["num_speakers"] = str(num_speakers)
        if entity_detection:
            data["entity_detection"] = entity_detection
        if language_code:
            data["language_code"] = language_code

        response = requests.post(
            url,
            headers=self._headers,
            files=files,
            data=data,
            timeout=120,  # Transcription can take a while for long audio
        )

        if response.status_code != 200:
            logger.error(
                f"ElevenLabs transcription failed: {response.status_code} {response.text[:500]}"
            )
            response.raise_for_status()

        result = response.json()

        logger.info(
            f"Transcription complete: {len(result.get('text', ''))} chars, "
            f"{len(result.get('segments', []))} segments, "
            f"{len(result.get('entities', []))} entities"
        )

        return result

    def get_realtime_token(self, ttl_seconds: int = 900) -> str:
        """
        Generate a single-use token for WebSocket-based realtime transcription.

        The token allows frontend clients to connect directly to ElevenLabs
        without exposing the API key.

        Args:
            ttl_seconds: Token time-to-live in seconds (max 900 = 15 min)

        Returns:
            Single-use token string

        Raises:
            ValueError: If token generation fails
        """
        url = f"{ELEVENLABS_BASE_URL}/tokens/create"

        response = requests.post(
            url,
            headers={**self._headers, "Content-Type": "application/json"},
            json={"ttl_seconds": min(ttl_seconds, 900)},
            timeout=10,
        )

        if response.status_code != 200:
            logger.error(
                f"Token generation failed: {response.status_code} {response.text[:500]}"
            )
            response.raise_for_status()

        result = response.json()
        token = result.get("token")

        if not token:
            raise ValueError(f"No token in response: {result}")

        logger.info("Generated single-use realtime STT token")
        return token
