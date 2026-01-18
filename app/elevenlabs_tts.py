"""ElevenLabs Text-to-Speech integration for converting text answers to audio."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)


class ElevenLabsTTSError(Exception):
    """Custom exception for ElevenLabs TTS errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def text_to_speech(
    text: str,
    api_key: str,
    language: str = "en",
    voice_id: Optional[str] = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "pcm_44100",
) -> bytes:
    """
    Convert text to speech using ElevenLabs API.
    
    Args:
        text: The text to convert to speech
        api_key: ElevenLabs API key
        language: Language code ("en" or "ar")
        voice_id: Optional voice ID. If not provided, uses default based on language
        model_id: ElevenLabs model ID (default: eleven_multilingual_v2)
        output_format: Output format. Use "pcm_44100" for WAV-compatible PCM
        
    Returns:
        Audio data as bytes (PCM format that can be converted to WAV)
        
    Raises:
        ElevenLabsTTSError: If the API call fails
        RuntimeError: If the elevenlabs package is not installed
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    if not api_key:
        raise ValueError("API key is required")
    
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency "elevenlabs". Install it with: pip install elevenlabs'
        ) from e
    
    # Default voice IDs based on language
    # These are common multilingual voices that support both English and Arabic
    default_voices = {
        "en": "JBFqnCBsd6RMkjVDRZzb",  # Default English voice
        "ar": "JBFqnCBsd6RMkjVDRZzb",  # Multilingual voice that supports Arabic
    }
    
    selected_voice_id = voice_id or default_voices.get(language, default_voices["en"])
    
    try:
        client = ElevenLabs(api_key=api_key)
        
        # Generate audio using ElevenLabs API
        # output_format "pcm_44100" gives us PCM data that we can convert to WAV
        audio_generator = client.text_to_speech.convert(
            voice_id=selected_voice_id,
            model_id=model_id,
            text=text,
            output_format=output_format,
        )
        
        # Collect audio bytes from generator
        audio_bytes = b""
        for chunk in audio_generator:
            if chunk:
                audio_bytes += chunk
        
        if not audio_bytes:
            raise ElevenLabsTTSError("No audio data received from ElevenLabs API")
        
        return audio_bytes
        
    except Exception as e:
        if isinstance(e, ElevenLabsTTSError):
            raise
        logger.error(f"ElevenLabs TTS error: {str(e)}")
        raise ElevenLabsTTSError(f"Failed to generate speech: {str(e)}") from e


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 44100, channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Convert PCM audio bytes to WAV format.
    
    Args:
        pcm_bytes: Raw PCM audio data
        sample_rate: Sample rate in Hz (default: 44100)
        channels: Number of audio channels (default: 1 for mono)
        sample_width: Sample width in bytes (default: 2 for 16-bit)
        
    Returns:
        WAV file as bytes
    """
    import struct
    
    # WAV file header
    # RIFF header
    riff_header = b"RIFF"
    file_size = 36 + len(pcm_bytes)
    riff_chunk_size = struct.pack("<I", file_size)
    wave_format = b"WAVE"
    
    # fmt chunk
    fmt_chunk_id = b"fmt "
    fmt_chunk_size = struct.pack("<I", 16)  # PCM format chunk size
    audio_format = struct.pack("<H", 1)  # PCM = 1
    num_channels = struct.pack("<H", channels)
    sample_rate_bytes = struct.pack("<I", sample_rate)
    byte_rate = struct.pack("<I", sample_rate * channels * sample_width)
    block_align = struct.pack("<H", channels * sample_width)
    bits_per_sample = struct.pack("<H", sample_width * 8)
    
    # data chunk
    data_chunk_id = b"data"
    data_chunk_size = struct.pack("<I", len(pcm_bytes))
    
    # Combine all parts
    wav_bytes = (
        riff_header +
        riff_chunk_size +
        wave_format +
        fmt_chunk_id +
        fmt_chunk_size +
        audio_format +
        num_channels +
        sample_rate_bytes +
        byte_rate +
        block_align +
        bits_per_sample +
        data_chunk_id +
        data_chunk_size +
        pcm_bytes
    )
    
    return wav_bytes


def text_to_speech_wav(
    text: str,
    api_key: str,
    language: str = "en",
    voice_id: Optional[str] = None,
    model_id: str = "eleven_multilingual_v2",
) -> tuple[bytes, str]:
    """
    Convert text to speech and return audio bytes with format.
    
    Uses MP3 format from ElevenLabs (free tier compatible) and converts to WAV if possible.
    
    Args:
        text: The text to convert to speech
        api_key: ElevenLabs API key
        language: Language code ("en" or "ar")
        voice_id: Optional voice ID
        model_id: ElevenLabs model ID
        
    Returns:
        Tuple of (audio_bytes, format_string) where format is "wav" or "mp3"
    """
    try:
        # Use MP3 format (available on free tier)
        mp3_audio = text_to_speech(
            text=text,
            api_key=api_key,
            language=language,
            voice_id=voice_id,
            model_id=model_id,
            output_format="mp3_44100_128",  # Free tier compatible format
        )
        
        # Try to convert MP3 to WAV using pydub if available
        # Note: pydub requires ffmpeg for MP3 support (system dependency)
        try:
            from pydub import AudioSegment
            
            # Load MP3 from bytes
            audio_segment = AudioSegment.from_mp3(BytesIO(mp3_audio))
            # Export as WAV
            wav_buffer = BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            return (wav_buffer.getvalue(), "wav")
        except (ImportError, Exception) as e:
            # If pydub is not available, ffmpeg is missing, or conversion fails, return MP3
            # MP3 format works fine in browsers and Streamlit
            logger.info(f"MP3 to WAV conversion not available ({str(e)}), returning MP3 format")
            return (mp3_audio, "mp3")
                
    except Exception as e:
        # If MP3 fails, try pcm_22050 as fallback (might work on some free tier accounts)
        logger.warning(f"MP3 format failed: {str(e)}, trying pcm_22050 format")
        try:
            pcm_audio = text_to_speech(
                text=text,
                api_key=api_key,
                language=language,
                voice_id=voice_id,
                model_id=model_id,
                output_format="pcm_22050",  # Lower sample rate, might work on free tier
            )
            wav_audio = pcm_to_wav(pcm_audio, sample_rate=22050, channels=1, sample_width=2)
            return (wav_audio, "wav")
        except Exception as e2:
            # If all formats fail, re-raise the original error
            raise ElevenLabsTTSError(f"Failed to generate audio with any supported format. MP3 error: {str(e)}, PCM error: {str(e2)}") from e2

