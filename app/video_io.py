from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from typing import Optional

import httpx


class VideoTooLargeError(ValueError):
    pass


class VideoDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoPayload:
    data: bytes
    mime_type: str
    source: str  # "upload" | "url"


def _guess_mime_type_from_url(url: str) -> str:
    mime, _ = mimetypes.guess_type(url)
    if mime and mime.startswith("video/"):
        return mime
    return "video/mp4"


def _is_youtube_url(url: str) -> bool:
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)


def _looks_like_mp4(data: bytes) -> bool:
    # MP4 typically contains "ftyp" near the beginning (e.g., bytes 4..8).
    return len(data) >= 12 and (b"ftyp" in data[:64])


def _looks_like_webm(data: bytes) -> bool:
    # WebM is an EBML container, usually starting with 1A 45 DF A3.
    return len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3"


async def download_video(url: str, *, max_bytes: int, timeout_s: float = 30.0) -> VideoPayload:
    if not url.lower().startswith(("http://", "https://")):
        raise VideoDownloadError("video_url must start with http:// or https://")

    if _is_youtube_url(url):
        raise VideoDownloadError(
            "YouTube URLs are not direct video files. Please upload the video file instead, "
            "or provide a direct link to an .mp4/.webm file."
        )

    guessed_mime = _guess_mime_type_from_url(url)

    limits = httpx.Limits(max_connections=20, max_keepalive_connections=5)
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True, limits=limits) as client:
        try:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                header_ct = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                if header_ct.startswith("video/"):
                    content_type = header_ct
                elif header_ct in {"application/octet-stream", "binary/octet-stream"} and guessed_mime.startswith("video/"):
                    content_type = guessed_mime
                else:
                    raise VideoDownloadError(
                        f"URL did not return a video content-type (got '{header_ct or 'missing'}'). "
                        "Please provide a direct link to a video file (.mp4/.webm), or upload the file."
                    )

                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise VideoTooLargeError(f"Video exceeds max_bytes={max_bytes}")
        except VideoTooLargeError:
            raise
        except Exception as e:  # noqa: BLE001 (keep error message)
            raise VideoDownloadError(str(e)) from e

    data = bytes(buf)
    # Lightweight sanity checks to avoid sending HTML/garbage to Gemini.
    if content_type == "video/mp4" and not _looks_like_mp4(data):
        raise VideoDownloadError("Downloaded content does not look like a valid MP4 video.")
    if content_type in {"video/webm", "video/x-matroska"} and not _looks_like_webm(data):
        raise VideoDownloadError("Downloaded content does not look like a valid WebM/EBML video.")

    return VideoPayload(data=data, mime_type=content_type, source="url")


async def read_upload(file_bytes: bytes, filename: Optional[str], *, max_bytes: int) -> VideoPayload:
    if len(file_bytes) > max_bytes:
        raise VideoTooLargeError(f"File exceeds max_bytes={max_bytes}")

    mime = None
    if filename:
        mime, _ = mimetypes.guess_type(filename)
    
    # Support both video and image MIME types
    if not mime:
        # Try to guess from filename extension
        if filename:
            filename_lower = filename.lower()
            if filename_lower.endswith(('.jpg', '.jpeg')):
                mime = "image/jpeg"
            elif filename_lower.endswith('.png'):
                mime = "image/png"
            elif filename_lower.endswith('.webp'):
                mime = "image/webp"
            elif filename_lower.endswith('.gif'):
                mime = "image/gif"
            elif filename_lower.endswith(('.mp4', '.mov', '.avi')):
                mime = "video/mp4"
            elif filename_lower.endswith('.webm'):
                mime = "video/webm"
    
    # Default fallback
    if not mime:
        # Check if it looks like an image by checking magic bytes
        if len(file_bytes) >= 4:
            # JPEG: FF D8 FF
            if file_bytes[:3] == b'\xff\xd8\xff':
                mime = "image/jpeg"
            # PNG: 89 50 4E 47
            elif file_bytes[:4] == b'\x89PNG':
                mime = "image/png"
            # GIF: 47 49 46 38
            elif file_bytes[:4] == b'GIF8':
                mime = "image/gif"
            # WebP: RIFF...WEBP
            elif file_bytes[:4] == b'RIFF' and len(file_bytes) >= 12 and file_bytes[8:12] == b'WEBP':
                mime = "image/webp"
            # MP4: ftyp box
            elif b"ftyp" in file_bytes[:64]:
                mime = "video/mp4"
            # WebM: EBML header
            elif file_bytes[:4] == b"\x1a\x45\xdf\xa3":
                mime = "video/webm"
        
        # Final fallback
        if not mime:
            mime = "video/mp4"  # Default to video

    return VideoPayload(data=file_bytes, mime_type=mime, source="upload")


