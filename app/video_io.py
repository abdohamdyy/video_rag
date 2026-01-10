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
        raise VideoTooLargeError(f"Video exceeds max_bytes={max_bytes}")

    mime = None
    if filename:
        mime, _ = mimetypes.guess_type(filename)
    if not mime or not mime.startswith("video/"):
        mime = "video/mp4"

    return VideoPayload(data=file_bytes, mime_type=mime, source="upload")


