"""Conversation state management for tracking messages and citations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "user" or "agent"
    text: str
    timestamp: float
    citations: List[Dict[str, Any]] = field(default_factory=list)
    audio_bytes: Optional[bytes] = None
    audio_format: Optional[str] = None


@dataclass
class ConversationState:
    """Manages state of a conversation session."""

    conversation_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    all_citations_used: List[Dict[str, Any]] = field(default_factory=list)
    followup_mode: str = "troubleshooting"  # "troubleshooting" or "wrap-up"
    video_context: Optional[Dict[str, Any]] = None
    started: bool = False
    ended: bool = False

    def add_message(
        self,
        role: str,
        text: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        audio_bytes: Optional[bytes] = None,
        audio_format: Optional[str] = None,
    ) -> None:
        """Add a message to the conversation."""
        import time

        message = Message(
            role=role,
            text=text,
            timestamp=time.time(),
            citations=citations or [],
            audio_bytes=audio_bytes,
            audio_format=audio_format,
        )
        self.messages.append(message)

        # Track citations
        if citations:
            for citation in citations:
                # Avoid duplicates
                if citation not in self.all_citations_used:
                    self.all_citations_used.append(citation)

        logger.debug(f"Added {role} message with {len(citations or [])} citations")

    def add_message_with_citations(
        self,
        role: str,
        text: str,
        citations: List[Dict[str, Any]],
        audio_bytes: Optional[bytes] = None,
        audio_format: Optional[str] = None,
    ) -> None:
        """Add a message with citations (convenience method)."""
        self.add_message(
            role=role,
            text=text,
            citations=citations,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
        )

    def track_citations_used(self, citations: List[Dict[str, Any]]) -> None:
        """Track citations used in the conversation."""
        for citation in citations:
            if citation not in self.all_citations_used:
                self.all_citations_used.append(citation)

    def switch_followup_mode(self, mode: str) -> None:
        """Switch follow-up mode (troubleshooting or wrap-up)."""
        if mode not in ["troubleshooting", "wrap-up"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'troubleshooting' or 'wrap-up'")
        self.followup_mode = mode
        logger.info(f"Switched to {mode} mode")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation with citations."""
        # Group citations by file
        citations_by_file: Dict[str, List[int]] = {}
        for citation in self.all_citations_used:
            file_name = citation.get("file", "Unknown")
            page = citation.get("page")
            if file_name not in citations_by_file:
                citations_by_file[file_name] = []
            if page is not None and page not in citations_by_file[file_name]:
                citations_by_file[file_name].append(page)

        # Format citations summary
        citations_summary = []
        for file_name, pages in citations_by_file.items():
            if pages:
                pages_str = ", ".join(map(str, sorted(pages)))
                citations_summary.append(f"{file_name} – Pages {pages_str}")
            else:
                citations_summary.append(file_name)

        return {
            "conversation_id": self.conversation_id,
            "total_messages": len(self.messages),
            "user_messages": len([m for m in self.messages if m.role == "user"]),
            "agent_messages": len([m for m in self.messages if m.role == "agent"]),
            "citations_used": self.all_citations_used,
            "citations_summary": citations_summary,
            "followup_mode": self.followup_mode,
            "video_context": self.video_context,
        }

    def get_messages_dict(self) -> List[Dict[str, Any]]:
        """Get messages as list of dictionaries."""
        result = []
        for msg in self.messages:
            msg_dict = {
                "role": msg.role,
                "text": msg.text,
                "timestamp": msg.timestamp,
                "citations": msg.citations,
            }
            if msg.audio_bytes:
                msg_dict["has_audio"] = True
                msg_dict["audio_format"] = msg.audio_format
            result.append(msg_dict)
        return result

    def set_video_context(self, video_analysis: Dict[str, Any]) -> None:
        """Set video analysis context for the conversation."""
        self.video_context = video_analysis
        logger.info("Set video context for conversation")
