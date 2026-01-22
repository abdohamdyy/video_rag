"""
Streamlit UI for Video Analysis and Repair Assistance
Analyzes videos to identify appliance defects and provides repair instructions with source citations.
"""

import asyncio
import json
import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

from app.gemini_video_understanding import GeminiAPIError, analyze_video_with_gemini
from app.question_generation import (
    build_retrieval_queries,
    extract_user_questions,
    generate_clarifying_questions,
)
from app.rag_orchestrator import (
    answer_with_gemini,
    compose_grounded_prompt,
    retrieve_support_docs,
)
from app.settings import get_settings
from app.video_io import VideoDownloadError, VideoTooLargeError, download_video
from app.elevenlabs_tts import ElevenLabsTTSError, text_to_speech_wav
from app.conversation_manager import ConversationState
from app.elevenlabs_agent import ElevenLabsAgentError

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Video Repair Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .chat-message.agent {
        background-color: #f1f8e9;
        margin-right: 20%;
    }
    .citation-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        background-color: #fff3cd;
        font-size: 0.85rem;
        border: 1px solid #ffc107;
    }
    .knowledge-base-info {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_async(coro):
    """Run async function in Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def process_video_assistance(media_bytes: bytes, mime_type: str, language: str, user_hint: Optional[str] = None, part_number: Optional[str] = None, is_image: bool = False):
    """Process video/image through the full assistance pipeline."""
    settings = get_settings()
    
    if not settings.gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please set it in your .env file or environment variables.")
    
    # Step 1: Media analysis
    media_type = "image" if is_image else "video"
    with st.status(f"🎥 Analyzing {media_type} with AI...", expanded=True) as status:
        status.update(label=f"🎥 Analyzing {media_type} with Gemini AI...", state="running")
        result = analyze_video_with_gemini(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            video_bytes=media_bytes,
            video_mime_type=mime_type,
            language=language,
            user_hint=user_hint,
            is_image=is_image,
        )
        analysis = result.data or {}
        transcript = analysis.get("transcript", "") or ""
        status.update(label="✅ Video analysis complete", state="complete")
    
    # Step 2: Question generation
    with st.status("❓ Generating questions...", expanded=False) as status:
        status.update(label="❓ Extracting questions from transcript...", state="running")
        user_questions = extract_user_questions(transcript)
        clarifying_questions = generate_clarifying_questions(
            appliance_type=analysis.get("appliance_type"),
            brand_or_model=analysis.get("brand_or_model"),
            issue_summary=analysis.get("issue_summary"),
            language=language,
        )
        status.update(label="✅ Questions generated", state="complete")
    
    # Step 3: Retrieval
    with st.status("📚 Retrieving relevant documentation...", expanded=False) as status:
        status.update(label="📚 Searching knowledge base...", state="running")
        queries = build_retrieval_queries(
            appliance_type=analysis.get("appliance_type"),
            brand_or_model=analysis.get("brand_or_model"),
            issue_summary=analysis.get("issue_summary"),
            user_questions=user_questions,
        )
        # Use part_number from user input if provided, otherwise from video analysis
        effective_part_number = part_number if part_number and part_number.strip() else analysis.get("part_number")
        citations = retrieve_support_docs(queries, part_number=effective_part_number)
        status.update(label=f"✅ Found {len(citations)} relevant sources", state="complete")
    
    # Step 4: Generate grounded answer
    with st.status("💡 Generating repair instructions...", expanded=False) as status:
        status.update(label="💡 Generating detailed repair instructions with citations...", state="running")
        grounded_prompt = compose_grounded_prompt(
            transcript=transcript,
            analysis=analysis,
            clarifying_questions=clarifying_questions,
            citations=citations,
            language=language,
            part_number=effective_part_number,
        )
        answer = answer_with_gemini(
            api_key=settings.gemini_api_key,
            model=settings.gemini_answer_model,
            prompt=grounded_prompt,
        )
        status.update(label="✅ Repair instructions ready", state="complete")
    
    # Step 5: Generate audio from answer text
    audio_bytes = None
    audio_format = None
    if settings.elevenlabs_api_key:
        with st.status("🔊 Generating audio...", expanded=False) as status:
            status.update(label="🔊 Converting text to speech...", state="running")
            try:
                answer_text = answer.get("text", "")
                if answer_text:
                    audio_bytes, audio_format = text_to_speech_wav(
                        text=answer_text,
                        api_key=settings.elevenlabs_api_key,
                        language=language,
                    )
                    status.update(label="✅ Audio generated", state="complete")
                else:
                    status.update(label="⚠️ No text to convert", state="complete")
            except ElevenLabsTTSError as e:
                status.update(label=f"⚠️ Audio generation failed: {str(e)}", state="error")
                logger.warning(f"ElevenLabs TTS error: {str(e)}")
            except Exception as e:
                status.update(label=f"⚠️ Audio generation error: {str(e)}", state="error")
                logger.warning(f"Unexpected TTS error: {str(e)}")
    
    # Prepare follow-up questions
    follow_ups = analysis.get("questions_to_confirm") or clarifying_questions
    
    return {
        "analysis": analysis,
        "transcript": transcript,
        "user_questions": user_questions,
        "clarifying_questions": clarifying_questions,
        "citations": citations,
        "answer": answer,
        "follow_up_questions": follow_ups,
        "audio_bytes": audio_bytes,
        "audio_format": audio_format,
    }


def display_results(results: dict):
    """Display analysis results in a well-formatted UI."""
    analysis = results["analysis"]
    answer_text = results["answer"].get("text", "")
    citations = results["citations"]
    audio_bytes = results.get("audio_bytes")
    
    # Device Information Section
    st.header("📱 Device Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        appliance_type = analysis.get("appliance_type") or "Unknown"
        st.metric("Appliance Type", appliance_type)
    
    with col2:
        brand_model = analysis.get("brand_or_model") or "Not identified"
        st.metric("Brand/Model", brand_model)
    
    with col3:
        part_num = analysis.get("part_number")
        part_source = analysis.get("part_number_source", "unknown")
        if part_num:
            # Show Part Number with source indicator and visual styling
            if part_source == "extracted":
                st.metric(
                    "Part Number", 
                    part_num, 
                    help="✅ Extracted directly from the image/video by analyzing visible text, labels, or nameplates"
                )
                st.caption("✅ Extracted from video/image")
            elif part_source == "predicted":
                st.metric(
                    "Part Number", 
                    part_num, 
                    help="🔮 Predicted based on appliance type. The system guessed this Part Number because it couldn't find one in the video, but this is the only Part Number available in the database for this appliance type."
                )
                st.caption("🔮 Predicted (not visible in video)")
            else:
                st.metric("Part Number", part_num)
                st.caption("Part Number identified")
        else:
            st.metric("Part Number", "Not found", delta=None)
            st.caption("⚠️ Could not extract or predict Part Number")
            st.info("💡 **Tip:** Make sure the video/image shows nameplates, labels, or documentation with part numbers clearly visible.")
    
    # Problem Analysis Section
    st.header("🔍 Problem Analysis")
    
    issue_summary = analysis.get("issue_summary") or "No issue summary available."
    st.info(f"**Issue Summary:** {issue_summary}")
    
    # Root Causes
    root_causes = analysis.get("likely_root_causes") or []
    if root_causes:
        st.subheader("Likely Root Causes")
        for i, cause in enumerate(root_causes, 1):
            st.write(f"{i}. {cause}")
    
    # Safety Warnings
    safety_warnings = analysis.get("safety_warnings") or []
    if safety_warnings:
        st.header("⚠️ Safety Warnings")
        for warning in safety_warnings:
            st.warning(warning)
    
    # Repair Instructions Section
    st.header("🔧 Repair Instructions")
    
    # Recommended fix steps from analysis
    fix_steps = analysis.get("recommended_fix_steps") or []
    if fix_steps:
        st.subheader("Recommended Steps")
        for i, step in enumerate(fix_steps, 1):
            st.write(f"**Step {i}:** {step}")
    
    # RAG-generated answer (main answer)
    if answer_text:
        st.subheader("Detailed Repair Guide")
        st.markdown(answer_text)
    
    # Tools and Parts Needed
    tools_parts = analysis.get("tools_or_parts_needed") or []
    if tools_parts:
        st.subheader("Required Tools & Parts")
        st.write(", ".join(tools_parts))
    
    # Sources and Citations
    if citations:
        st.header("📖 Sources & References")
        st.write(f"Found {len(citations)} relevant document sections:")
        
        with st.expander("View All Citations", expanded=False):
            for i, citation in enumerate(citations, 1):
                with st.container():
                    file_name = citation.get("file") or "Unknown"
                    page_num = citation.get("page")
                    snippet = citation.get("snippet", "")
                    distance = citation.get("distance")
                    
                    st.markdown(f"**Citation {i}**")
                    st.markdown(f"- **Source:** {file_name}")
                    if page_num is not None:
                        st.markdown(f"- **Page:** {page_num}")
                    if distance is not None:
                        st.markdown(f"- **Relevance Score:** {distance:.3f}")
                    if snippet:
                        st.markdown(f"- **Excerpt:** {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
                    st.divider()
    
    # Follow-up Questions
    follow_ups = results.get("follow_up_questions") or []
    if follow_ups:
        st.header("❓ Follow-up Questions")
        st.write("To help diagnose the issue more accurately, consider answering these:")
        for i, question in enumerate(follow_ups[:5], 1):  # Show top 5
            st.write(f"{i}. {question}")
    
    # Transcript (expandable)
    transcript = results.get("transcript", "")
    if transcript:
        with st.expander("View Video Transcript", expanded=False):
            st.write(transcript)


def main():
    """Main Streamlit application."""
    st.markdown('<p class="main-header">🔧 Video Repair Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a video or provide a URL to get AI-powered repair instructions</p>',
        unsafe_allow_html=True,
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selector
        language = st.selectbox(
            "Response Language",
            options=["en", "ar"],
            index=0,
            help="Select the language for the AI's response",
        )
        
        # User hint input
        user_hint = st.text_area(
            "Additional Context (Optional)",
            height=100,
            help="Provide any additional context about the issue, symptoms, or questions you have",
            placeholder="E.g., 'The machine stopped working after cleaning' or 'Customer reports unusual noise'",
        )
        
        # Part number input
        part_number = st.text_input(
            "Part Number (Optional)",
            help="Enter the part number to filter results (e.g., CHS199100RECiN). If not provided, the system will try to extract it from the video.",
            placeholder="E.g., CHS199100RECiN",
        )
        
        st.divider()
        st.caption("💡 Tip: Provide context to get more accurate results")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Video", 
        "🖼️ Upload Image", 
        "🔗 Video URL", 
        "📚 Knowledge Base"
    ])
    
    media_bytes = None
    mime_type = None
    is_image = False
    
    with tab1:
        st.subheader("Upload a Video File")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "webm", "mov", "avi"],
            help="Supported formats: MP4, WebM, MOV, AVI. Maximum size: 25MB",
            key="video_uploader",
        )
        
        if uploaded_file is not None:
            # Read file bytes once
            media_bytes = uploaded_file.read()
            mime_type = uploaded_file.type or "video/mp4"
            is_image = False
            # Display video from bytes
            st.video(media_bytes)
    
    with tab2:
        st.subheader("Upload an Image File")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            help="Supported formats: JPG, JPEG, PNG, WebP, GIF. Maximum size: 20MB",
            key="image_uploader",
        )
        
        if uploaded_image is not None:
            # Read file bytes once
            media_bytes = uploaded_image.read()
            # Determine MIME type
            if uploaded_image.type:
                mime_type = uploaded_image.type
            elif uploaded_image.name.lower().endswith(('.png',)):
                mime_type = "image/png"
            elif uploaded_image.name.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            elif uploaded_image.name.lower().endswith(('.webp',)):
                mime_type = "image/webp"
            elif uploaded_image.name.lower().endswith(('.gif',)):
                mime_type = "image/gif"
            else:
                mime_type = "image/jpeg"  # Default
            is_image = True
            # Display image
            st.image(media_bytes, use_container_width=True)
    
    with tab3:
        st.subheader("Provide Video URL")
        video_url = st.text_input(
            "Enter video URL",
            placeholder="https://example.com/video.mp4",
            help="Direct link to a video file (not YouTube or streaming services)",
        )
        
        if video_url:
            try:
                settings = get_settings()
                with st.spinner("Downloading video..."):
                    payload = run_async(
                        download_video(video_url, max_bytes=settings.max_video_bytes)
                    )
                    media_bytes = payload.data
                    mime_type = payload.mime_type
                    is_image = False
                    st.success(f"Video downloaded successfully ({len(media_bytes) / 1024 / 1024:.2f} MB)")
                    st.video(media_bytes)
            except VideoTooLargeError as e:
                st.error(f"Video too large: {str(e)}")
                media_bytes = None
            except VideoDownloadError as e:
                st.error(f"Failed to download video: {str(e)}")
                media_bytes = None
            except Exception as e:
                st.error(f"Error downloading video: {str(e)}")
                media_bytes = None
    
    # Analyze button
    if media_bytes:
        st.divider()
        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
            try:
                settings = get_settings()
                
                # Validate media size
                max_size = settings.max_video_bytes
                if len(media_bytes) > max_size:
                    st.error(
                        f"File is too large. Maximum size: {max_size / 1024 / 1024:.0f} MB. "
                        f"Your file: {len(media_bytes) / 1024 / 1024:.2f} MB"
                    )
                else:
                    # Process through full pipeline
                    if mime_type is None:
                        mime_type = "video/mp4" if not is_image else "image/jpeg"  # Default fallback
                    start_time = time.time()
                    results = process_video_assistance(
                        media_bytes,
                        mime_type,
                        language,
                        user_hint if user_hint.strip() else None,
                        part_number if part_number.strip() else None,
                        is_image=is_image,
                    )
                    elapsed_time = time.time() - start_time
                    
                    # Display results
                    st.success(f"✅ Analysis complete in {elapsed_time:.1f} seconds!")
                    
                    # Store analysis data for voice conversation
                    st.session_state.video_analysis_data = results["analysis"]
                    analysis = results["analysis"]
                    
                    display_results(results)
                    
                    # Update Agent System Instructions and display Widget
                    st.markdown("---")
                    st.subheader("🎤 Voice Conversation with AI Agent")
                    
                    if settings.elevenlabs_agent_id:
                        # Get write API key (required for convai_write permission)
                        api_key = settings.get_elevenlabs_api_key_for_write()
                        if not api_key:
                            st.warning("⚠️ ELEVENLABS_API_KEY or ELEVENLABS_API_KEY_WRITE not configured. Please set it in .env to enable agent updates.")
                            logger.warning("ELEVENLABS_API_KEY not configured for write operations")
                        else:
                            # Update Agent System Instructions with video context
                            with st.spinner("🔄 Updating Agent with video analysis context..."):
                                try:
                                    from app.elevenlabs_agent import update_agent_system_instructions
                                    
                                    logger.info("Starting agent system instructions update")
                                    logger.info(f"Agent ID: {settings.elevenlabs_agent_id}")
                                    logger.info(f"Using API key for write operations")
                                    logger.info(f"Video analysis - Appliance: {analysis.get('appliance_type')}, Part: {analysis.get('part_number')}")
                                    
                                    update_result = update_agent_system_instructions(
                                        agent_id=settings.elevenlabs_agent_id,
                                        api_key=api_key,
                                        video_analysis=analysis,
                                        language=language,
                                    )
                                    
                                    if update_result.get("success"):
                                        st.success("✅ Agent updated with video analysis context!")
                                        logger.info(f"Agent update successful: {update_result}")
                                        
                                        # Display update info
                                        with st.expander("📋 Agent Update Details", expanded=False):
                                            st.json({
                                                "agent_id": update_result.get("agent_id"),
                                                "agent_name": update_result.get("agent_name"),
                                                "system_instructions_length": update_result.get("system_instructions_length"),
                                                "video_context": update_result.get("video_context"),
                                            })
                                    else:
                                        st.error("❌ Failed to update Agent. Please check logs.")
                                        logger.error(f"Agent update failed: {update_result}")
                                        
                                except ElevenLabsAgentError as e:
                                    st.error(f"❌ Failed to update Agent: {str(e)}")
                                    logger.error(f"ElevenLabs Agent Error: {str(e)}")
                                    st.warning("⚠️ Widget will still work, but Agent may not have the latest video context.")
                                    
                                except Exception as e:
                                    st.error(f"❌ Unexpected error updating Agent: {str(e)}")
                                    logger.exception(f"Unexpected error updating agent: {str(e)}")
                                    st.warning("⚠️ Widget will still work, but Agent may not have the latest video context.")
                            
                            # Voice conversation using SDK (instead of widget)
                            st.markdown("**🎤 Voice Conversation with AI Agent:**")
                            
                            # Initialize conversation state if not exists
                            if "conversation_id" not in st.session_state:
                                st.session_state.conversation_id = None
                            if "conversation_messages" not in st.session_state:
                                st.session_state.conversation_messages = []
                            if "is_recording" not in st.session_state:
                                st.session_state.is_recording = False
                            if "audio_bytes" not in st.session_state:
                                st.session_state.audio_bytes = None
                            if "last_processed_audio_hash" not in st.session_state:
                                st.session_state.last_processed_audio_hash = None
                            
                            # Try to import audio recorder
                            try:
                                from audio_recorder_streamlit import audio_recorder
                                
                                # Audio recorder with microphone icon
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.markdown("### 🎙️ Press to Record")
                                    audio_bytes_recorded = audio_recorder(
                                        text="",
                                        recording_color="#e74c3c",
                                        neutral_color="#34495e",
                                        icon_name="microphone",
                                        icon_size="3x",
                                    )
                                    
                                    if audio_bytes_recorded:
                                        import hashlib
                                        
                                        # Calculate hash of recorded audio
                                        audio_hash = hashlib.md5(audio_bytes_recorded).hexdigest()
                                        
                                        # Only process if this is new audio (not already processed)
                                        if audio_hash != st.session_state.last_processed_audio_hash:
                                            st.session_state.audio_bytes = audio_bytes_recorded
                                            st.session_state.is_recording = False
                                            
                                            # Process audio
                                            with st.spinner("🔄 Sending audio to agent..."):
                                                try:
                                                    from app.elevenlabs_agent import send_audio_to_agent
                                                    
                                                    response = send_audio_to_agent(
                                                        agent_id=settings.elevenlabs_agent_id,
                                                        api_key=api_key,
                                                        audio_bytes=audio_bytes_recorded,
                                                        conversation_id=st.session_state.conversation_id,
                                                        audio_format="audio/wav",
                                                    )
                                                    
                                                    if response.get("success"):
                                                        # Update conversation ID
                                                        if response.get("conversation_id"):
                                                            st.session_state.conversation_id = response.get("conversation_id")
                                                        
                                                        # Get response text and audio
                                                        response_text = response.get("response_text", "")
                                                        response_audio = response.get("response_audio")
                                                        citations = response.get("citations", [])
                                                        
                                                        # Add user message (transcript from audio)
                                                        user_transcript = "🎤 [Audio message]"  # Could be extracted from audio if available
                                                        st.session_state.conversation_messages.append({
                                                            "role": "user",
                                                            "text": user_transcript
                                                        })
                                                        
                                                        # Add agent response
                                                        st.session_state.conversation_messages.append({
                                                            "role": "assistant",
                                                            "text": response_text,
                                                            "citations": citations,
                                                            "audio": response_audio
                                                        })
                                                        
                                                        # Play agent audio response if available
                                                        if response_audio:
                                                            st.audio(response_audio, format="audio/wav")
                                                        
                                                        # Clear audio bytes
                                                        st.session_state.audio_bytes = None
                                                        
                                                        # Mark this audio as processed
                                                        st.session_state.last_processed_audio_hash = audio_hash
                                                        
                                                        # Rerun to show new messages (only after successful processing)
                                                        st.rerun()
                                                    else:
                                                        st.error("❌ Failed to get response from agent")
                                                        
                                                except Exception as e:
                                                    st.error(f"❌ Error: {str(e)}")
                                                    logger.exception(f"Error sending audio: {str(e)}")
                                
                            except ImportError:
                                st.warning("⚠️ `audio-recorder-streamlit` not installed. Installing...")
                                st.code("pip install audio-recorder-streamlit", language="bash")
                                st.info("Please install the package and restart the app.")
                            
                            # Display conversation history
                            if st.session_state.conversation_messages:
                                st.markdown("---")
                                st.markdown("### 📜 Conversation History")
                                for i, msg in enumerate(st.session_state.conversation_messages):
                                    if msg["role"] == "user":
                                        with st.chat_message("user"):
                                            st.write(msg["text"])
                                    else:
                                        with st.chat_message("assistant"):
                                            st.write(msg["text"])
                                            
                                            # Play audio if available
                                            if msg.get("audio"):
                                                st.audio(msg["audio"], format="audio/wav")
                                            
                                            # Show citations
                                            if msg.get("citations"):
                                                with st.expander("📖 Citations", expanded=False):
                                                    for citation in msg["citations"]:
                                                        st.write(f"- {citation.get('file', 'Unknown')} (Page {citation.get('page', 'N/A')})")
                            
                            # Clear conversation button
                            col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])
                            with col_clear2:
                                if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
                                    st.session_state.conversation_id = None
                                    st.session_state.conversation_messages = []
                                    st.session_state.audio_bytes = None
                                    st.rerun()
                            
                            # Additional info
                            st.markdown("---")
                            if settings.elevenlabs_knowledge_base_id:
                                st.info(f"💡 **Agent is configured with Knowledge Base:** `{settings.elevenlabs_knowledge_base_id}`")
                            st.info("💡 **Tip:** The Agent now has access to the video analysis context and will use the Knowledge Base to provide accurate answers with citations.")
                            st.info("💡 **Note:** This conversation uses SDK directly, ensuring full video context is available to the agent.")
                    else:
                        st.warning("⚠️ ELEVENLABS_AGENT_ID not configured. Please set it in .env to enable voice conversation.")
                        logger.warning("ELEVENLABS_AGENT_ID not configured")
                    
            except ValueError as e:
                st.error(f"Configuration error: {str(e)}")
                st.info("Please check your .env file or environment variables for GEMINI_API_KEY")
            except VideoDownloadError as e:
                st.error(f"Video processing error: {str(e)}")
            except GeminiAPIError as e:
                st.error(f"AI API error: {str(e)}")
                if e.retry_after_seconds:
                    st.info(f"Please retry after {e.retry_after_seconds} seconds")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Please upload a video/image file or provide a video URL to begin analysis")
    
    # Knowledge Base Management Tab
    with tab4:
        st.header("📚 Knowledge Base Management")
        st.markdown("**Manage documents in ElevenLabs Knowledge Base with part numbers.**")
        
        settings = get_settings()
        api_url = "http://localhost:8000"  # Default FastAPI URL
        
        if not settings.elevenlabs_api_key:
            st.error("⚠️ Missing ELEVENLABS_API_KEY. Please set it in your .env file.")
        elif not settings.elevenlabs_knowledge_base_id:
            st.warning("⚠️ Knowledge Base ID not set. Please create a Knowledge Base first.")
            
            # Create KB section
            st.subheader("Create Knowledge Base")
            kb_name = st.text_input("Knowledge Base Name", value="Technical Support Knowledge Base")
            if st.button("Create Knowledge Base"):
                with st.spinner("Creating Knowledge Base..."):
                    try:
                        from app.elevenlabs_knowledge_base import create_knowledge_base
                        result = create_knowledge_base(
                            api_key=settings.elevenlabs_api_key,
                            name=kb_name,
                        )
                        st.success(f"✅ Knowledge Base created! ID: {result['id']}")
                        st.info(f"Please add this to your .env file: ELEVENLABS_KNOWLEDGE_BASE_ID={result['id']}")
                    except Exception as e:
                        st.error(f"Error creating Knowledge Base: {str(e)}")
        else:
            st.success(f"✅ Knowledge Base ID: {settings.elevenlabs_knowledge_base_id}")
            
            # Agent Assignment Section
            st.subheader("🤖 Agent Assignment")
            col_agent1, col_agent2 = st.columns([2, 1])
            with col_agent1:
                if settings.elevenlabs_agent_id:
                    st.success(f"✅ Agent ID: {settings.elevenlabs_agent_id}")
                else:
                    st.warning("⚠️ Agent ID not set. Please set ELEVENLABS_AGENT_ID in .env")
            with col_agent2:
                if st.button("🔗 Assign KB to Agent", type="primary", use_container_width=True):
                    if not settings.elevenlabs_agent_id:
                        st.error("⚠️ Please set ELEVENLABS_AGENT_ID in .env first")
                    else:
                        with st.spinner("Assigning Knowledge Base to Agent..."):
                            try:
                                response = httpx.post(
                                    f"{api_url}/knowledge-base/assign-to-agent",
                                    timeout=30.0,
                                )
                                if response.status_code == 200:
                                    result = response.json()
                                    assignment = result.get("assignment", {})
                                    status = assignment.get("status", "unknown")
                                    
                                    if status == "success":
                                        st.success("✅ Knowledge Base assigned to Agent successfully!")
                                    elif status == "configured":
                                        st.info("ℹ️ Agent configuration checked. Knowledge Base should be set in Agent settings.")
                                        if assignment.get("note"):
                                            st.markdown(assignment.get("note"))
                                    elif status == "manual_required":
                                        st.warning("⚠️ Manual assignment required via Dashboard")
                                        if assignment.get("note"):
                                            st.markdown(assignment.get("note"))
                                        if assignment.get("dashboard_url"):
                                            st.markdown(f"🔗 **Dashboard Link:** [{assignment.get('dashboard_url')}]({assignment.get('dashboard_url')})")
                                    else:
                                        st.warning(f"⚠️ Assignment status: {status}")
                                        if assignment.get("note"):
                                            st.markdown(assignment.get("note"))
                                else:
                                    st.error(f"Failed to assign Knowledge Base: {response.text}")
                            except Exception as e:
                                st.error(f"Error assigning Knowledge Base: {str(e)}")
            
            # List Documents Section
            st.subheader("📄 Documents in Knowledge Base")
            
            # Folder filter
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                folder_name = st.text_input(
                    "📁 Folder Name (optional)",
                    value="",
                    help="Enter folder name (e.g., 'sm') to list documents from that folder only",
                    placeholder="e.g., sm"
                )
            with col_filter2:
                parent_folder_id = st.text_input(
                    "📁 Folder ID (optional)",
                    value="",
                    help="Or enter folder ID directly",
                    placeholder="folder_id_here"
                )
            
            if st.button("🔄 Refresh Documents List", type="secondary"):
                with st.spinner("Loading documents..."):
                    try:
                        params = {}
                        if folder_name:
                            params["folder_name"] = folder_name
                        if parent_folder_id:
                            params["parent_folder_id"] = parent_folder_id
                        
                        response = httpx.get(
                            f"{api_url}/knowledge-base/documents",
                            params=params,
                            timeout=30.0,
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.kb_documents = result.get("documents", [])
                            st.success(f"✅ Loaded {result.get('count', 0)} documents")
                        else:
                            st.error(f"Failed to load documents: {response.text}")
                    except Exception as e:
                        st.error(f"Error loading documents: {str(e)}")
            
            # Display documents
            if "kb_documents" in st.session_state and st.session_state.kb_documents:
                st.write(f"**Total Documents:** {len(st.session_state.kb_documents)}")
                
                for doc in st.session_state.kb_documents:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{doc.get('name', 'Unknown')}**")
                            if doc.get('part_number'):
                                st.caption(f"Part Number: {doc['part_number']}")
                        with col2:
                            st.caption(f"ID: {doc.get('document_id', '')[:8]}...")
                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_{doc.get('document_id')}", type="secondary"):
                                with st.spinner("Deleting document..."):
                                    try:
                                        response = httpx.delete(
                                            f"{api_url}/knowledge-base/documents/{doc.get('document_id')}",
                                            timeout=30.0,
                                        )
                                        if response.status_code == 200:
                                            st.success("✅ Document deleted")
                                            # Refresh list
                                            if "kb_documents" in st.session_state:
                                                st.session_state.kb_documents = [
                                                    d for d in st.session_state.kb_documents
                                                    if d.get('document_id') != doc.get('document_id')
                                                ]
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to delete: {response.text}")
                                    except Exception as e:
                                        st.error(f"Error deleting document: {str(e)}")
                        st.divider()
            
            # Upload New Document Section
            st.subheader("📤 Upload Documents")
            
            uploaded_docs = st.file_uploader(
                "Choose document files (you can select multiple files)",
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True,
                help="Upload one or more documents to add to the Knowledge Base. All files will be uploaded to the same folder based on part number.",
            )
            
            col_upload1, col_upload2 = st.columns([2, 1])
            with col_upload1:
                doc_part_number = st.text_input(
                    "Part Number",
                    placeholder="E.g., CHS199100RECiN",
                    help="Enter the part number for all documents. All files will be organized in folder: Part_{part_number}",
                )
            with col_upload2:
                doc_custom_names = st.text_input(
                    "Custom Names (Optional)",
                    placeholder="Manual_1, Manual_2, Manual_3",
                    help="Optional comma-separated custom names (must match number of files)",
                )
            
            # Display uploaded files
            if uploaded_docs:
                st.write(f"**Selected Files ({len(uploaded_docs)}):**")
                for i, doc in enumerate(uploaded_docs, 1):
                    file_size = len(doc.read()) / 1024 / 1024  # Size in MB
                    doc.seek(0)  # Reset file pointer
                    st.caption(f"{i}. {doc.name} ({file_size:.2f} MB)")
            
            # Upload button
            upload_button_text = "📤 Upload All Documents" if uploaded_docs and len(uploaded_docs) > 1 else "📤 Upload Document"
            if st.button(upload_button_text, type="primary", use_container_width=True):
                if not uploaded_docs or len(uploaded_docs) == 0:
                    st.error("⚠️ Please select at least one file to upload")
                elif not doc_part_number:
                    st.error("⚠️ Please enter a part number")
                else:
                    # Validate custom names if provided
                    custom_names_list = None
                    if doc_custom_names:
                        custom_names_list = [name.strip() for name in doc_custom_names.split(",")]
                        if len(custom_names_list) != len(uploaded_docs):
                            st.error(f"⚠️ Number of custom names ({len(custom_names_list)}) must match number of files ({len(uploaded_docs)})")
                            st.stop()
                    
                    # Prepare files for upload
                    files = []
                    for doc in uploaded_docs:
                        doc.seek(0)  # Reset file pointer
                        files.append(("files", (doc.name, doc.read(), doc.type)))
                    
                    # Prepare form data
                    data = {
                        "part_number": doc_part_number,
                    }
                    if custom_names_list:
                        data["custom_names"] = ",".join(custom_names_list)
                    
                    # Upload with progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text(f"Uploading {len(uploaded_docs)} file(s)...")
                        progress_bar.progress(0.1)
                        
                        response = httpx.post(
                            f"{api_url}/knowledge-base/documents/batch",
                            files=files,
                            data=data,
                            timeout=300.0,  # Longer timeout for multiple files
                        )
                        
                        progress_bar.progress(0.9)
                        
                        if response.status_code == 200:
                            result = response.json()
                            progress_bar.progress(1.0)
                            status_text.empty()
                            
                            # Display success message
                            successful = result.get("successful_uploads", 0)
                            failed = result.get("failed_uploads", 0)
                            total = result.get("total_files", 0)
                            folder_id = result.get("folder_id")
                            
                            if successful > 0:
                                st.success(f"✅ Successfully uploaded {successful}/{total} document(s)!")
                                
                                if folder_id:
                                    st.info(f"📁 Folder: Part_{doc_part_number} (ID: {folder_id})")
                                else:
                                    st.info(f"📁 Documents uploaded (folder creation not available)")
                                
                                # Display successful uploads
                                if successful > 0:
                                    with st.expander(f"✅ Successful Uploads ({successful})", expanded=True):
                                        for upload_result in result.get("results", []):
                                            st.markdown(f"**{upload_result.get('file_name')}**")
                                            st.caption(f"Document ID: {upload_result.get('document_id')}")
                                            st.caption(f"Name: {upload_result.get('name')}")
                                            st.divider()
                            
                            # Display failed uploads
                            if failed > 0:
                                with st.expander(f"❌ Failed Uploads ({failed})", expanded=False):
                                    for failed_upload in result.get("failed", []):
                                        st.error(f"**{failed_upload.get('file_name')}**")
                                        st.caption(f"Error: {failed_upload.get('error')}")
                                        st.divider()
                            
                            # Refresh documents list
                            if "kb_documents" in st.session_state:
                                del st.session_state.kb_documents
                        else:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"Failed to upload documents: {response.text}")
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Error uploading documents: {str(e)}")
                        st.exception(e)
    
    # Footer
    st.divider()
    # st.caption("Powered by Gemini AI • Built with Streamlit • RAG-powered repair assistance")


if __name__ == "__main__":
    main()

