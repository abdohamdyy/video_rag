"""
Streamlit UI for Video Analysis and Repair Assistance
Analyzes videos to identify appliance defects and provides repair instructions with source citations.
"""

import asyncio
import time
from io import BytesIO
from typing import Optional

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


def process_video_assistance(video_bytes: bytes, mime_type: str, language: str, user_hint: Optional[str] = None):
    """Process video through the full assistance pipeline."""
    settings = get_settings()
    
    if not settings.gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please set it in your .env file or environment variables.")
    
    # Step 1: Video analysis
    with st.status("🎥 Analyzing video with AI...", expanded=True) as status:
        status.update(label="🎥 Analyzing video with Gemini AI...", state="running")
        result = analyze_video_with_gemini(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            video_bytes=video_bytes,
            video_mime_type=mime_type,
            language=language,
            user_hint=user_hint,
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
        citations = retrieve_support_docs(queries)
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
        )
        answer = answer_with_gemini(
            api_key=settings.gemini_api_key,
            model=settings.gemini_answer_model,
            prompt=grounded_prompt,
        )
        status.update(label="✅ Repair instructions ready", state="complete")
    
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
    }


def display_results(results: dict):
    """Display analysis results in a well-formatted UI."""
    analysis = results["analysis"]
    answer_text = results["answer"].get("text", "")
    citations = results["citations"]
    
    # Device Information Section
    st.header("📱 Device Information")
    col1, col2 = st.columns(2)
    
    with col1:
        appliance_type = analysis.get("appliance_type") or "Unknown"
        st.metric("Appliance Type", appliance_type)
    
    with col2:
        brand_model = analysis.get("brand_or_model") or "Not identified"
        st.metric("Brand/Model", brand_model)
    
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
    
    # # Repair Instructions Section
    # st.header("🔧 Repair Instructions")
    
    # # Recommended fix steps from analysis
    # fix_steps = analysis.get("recommended_fix_steps") or []
    # if fix_steps:
    #     st.subheader("Recommended Steps")
    #     for i, step in enumerate(fix_steps, 1):
    #         st.write(f"**Step {i}:** {step}")
    
    # # RAG-generated answer (main answer)
    # if answer_text:
    #     st.subheader("Detailed Repair Guide")
    #     st.markdown(answer_text)
    
    # # Tools and Parts Needed
    # tools_parts = analysis.get("tools_or_parts_needed") or []
    # if tools_parts:
    #     st.subheader("Required Tools & Parts")
    #     st.write(", ".join(tools_parts))
    
    # # Sources and Citations
    # if citations:
    #     st.header("📖 Sources & References")
    #     st.write(f"Found {len(citations)} relevant document sections:")
        
    #     with st.expander("View All Citations", expanded=False):
    #         for i, citation in enumerate(citations, 1):
    #             with st.container():
    #                 file_name = citation.get("file") or "Unknown"
    #                 page_num = citation.get("page")
    #                 snippet = citation.get("snippet", "")
    #                 distance = citation.get("distance")
                    
    #                 st.markdown(f"**Citation {i}**")
    #                 st.markdown(f"- **Source:** {file_name}")
    #                 if page_num is not None:
    #                     st.markdown(f"- **Page:** {page_num}")
    #                 if distance is not None:
    #                     st.markdown(f"- **Relevance Score:** {distance:.3f}")
    #                 if snippet:
    #                     st.markdown(f"- **Excerpt:** {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
    #                 st.divider()
    
    # # Follow-up Questions
    # follow_ups = results.get("follow_up_questions") or []
    # if follow_ups:
    #     st.header("❓ Follow-up Questions")
    #     st.write("To help diagnose the issue more accurately, consider answering these:")
    #     for i, question in enumerate(follow_ups[:5], 1):  # Show top 5
    #         st.write(f"{i}. {question}")
    
    # # Transcript (expandable)
    # transcript = results.get("transcript", "")
    # if transcript:
    #     with st.expander("View Video Transcript", expanded=False):
    #         st.write(transcript)


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
        
        st.divider()
        st.caption("💡 Tip: Provide context to get more accurate results")
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload Video", "🔗 Video URL"])
    
    video_bytes = None
    mime_type = None
    
    with tab1:
        st.subheader("Upload a Video File")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "webm", "mov", "avi"],
            help="Supported formats: MP4, WebM, MOV, AVI. Maximum size: 25MB",
        )
        
        if uploaded_file is not None:
            # Read file bytes once
            video_bytes = uploaded_file.read()
            mime_type = uploaded_file.type or "video/mp4"
            # Display video from bytes
            st.video(video_bytes)
    
    with tab2:
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
                    video_bytes = payload.data
                    mime_type = payload.mime_type
                    st.success(f"Video downloaded successfully ({len(video_bytes) / 1024 / 1024:.2f} MB)")
                    st.video(video_bytes)
            except VideoTooLargeError as e:
                st.error(f"Video too large: {str(e)}")
                video_bytes = None
            except VideoDownloadError as e:
                st.error(f"Failed to download video: {str(e)}")
                video_bytes = None
            except Exception as e:
                st.error(f"Error downloading video: {str(e)}")
                video_bytes = None
    
    # Analyze button
    if video_bytes:
        st.divider()
        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
            try:
                settings = get_settings()
                
                # Validate video size
                if len(video_bytes) > settings.max_video_bytes:
                    st.error(
                        f"Video is too large. Maximum size: {settings.max_video_bytes / 1024 / 1024:.0f} MB. "
                        f"Your video: {len(video_bytes) / 1024 / 1024:.2f} MB"
                    )
                else:
                    # Process through full pipeline
                    start_time = time.time()
                    results = process_video_assistance(
                        video_bytes,
                        mime_type,
                        language,
                        user_hint if user_hint.strip() else None,
                    )
                    elapsed_time = time.time() - start_time
                    
                    # Display results
                    st.success(f"✅ Analysis complete in {elapsed_time:.1f} seconds!")
                    display_results(results)
                    
            except ValueError as e:
                st.error(f"Configuration error: {str(e)}")
                st.info("Please check your .env file or environment variables for GEMINI_API_KEY")
            except VideoTooLargeError as e:
                st.error(f"Video too large: {str(e)}")
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
        st.info("👆 Please upload a video file or provide a video URL to begin analysis")
    
    # Footer
    st.divider()
    # st.caption("Powered by Gemini AI • Built with Streamlit • RAG-powered repair assistance")


if __name__ == "__main__":
    main()

