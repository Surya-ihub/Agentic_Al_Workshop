
import os
import streamlit as st
import json
import tempfile
import warnings
import time
from pypdf import PdfReader
import numpy as np
import faiss
import pickle
import re
import assemblyai as aai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from typing import Dict, List, Optional, Tuple
import asyncio
import functools
import random

# Configuration
GOOGLE_API_KEY = "AIzaSyBUgD1d0p8SNP73gbGTMb5mGCZajuHQ4Mo"  # Replace with your actual API key
ASSEMBLYAI_API_KEY = "c4d1580904294010b36eff07a1dbc992"  # Replace with your actual API key

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Suppress warnings
warnings.filterwarnings("ignore")

# Retry decorator with exponential backoff
def retry_with_exponential_backoff(max_retries=3, initial_delay=1, max_delay=30):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        retries += 1
                        if retries > max_retries:
                            raise
                        time.sleep(delay + random.uniform(0, 1))
                        delay = min(delay * 2, max_delay)
                    else:
                        raise
        return wrapper
    return decorator


class InterviewTranscriberAgent:
    """Agent for handling audio transcription using AssemblyAI"""
    
    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        
    def transcribe_with_speakers(self, audio_path: str) -> Dict:
        """Transcribe audio with speaker diarization using AssemblyAI"""
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=2,  # Interviewer and Candidate
                auto_highlights=True,
                sentiment_analysis=True,
                entity_detection=True
            )
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            # Convert to structured format
            segments = []
            for utterance in transcript.utterances:
                # Map speaker labels to roles
                speaker = "Interviewer" if utterance.speaker == "A" else "Candidate"
                segments.append({
                    "start_time": self._ms_to_timestamp(utterance.start),
                    "end_time": self._ms_to_timestamp(utterance.end),
                    "speaker": speaker,
                    "text": utterance.text,
                    "confidence": utterance.confidence
                })
            
            # Safely get sentiment analysis results
            sentiment = []
            try:
                if hasattr(transcript, 'sentiment_analysis_results') and transcript.sentiment_analysis_results:
                    sentiment = transcript.sentiment_analysis_results
            except AttributeError:
                pass
            
            return {
                "segments": segments,
                "summary": [],
                "sentiment": sentiment
            }
            
        except Exception as e:
            st.error(f"AssemblyAI transcription failed: {str(e)}")
            return {"segments": [], "summary": [], "sentiment": []}
    
    def _ms_to_timestamp(self, ms: int) -> str:
        """Convert milliseconds to HH:MM:SS format"""
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def clean_transcript_segments(self, segments: List[Dict]) -> List[Dict]:
        """Clean transcript segments using LangChain"""
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        prompt = ChatPromptTemplate.from_template("""
        Clean the following interview transcript segments by:
        1. Removing filler words and irrelevant content
        2. Keeping only meaningful questions and substantive answers
        3. Maintaining the JSON structure
        
        Segments: {segments}
        
        Return only the cleaned segments in JSON format.
        """)
        
        try:
            result = (prompt | llm).invoke({"segments": json.dumps(segments)})
            cleaned_text = result.content.strip()
            
            # Remove code block formatting if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()
            
            cleaned_segments = json.loads(cleaned_text)
            return cleaned_segments if isinstance(cleaned_segments, list) else segments
            
        except Exception as e:
            st.warning(f"Transcript cleaning failed: {str(e)}")
            return segments
    
    def extract_qa_pairs(self, segments: List[Dict]) -> List[Dict]:
        """Extract Q&A pairs from transcript segments, filtering out greetings"""
        qa_pairs = []
        current_question = None
        current_response = []
        
        # Filter out greetings and filler phrases
        FILTER_PHRASES = [
            "hello", "hi there", "good morning", "good afternoon", 
            "thanks for joining", "welcome", "how are you", "nice to meet you",
            "great", "perfect", "okay", "sure", "got it", "thank you"
        ]
        
        for segment in segments:
            text = segment["text"].strip().lower()
            speaker = segment["speaker"]
            
            # Skip greetings and filler responses
            if any(phrase in text for phrase in FILTER_PHRASES):
                continue
                
            if speaker == "Interviewer":
                # Save previous Q&A pair if exists
                if current_question and current_response:
                    qa_pairs.append({
                        "question": current_question,
                        "response": " ".join(current_response).strip(),
                        "confidence": segment.get("confidence", 0.0)
                    })
                
                current_question = segment["text"]
                current_response = []
            else:
                if current_question:  # Only collect responses if we have a question
                    current_response.append(segment["text"])
        
        # Add final Q&A pair
        if current_question and current_response:
            qa_pairs.append({
                "question": current_question,
                "response": " ".join(current_response).strip(),
                "confidence": segments[-1].get("confidence", 0.0)
            })
        
        return qa_pairs


class ExpectedAnswerRetrieverAgent:
    """Enhanced RAG system using LangChain for retrieving expected answers"""
    
    def __init__(self, google_api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        self.vectorstore = None
        self.documents = []
    
    def build_knowledge_base(self, pdf_file) -> bool:
        """Build knowledge base from PDF using LangChain"""
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name
            
            # Load and process PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            self.documents = text_splitter.split_documents(pages)
            
            if self.documents:
                # Create vector store
                self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Knowledge base creation failed: {str(e)}")
            return False
    
    def retrieve_relevant_context(self, question: str, k: int = 3) -> List[str]:
        """Retrieve relevant context for a question"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(question, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            st.warning(f"Context retrieval failed: {str(e)}")
            return []
    
    def extract_focused_reference(self, question: str, contexts: List[str]) -> str:
        """Extract the most relevant reference from context"""
        if not contexts:
            return ""
        
        # Combine contexts
        context_text = "\n\n".join(contexts)
        
        # Use LLM to extract focused reference
        prompt = ChatPromptTemplate.from_template("""
        Extract ONLY the most relevant reference information that directly answers this specific question:
        Question: {question}
        
        From this context:
        {context}
        
        Return ONLY the relevant information as a concise reference answer.
        """)
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        chain = prompt | llm
        
        try:
            result = chain.invoke({"question": question, "context": context_text})
            return result.content.strip()
        except Exception as e:
            st.warning(f"Reference extraction failed: {str(e)}")
            return contexts[0] if contexts else ""


class ResponseEvaluatorAgent:
    """Agent for evaluating technical accuracy, communication, and behavioral aspects"""
    
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=google_api_key
        )
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup tools for the evaluator agent"""
        
        @retry_with_exponential_backoff(max_retries=3)
        def technical_evaluator(input_text: str) -> str:
            """Evaluate technical aspects of the answer"""
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            prompt = ChatPromptTemplate.from_template("""
            As a technical interviewer, evaluate ONLY the technical accuracy and depth of this answer:
            
            Question: {question}
            Answer: {answer}
            Reference: {reference}
            
            Provide:
            1. Technical accuracy score (1-5)
            2. Depth of knowledge demonstrated
            3. Areas of strength
            4. Technical gaps identified
            5. Specific improvement suggestions
            
            Format as JSON with keys: score, accuracy, depth, strengths, gaps, suggestions
            """)
            
            try:
                parts = input_text.split("|||")
                if len(parts) >= 3:
                    result = (prompt | self.llm).invoke({
                        "question": parts[0],
                        "answer": parts[1], 
                        "reference": parts[2]
                    })
                    return result.content
                return "Invalid input format"
            except Exception as e:
                return f"Technical evaluation failed: {str(e)}"
        
        @retry_with_exponential_backoff(max_retries=3)
        def communication_evaluator(input_text: str) -> str:
            """Evaluate communication skills"""
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            prompt = ChatPromptTemplate.from_template("""
            As a communication expert, evaluate ONLY the communication effectiveness:
            
            Answer: {answer}
            
            Assess:
            1. Clarity and structure (1-5)
            2. Professional language use
            3. Confidence level
            4. Engagement and enthusiasm
            5. Areas for improvement
            
            Format as JSON with keys: clarity_score, language_quality, confidence_level, engagement, improvements
            """)
            
            try:
                result = (prompt | self.llm).invoke({"answer": input_text})
                return result.content
            except Exception as e:
                return f"Communication evaluation failed: {str(e)}"
        
        @retry_with_exponential_backoff(max_retries=3)
        def behavioral_evaluator(input_text: str) -> str:
            """Evaluate behavioral aspects"""
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            prompt = ChatPromptTemplate.from_template("""
            As a behavioral interviewer, evaluate ONLY the behavioral indicators:
            
            Question: {question}
            Answer: {answer}
            
            Look for:
            1. Leadership qualities
            2. Problem-solving approach
            3. Teamwork indicators
            4. Adaptability
            5. Cultural fit indicators
            
            Format as JSON with keys: leadership_score, problem_solving, teamwork, adaptability, cultural_fit, examples
            """)
            
            try:
                parts = input_text.split("|||")
                if len(parts) >= 2:
                    result = (prompt | self.llm).invoke({
                        "question": parts[0],
                        "answer": parts[1]
                    })
                    return result.content
                return "Invalid input format"
            except Exception as e:
                return f"Behavioral evaluation failed: {str(e)}"
        
        @retry_with_exponential_backoff(max_retries=3)
        def answer_rater(input_text: str) -> str:
            """Rate answer against reference"""
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            prompt = ChatPromptTemplate.from_template("""
            Rate the candidate's answer against the reference answer on a scale of 1-10:
            
            Question: {question}
            Candidate Answer: {answer}
            Reference Answer: {reference}
            
            Provide:
            1. Numerical rating (1-10)
            2. Key strengths
            3. Key weaknesses
            4. Comparison analysis
            
            Format as JSON with keys: rating, strengths, weaknesses, comparison
            """)
            
            try:
                parts = input_text.split("|||")
                if len(parts) >= 3:
                    result = (prompt | self.llm).invoke({
                        "question": parts[0],
                        "answer": parts[1], 
                        "reference": parts[2]
                    })
                    return result.content
                return "Invalid input format"
            except Exception as e:
                return f"Answer rating failed: {str(e)}"
        
        self.tools = {
            "technical_evaluator": technical_evaluator,
            "communication_evaluator": communication_evaluator,
            "behavioral_evaluator": behavioral_evaluator,
            "answer_rater": answer_rater
        }
    
    def evaluate_response(self, question: str, answer: str, reference: str) -> Dict:
        """Evaluate candidate response across multiple dimensions"""
        results = {}
        
        try:
            # Technical evaluation
            tech_input = f"{question}|||{answer}|||{reference}"
            results["technical"] = self.tools["technical_evaluator"](tech_input)
            
            # Communication evaluation
            results["communication"] = self.tools["communication_evaluator"](answer)
            
            # Behavioral evaluation
            behav_input = f"{question}|||{answer}"
            results["behavioral"] = self.tools["behavioral_evaluator"](behav_input)
            
            # Answer rating
            rating_input = f"{question}|||{answer}|||{reference}"
            results["rating"] = self.tools["answer_rater"](rating_input)
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            results["error"] = str(e)
        
        return results


class FeedbackGeneratorAgent:
    """Agent for generating actionable feedback and improvement suggestions"""
    
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=google_api_key
        )
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup tools for the feedback generator"""
        
        @retry_with_exponential_backoff(max_retries=3)
        def improvement_advisor(input_text: str) -> str:
            """Provide specific improvement recommendations"""
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            prompt = ChatPromptTemplate.from_template("""
            Based on the evaluation results, provide specific, actionable improvement advice:
            
            Evaluation Data: {eval_data}
            
            Provide:
            1. Top 3 priority areas for improvement
            2. Specific action steps for each area
            3. Practice exercises or techniques
            4. Recommended resources (books, courses, websites)
            5. Timeline for improvement
            
            Format as structured advice with clear action items.
            """)
            
            try:
                result = (prompt | self.llm).invoke({"eval_data": input_text})
                return result.content
            except Exception as e:
                return f"Improvement advice generation failed: {str(e)}"
        
        @retry_with_exponential_backoff(max_retries=3)
        def grammar_corrector(input_text: str) -> str:
            """Correct grammar and improve phrasing"""
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            prompt = ChatPromptTemplate.from_template("""
            Correct and improve the following response while maintaining the original meaning:
            
            Original Response: {response}
            
            Provide:
            1. Grammatically corrected version
            2. Improved phrasing suggestions
            3. Explanation of changes
            
            Format as JSON with keys: corrected_response, suggestions, explanation
            """)
            
            try:
                result = (prompt | self.llm).invoke({"response": input_text})
                return result.content
            except Exception as e:
                return f"Grammar correction failed: {str(e)}"
        
        @retry_with_exponential_backoff(max_retries=3)
        def feedback_formatter(evaluation_results: Dict) -> str:
            """Format evaluation results into comprehensive feedback"""
            # Add delay to prevent rate limiting
            time.sleep(1)
            
            prompt = ChatPromptTemplate.from_template("""
            Compile the following evaluation results into a structured feedback report:
            
            Technical Evaluation: {technical}
            Communication Assessment: {communication}
            Behavioral Analysis: {behavioral}
            Answer Rating: {rating}
            
            Organize the feedback into these sections:
            1. Overall Performance Summary
            2. Technical Skills Assessment
            3. Communication Effectiveness
            4. Behavioral Indicators
            5. Final Rating and Comparison
            
            Use clear headings and bullet points for readability.
            """)
            
            try:
                result = (prompt | self.llm).invoke({
                    "technical": evaluation_results.get("technical", "N/A"),
                    "communication": evaluation_results.get("communication", "N/A"),
                    "behavioral": evaluation_results.get("behavioral", "N/A"),
                    "rating": evaluation_results.get("rating", "N/A")
                })
                return result.content
            except Exception as e:
                return f"Feedback formatting failed: {str(e)}"
        
        self.tools = {
            "improvement_advisor": improvement_advisor,
            "grammar_corrector": grammar_corrector,
            "feedback_formatter": feedback_formatter
        }
    
    def generate_feedback(self, evaluation_results: Dict, answer: str) -> Dict:
        """Generate comprehensive feedback from evaluation results"""
        feedback = {}
        
        try:
            # Format comprehensive feedback
            feedback["formatted"] = self.tools["feedback_formatter"](evaluation_results)
            
            # Generate improvement suggestions
            eval_summary = f"Technical: {evaluation_results.get('technical', '')}\n"
            eval_summary += f"Communication: {evaluation_results.get('communication', '')}\n"
            eval_summary += f"Behavioral: {evaluation_results.get('behavioral', '')}\n"
            eval_summary += f"Rating: {evaluation_results.get('rating', '')}"
            
            feedback["improvement_advice"] = self.tools["improvement_advisor"](eval_summary)
            
            # Generate grammar corrections
            feedback["grammar_correction"] = self.tools["grammar_corrector"](answer)
            
        except Exception as e:
            st.error(f"Feedback generation failed: {str(e)}")
            feedback["error"] = str(e)
        
        return feedback


class AgentOrchestrator:
    """Coordinates the agent workflow and manages state"""
    
    def __init__(self):
        self.transcriber = InterviewTranscriberAgent(ASSEMBLYAI_API_KEY)
        self.retriever = ExpectedAnswerRetrieverAgent(GOOGLE_API_KEY)
        self.evaluator = ResponseEvaluatorAgent(GOOGLE_API_KEY)
        self.feedback_generator = FeedbackGeneratorAgent(GOOGLE_API_KEY)
        self.interview_data = {}
    
    def process_interview(self, audio_file, pdf_file, evaluation_depth, 
                         include_sentiment, include_highlights):
        """Process interview through the agent chain"""
        # Step 1: Build knowledge base from PDF
        if pdf_file:
            with st.spinner("🔨 Building knowledge base..."):
                if not self.retriever.build_knowledge_base(pdf_file):
                    st.error("Failed to build knowledge base")
                    return
        
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.getvalue())
            audio_path = tmp.name
        
        # Step 2: Transcribe audio
        with st.spinner("🎯 Transcribing audio..."):
            transcript_data = self.transcriber.transcribe_with_speakers(audio_path)
            
            if not transcript_data["segments"]:
                st.error("Transcription failed. Please check your audio file.")
                return
            
            # Clean and extract Q&A pairs
            cleaned_segments = self.transcriber.clean_transcript_segments(
                transcript_data["segments"]
            )
            qa_pairs = self.transcriber.extract_qa_pairs(cleaned_segments)
            
            if not qa_pairs:
                st.warning("No clear question-answer pairs found in the transcript.")
                return
            
            self.interview_data = {
                "transcript": transcript_data,
                "qa_pairs": qa_pairs
            }
        
        # Step 3: Process each Q&A pair through the agent chain
        results = []
        for i, pair in enumerate(qa_pairs):
            with st.spinner(f"🔍 Analyzing question {i+1}/{len(qa_pairs)}..."):
                # Step 3a: Retrieve reference answer
                relevant_contexts = self.retriever.retrieve_relevant_context(pair["question"])
                reference_answer = self.retriever.extract_focused_reference(
                    pair["question"], relevant_contexts
                )
                
                # Step 3b: Evaluate response
                evaluation_results = self.evaluator.evaluate_response(
                    pair["question"], pair["response"], reference_answer
                )
                
                # Step 3c: Generate feedback
                feedback = self.feedback_generator.generate_feedback(
                    evaluation_results, pair["response"]
                )
                
                # Add to results
                results.append({
                    "question": pair["question"],
                    "response": pair["response"],
                    "reference": reference_answer,
                    "evaluation": evaluation_results,
                    "feedback": feedback
                })
                
                # Add delay between questions to prevent rate limiting
                time.sleep(10)  # Increased delay to prevent API rate limits
        
        self.interview_data["results"] = results
        return self.interview_data


def main():
    st.set_page_config(
        page_title="AI Mock Interview Feedback System", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎤 AI Mock Interview Feedback System")
    st.markdown("""
    **Enhanced Multi-Agent Architecture:**
    - 🎙️ **Interview Transcriber Agent** - Converts audio to structured transcripts
    - 🔍 **Expected Answer Retriever Agent** - Retrieves benchmark answers from knowledge base
    - 📊 **Response Evaluator Agent** - Assesses technical, communication, and behavioral aspects
    - ✨ **Feedback Generator Agent** - Creates actionable improvement suggestions
    """)
    
    # Initialize orchestrator
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = AgentOrchestrator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Configuration")
        
        # PDF upload
        pdf_file = st.file_uploader("Upload Reference Q&A PDF", type=["pdf"])
        
        # Settings
        # st.divider()
        # st.subheader("⚙️ Settings")
        # evaluation_depth = st.selectbox(
        #     "Evaluation Depth",
        #     ["Quick", "Comprehensive", "Detailed"],
        #     index=1
        # )
        
        # include_sentiment = st.checkbox("Include Sentiment Analysis", True)
        # include_highlights = st.checkbox("Include Key Highlights", False)  # Disabled by default
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Audio upload
        uploaded_file = st.file_uploader(
            "🎵 Upload Interview Recording", 
            type=["mp3", "wav", "m4a", "flac"]
        )
        
        if uploaded_file:
            if st.button("🚀 Process Interview"):
                with st.spinner("Processing interview through agent chain..."):
                    interview_data = st.session_state.orchestrator.process_interview(
                        uploaded_file, 
                        pdf_file,
                        evaluation_depth,
                        include_sentiment,
                        include_highlights
                    )
                    
                    if "results" in interview_data:
                        display_results(interview_data)
    
    with col2:
        st.subheader("📊 System Status")
        
        # Status indicators
        kb_status = "✅ Ready" if st.session_state.orchestrator.retriever.vectorstore else "❌ Not Built"
        st.metric("Knowledge Base", kb_status)
        
        transcriber_status = "✅ Ready"
        st.metric("Transcriber Agent", transcriber_status)
        
        evaluator_status = "✅ Ready"
        st.metric("Evaluator Agent", evaluator_status)
        
        feedback_status = "✅ Ready"
        st.metric("Feedback Agent", feedback_status)
        
        # Rate limit protection section
        st.divider()
        st.subheader("⚠️ Rate Limit Protection")
        st.write("Gemini API Safeguards:")
        st.progress(85)
        st.caption("Automatic retries enabled with exponential backoff")
        st.caption("30s timeout per request with 3 retry attempts")


def display_results(interview_data: Dict):
    """Display the processed interview results"""
    st.subheader("📊 Interview Analysis Report")
    
    # Show transcript summary
    with st.expander("📝 View Transcript Summary", expanded=False):
        st.write(f"Found {len(interview_data['qa_pairs'])} question-answer pairs")
        
        for i, segment in enumerate(interview_data['transcript']['segments'][:5]):
            speaker_emoji = "👨‍💼" if segment["speaker"] == "Interviewer" else "👤"
            st.write(f"{speaker_emoji} **{segment['speaker']}** ({segment['start_time']}): {segment['text']}")
    
    # Display results for each question
    for i, result in enumerate(interview_data["results"]):
        with st.expander(f"❓ Question {i+1}: {result['question'][:80]}...", expanded=True):
            st.write("**❓ Question:**")
            st.write(result["question"])
            
            st.write("**💬 Candidate Response:**")
            st.write(result["response"])
            
            if result["reference"]:
                st.write("**📚 Reference Answer:**")
                st.info(result["reference"])
            
            # Display evaluation results
            st.subheader("📊 Evaluation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "technical" in result["evaluation"]:
                    st.write("**🔧 Technical Evaluation**")
                    st.json(result["evaluation"]["technical"])
                
                if "behavioral" in result["evaluation"]:
                    st.write("**🎭 Behavioral Analysis**")
                    st.json(result["evaluation"]["behavioral"])
            
            with col2:
                if "communication" in result["evaluation"]:
                    st.write("**💬 Communication Assessment**")
                    st.json(result["evaluation"]["communication"])
                
                if "rating" in result["evaluation"]:
                    st.write("**⭐ Answer Rating**")
                    st.json(result["evaluation"]["rating"])
            
            # Display feedback
            st.subheader("✨ Feedback & Suggestions")
            
            if "formatted" in result["feedback"]:
                st.write("**📝 Comprehensive Feedback**")
                st.write(result["feedback"]["formatted"])
            
            if "improvement_advice" in result["feedback"]:
                st.write("**📈 Improvement Recommendations**")
                st.success(result["feedback"]["improvement_advice"], icon="💡")
            
            if "grammar_correction" in result["feedback"]:
                st.write("**✍️ Grammar Correction**")
                st.json(result["feedback"]["grammar_correction"])
            
            st.divider()


if __name__ == "__main__":
    main()