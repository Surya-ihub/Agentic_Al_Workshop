# import os
# import streamlit as st
# import json
# import tempfile
# import google.generativeai as genai
# from pydub import AudioSegment
# import warnings
# from pypdf import PdfReader
# import numpy as np
# import faiss
# import pickle
# import re
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores.faiss import FAISS
# from langchain.prompts import ChatPromptTemplate

# # Suppress Pydub warning about ffmpeg
# warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

# # Configuration - FIX: Use your actual API key in both places
# API_KEY = "AIzaSyD-ammIVL6zKq2ay7F35e7sSp8fKGQRqAg"  # Your actual API key
# MODEL_NAME = "gemini-1.5-flash"

# # Set environment variable for LangChain components
# os.environ["GOOGLE_API_KEY"] = API_KEY

# # Initialize Gemini
# genai.configure(api_key=API_KEY)
# model = genai.GenerativeModel(MODEL_NAME)


# class InterviewTranscriberAgent:
#     def __init__(self):
#         self.model = genai.GenerativeModel(MODEL_NAME)

#     def transcribe(self, audio_path: str) -> dict:
#         """Transcribe audio to text with timestamps"""
#         try:
#             # Upload the audio file
#             audio_file = genai.upload_file(audio_path)

#             prompt = """
#             You are a professional interview transcription system. Transcribe the following interview audio 
#             and provide the output in valid JSON format with time-stamped segments:
            
#             {
#                 "segments": [
#                     {
#                         "start_time": "00:00:00",
#                         "speaker": "Interviewer" or "Candidate",
#                         "text": "transcribed text here"
#                     },
#                     ...
#                 ]
#             }
            
#             Important:
#             1. Ensure the output is valid JSON that can be parsed with json.loads()
#             2. Don't include any markdown or code formatting
#             3. Be precise with speaker identification
#             """

#             response = self.model.generate_content([prompt, audio_file])

#             # Clean the response text to ensure it's valid JSON
#             response_text = response.text.strip()
#             if response_text.startswith("```json"):
#                 response_text = response_text[7:-3].strip()
#             elif response_text.startswith("```"):
#                 response_text = response_text[3:-3].strip()

#             return json.loads(response_text)
#         except Exception as e:
#             st.error(f"Transcription failed: {str(e)}")
#             return {"segments": []}


# class ResponseEvaluatorAgent:
#     def __init__(self):
#         self.model = genai.GenerativeModel(MODEL_NAME)

#     def evaluate_response(self, question: str, response: str, role: str) -> dict:
#         """Evaluate a single Q&A pair"""
#         prompt = f"""
#         Evaluate this interview response for a {role} position. Provide feedback in valid JSON format:
        
#         Question: {question}
#         Response: {response}
        
#         Use this JSON structure:
#         {{
#             "technical": {{
#                 "score": 1-5,
#                 "feedback": "technical evaluation here"
#             }},
#             "communication": {{
#                 "score": 1-5,
#                 "feedback": "communication evaluation here"
#             }},
#             "behavioral": {{
#                 "score": 1-5,
#                 "feedback": "behavioral evaluation here"
#             }},
#             "overall_feedback": "summary feedback here"
#         }}
        
#         Important: Only return valid JSON, no additional text or formatting.
#         """

#         try:
#             result = self.model.generate_content(prompt)
#             return json.loads(result.text)
#         except Exception as e:
#             st.error(f"Evaluation failed: {str(e)}")
#             return {}


# def convert_audio_format(input_path, output_format="wav"):
#     """Convert audio to WAV format for better compatibility"""
#     sound = AudioSegment.from_file(input_path)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp:
#         sound.export(tmp.name, format=output_format)
#         return tmp.name


# def extract_qa_from_text(text):
#     lines = text.splitlines()
#     qa_pairs = []
#     question = None
#     answer_lines = []
#     for line in lines:
#         if line.strip().endswith("?"):
#             if question and answer_lines:
#                 qa_pairs.append(
#                     {"question": question, "answer": " ".join(answer_lines).strip()}
#                 )
#             question = line.strip()
#             answer_lines = []
#         else:
#             if line.strip():
#                 answer_lines.append(line.strip())
#     if question and answer_lines:
#         qa_pairs.append(
#             {"question": question, "answer": " ".join(answer_lines).strip()}
#         )
#     return qa_pairs


# def get_gemini_embedding(text):
#     """Get embedding for text using Gemini's embed_content API."""
#     try:
#         response = genai.embed_content(
#             model="models/embedding-001", content=text, task_type="retrieval_document"
#         )
#         return np.array(response["embedding"], dtype=np.float32)
#     except Exception as e:
#         st.error(f"Embedding failed: {str(e)}")
#         return None


# def retrieve_closest_qa(question, faiss_index, qa_pairs_pdf):
#     """Retrieve the closest Q&A pair from FAISS for a given question."""
#     if faiss_index is None or not qa_pairs_pdf:
#         return None
#     q_emb = get_gemini_embedding(question)
#     if q_emb is None:
#         return None
#     q_emb = np.expand_dims(q_emb, axis=0)
#     D, I = faiss_index.search(q_emb, 1)  # 1 nearest neighbor
#     idx = I[0][0]
#     if idx < len(qa_pairs_pdf):
#         return qa_pairs_pdf[idx]
#     return None


# def save_faiss_index(index, qa_pairs, path_prefix):
#     faiss.write_index(index, path_prefix + ".faiss")
#     with open(path_prefix + ".pkl", "wb") as f:
#         pickle.dump(qa_pairs, f)


# def load_faiss_index(path_prefix):
#     try:
#         index = faiss.read_index(path_prefix + ".faiss")
#         with open(path_prefix + ".pkl", "rb") as f:
#             qa_pairs = pickle.load(f)
#         return index, qa_pairs
#     except Exception:
#         return None, None


# def clean_transcript_with_gemini_lc(transcript_segments):
#     """Use LangChain Gemini LLM to clean transcript segments, keeping only meaningful Q&A, and post-filtering for short/filler segments."""
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     prompt = ChatPromptTemplate.from_template(
#         """
# You are an expert at cleaning interview transcripts. Given the following list of transcript segments (as JSON), do the following:
# - Remove all segments that are not actual interview questions (from the interviewer) or substantive answers (from the candidate).
# - Remove all short or irrelevant greetings, acknowledgments, and filler words (e.g., 'Yes, Lakshman.', 'Yeah.', 'Okay.', 'Hi.', 'Hello.', 'Thank you.', 'Thanks.', 'Okay.', 'Sure.', 'Yes.', 'No.', 'Alright.', 'Uh', 'Um', 'Okay. Thank you.', etc.).
# - Only keep segments where:
#     - The interviewer is asking a real question (usually ends with a question mark or starts with 'What', 'How', 'Why', 'Explain', etc.).
#     - The candidate is providing a meaningful answer (more than 5 words, not just 'Yes', 'No', 'Thank you', etc.).
#     - If there is a meaningful greeting (e.g., 'Good morning', 'Welcome to the interview'), keep it, but remove all other short greetings.
# Return the cleaned list in the same JSON format (list of segments, each with start_time, speaker, text).

# Segments:
# {segments}

# Important: Only return valid JSON, no extra text or formatting.
# """
#     )
#     chain = prompt | llm
#     try:
#         result = chain.invoke({"segments": transcript_segments})
#         cleaned_text = result.content.strip()
#         if cleaned_text.startswith("```json"):
#             cleaned_text = cleaned_text[7:-3].strip()
#         elif cleaned_text.startswith("```"):
#             cleaned_text = cleaned_text[3:-3].strip()
#         cleaned_segments = json.loads(cleaned_text)
#         if not cleaned_segments or not isinstance(cleaned_segments, list):
#             st.warning("Transcript cleaning returned no segments. Using original transcript.")
#             return transcript_segments
#         # Python post-filter: remove any segment with <5 words unless it's a valid greeting or question
#         valid_greetings = ["good morning", "welcome", "good afternoon", "good evening"]
#         def is_valid(segment):
#             text = segment["text"].strip().lower()
#             if len(text.split()) >= 5:
#                 return True
#             if any(greet in text for greet in valid_greetings):
#                 return True
#             if segment["speaker"].lower() == "interviewer" and (text.endswith("?") or text.startswith(("what", "how", "why", "explain"))):
#                 return True
#             return False
#         filtered_segments = [seg for seg in cleaned_segments if is_valid(seg)]
#         if not filtered_segments:
#             st.warning("All segments were filtered out. Using original transcript.")
#             return transcript_segments
#         return filtered_segments
#     except Exception as e:
#         st.warning(
#             f"Transcript cleaning failed: {str(e)}. Proceeding with original transcript."
#         )
#         return transcript_segments


# def process_pdf_with_langchain(pdf_file):
#     """Load PDF, split into chunks, embed with Gemini, and store in FAISS using LangChain."""
#     # Save PDF to a temp file for loader
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(pdf_file.getvalue())
#         tmp_path = tmp.name
#     # Load PDF with LangChain
#     loader = PyPDFLoader(tmp_path)
#     pages = loader.load()
#     # Combine all text for debugging purposes (optional, could be removed)
#     all_text = "\n".join([p.page_content for p in pages])
#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,  # Adjust chunk size as needed
#         chunk_overlap=200,  # Adjust chunk overlap as needed
#     )
#     lc_docs = text_splitter.split_documents(pages)
#     # Embed with Gemini (GoogleGenerativeAIEmbeddings)
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     if lc_docs:
#         vectorstore = FAISS.from_documents(lc_docs, embeddings)
#         return vectorstore, lc_docs, all_text  # Return all_text for debugging
#     else:
#         return None, lc_docs, all_text


# def extract_relevant_answer_from_chunk(question, chunk, llm):
#     """Use Gemini LLM to extract the most relevant answer from a chunk for a given question."""
#     prompt = ChatPromptTemplate.from_template(
#         """
# Given the following chunk of text and a specific interview question, extract only the answer that best matches the question. If the chunk contains multiple Q&A pairs, return only the answer for the given question. If the answer is not present, return an empty string.

# Chunk:
# {chunk}

# Question:
# {question}

# Important: Only return the answer text, no extra formatting or explanation.
# """
#     )
#     chain = prompt | llm
#     try:
#         result = chain.invoke({"chunk": chunk, "question": question})
#         answer = result.content.strip()
#         return answer
#     except Exception as e:
#         st.warning(f"Failed to extract relevant answer: {str(e)}. Using full chunk.")
#         return chunk


# def main():
#     st.set_page_config(page_title="AI Mock Interview Feedback", layout="wide")
#     st.title("AI Mock Interview Feedback Generator")
#     st.markdown(
#         """
#     **Instructions:**
#     1. Upload a PDF with Q&A pairs (format: Q: ... A: ...).
#     2. Upload an interview audio file (mp3, wav, m4a).
#     3. The app will extract, retrieve, and evaluate answers using AI.
#     """
#     )

#     # Initialize agents
#     transcriber = InterviewTranscriberAgent()
#     evaluator = ResponseEvaluatorAgent()

#     # Sidebar configuration
#     with st.sidebar:
#         st.header("Interview Details")
#         pdf_file = st.file_uploader("Upload Q&A PDF", type=["pdf"])

#     # PDF Q&A extraction and embedding
#     qa_pairs_pdf = []
#     faiss_index = None
#     qa_texts = []
#     pdf_id = None
#     if pdf_file is not None:
#         # Check file size (4MB = 4 * 1024 * 1024 bytes)
#         if len(pdf_file.getvalue()) > 4 * 1024 * 1024:
#             st.error("PDF file is too large. Please upload a file smaller than 4MB.")
#             return
#         # Check number of pages (max 8)
#         try:
#             reader = PdfReader(pdf_file)
#             num_pages = len(reader.pages)
#             if num_pages > 200:
#                 st.error(
#                     f"PDF has {num_pages} pages. Please upload a PDF with no more than 8 pages."
#                 )
#                 return
#         except Exception as e:
#             st.error(f"Failed to read PDF: {str(e)}")
#             return
#         pdf_id = str(hash(pdf_file.getvalue()))
#         with st.spinner(
#             "Extracting and chunking PDF content, building vector store with LangChain..."
#         ):
#             try:
#                 vectorstore, lc_docs, all_text = process_pdf_with_langchain(pdf_file)
#                 if not lc_docs:
#                     st.warning(
#                         "No text chunks extracted from PDF. This might indicate an issue with PDF text extraction. Here is the raw extracted text for debugging:"
#                     )
#                     st.code(all_text)
#                 else:
#                     st.success(f"Extracted {len(lc_docs)} text chunks from PDF.")
#                     st.subheader("First 5 Extracted Chunks from PDF (for verification)")
#                     for i, doc in enumerate(lc_docs[:5]):
#                         st.markdown(f"**Chunk {i+1}:** {doc.page_content[:200]}...")
#                     st.session_state["vectorstore"] = vectorstore
#                     st.session_state["lc_docs"] = lc_docs
#             except Exception as e:
#                 st.error(f"Failed to process PDF with LangChain: {str(e)}")

#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Upload Interview Recording", type=["mp3", "wav", "m4a"]
#     )

#     if uploaded_file is not None:
#         vectorstore = st.session_state.get("vectorstore")
#         lc_docs = st.session_state.get("lc_docs")
#         if not vectorstore or not lc_docs:
#             st.error(
#                 "Please upload a valid Q&A PDF and ensure the index is built before uploading audio."
#             )
#             return
#         with st.spinner("Processing your interview..."):
#             # Save and convert audio file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
#                 tmp.write(uploaded_file.getvalue())
#                 audio_path = tmp.name
#             try:
#                 audio_path = convert_audio_format(audio_path)
#             except Exception as e:
#                 st.warning(f"Audio conversion warning: {str(e)}")
#             # Step 1: Transcription
#             st.subheader("Interview Transcription")
#             transcript = transcriber.transcribe(audio_path)
#             if transcript and "segments" in transcript:
#                 st.json(transcript)
#                 # Step 1.5: Clean transcript using LangChain Gemini LLM
#                 with st.spinner("Cleaning transcript..."):
#                     cleaned_segments = clean_transcript_with_gemini_lc(
#                         transcript["segments"]
#                     )
#                 st.subheader("Cleaned Transcript Segments")
#                 st.json(cleaned_segments)
#                 # Step 2: Extract Q&A pairs from cleaned transcript
#                 st.subheader("Question & Answer Analysis (RAG)")
#                 qa_pairs = []
#                 current_question = None
#                 current_response = []
#                 for segment in cleaned_segments:
#                     if segment["speaker"] == "Interviewer":
#                         if current_question and current_response:
#                             qa_pairs.append(
#                                 {
#                                     "question": current_question,
#                                     "response": " ".join(current_response),
#                                 }
#                             )
#                         current_question = segment["text"]
#                         current_response = []
#                     else:
#                         current_response.append(segment["text"])
#                 if current_question and current_response:
#                     qa_pairs.append(
#                         {
#                             "question": current_question,
#                             "response": " ".join(current_response),
#                         }
#                     )
#                 # Step 3: RAG retrieval and feedback using LangChain
#                 llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#                 for i, pair in enumerate(qa_pairs):
#                     with st.expander(f"Q{i+1}: {pair['question'][:50]}..."):
#                         st.write(f"**User Response:** {pair['response']}")
#                         # Retrieve closest chunk from vectorstore (RAG)
#                         retr_res = vectorstore.similarity_search(pair["question"], k=1)
#                         if retr_res:
#                             # Use the entire retrieved chunk as the reference answer
#                             reference_chunk_content = retr_res[0].page_content
#                             # Extract only the most relevant answer for the current question
#                             relevant_answer = extract_relevant_answer_from_chunk(pair["question"], reference_chunk_content, llm)
#                             st.write(f"**Reference Answer:**")
#                             st.info(relevant_answer)
#                         else:
#                             st.warning(
#                                 "No relevant reference content found for this question."
#                             )
#                         # Generate structured feedback using LangChain Gemini LLM
#                         if retr_res:
#                             feedback_prompt = ChatPromptTemplate.from_template(
#                                 """
# You are an expert interview evaluator. Compare the user's answer to the provided reference answer.
# Return feedback in this JSON format:
# {{
#   "technical": {{"score": 1-5, "feedback": "..."}},
#   "communication": {{"score": 1-5, "feedback": "..."}},
#   "behavioral": {{"score": 1-5, "feedback": "..."}},
#   "improvement_tips": "...",
#   "resource_links": ["https://...", "..."]
# }}
# Question: {question}
# Reference Answer: {reference_answer}
# User Answer: {user_answer}
# Important: Only return valid JSON, no extra text.
# """
#                             )
#                             chain = feedback_prompt | llm
#                             try:
#                                 result = chain.invoke(
#                                     {
#                                         "question": pair["question"],
#                                         "reference_answer": relevant_answer,
#                                         "user_answer": pair["response"],
#                                     }
#                                 )
#                                 feedback_text = result.content.strip()
#                                 if feedback_text.startswith("```json"):
#                                     feedback_text = feedback_text[7:-3].strip()
#                                 elif feedback_text.startswith("```"):
#                                     feedback_text = feedback_text[3:-3].strip()
#                                 feedback = json.loads(feedback_text)
#                                 cols = st.columns(3)
#                                 with cols[0]:
#                                     st.metric(
#                                         "Technical",
#                                         feedback.get("technical", {}).get(
#                                             "score", "N/A"
#                                         ),
#                                         "out of 5",
#                                     )
#                                     st.caption(
#                                         feedback.get("technical", {}).get(
#                                             "feedback", ""
#                                         )
#                                     )
#                                 with cols[1]:
#                                     st.metric(
#                                         "Communication",
#                                         feedback.get("communication", {}).get(
#                                             "score", "N/A"
#                                         ),
#                                         "out of 5",
#                                     )
#                                     st.caption(
#                                         feedback.get("communication", {}).get(
#                                             "feedback", ""
#                                         )
#                                     )
#                                 with cols[2]:
#                                     st.metric(
#                                         "Behavioral",
#                                         feedback.get("behavioral", {}).get(
#                                             "score", "N/A"
#                                         ),
#                                         "out of 5",
#                                     )
#                                     st.caption(
#                                         feedback.get("behavioral", {}).get(
#                                             "feedback", ""
#                                         )
#                                     )
#                                 st.write("**Improvement Tips:**")
#                                 st.info(feedback.get("improvement_tips", ""))
#                                 resource_links = feedback.get("resource_links", [])
#                                 if resource_links:
#                                     st.write("**Curated Resources:**")
#                                     for link in resource_links:
#                                         st.markdown(f"- [{link}]({link})")
#                             except Exception as e:
#                                 st.error(f"Feedback generation failed: {str(e)}")
#             else:
#                 st.error("Failed to generate transcript. Please try again.")


# if __name__ == "__main__":
#     main()


# import os
# import streamlit as st
# import json
# import tempfile
# import warnings
# from pypdf import PdfReader
# import numpy as np
# import faiss
# import pickle
# import re
# import assemblyai as aai
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores.faiss import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import BaseMessage
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.tools import Tool
# from langchain import hub
# from typing import Dict, List, Optional
# import asyncio

# # Configuration
# GOOGLE_API_KEY = "AIzaSyBUgD1d0p8SNP73gbGTMb5mGCZajuHQ4Mo"  # Replace with your actual API key
# ASSEMBLYAI_API_KEY = "c4d1580904294010b36eff07a1dbc992"  # Replace with your actual API key

# # Set environment variables
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# aai.settings.api_key = ASSEMBLYAI_API_KEY

# # Suppress warnings
# warnings.filterwarnings("ignore")


# class AudioTranscriptionAgent:
#     """Agent for handling audio transcription using AssemblyAI"""
    
#     def __init__(self, api_key: str):
#         aai.settings.api_key = api_key
        
#     def transcribe_with_speakers(self, audio_path: str) -> Dict:
#         """Transcribe audio with speaker diarization using AssemblyAI"""
#         try:
#             config = aai.TranscriptionConfig(
#                 speaker_labels=True,
#                 speakers_expected=2,  # Interviewer and Candidate
#                 auto_highlights=True,
#                 sentiment_analysis=True,
#                 entity_detection=True
#             )
            
#             transcriber = aai.Transcriber()
#             transcript = transcriber.transcribe(audio_path, config)
            
#             if transcript.status == aai.TranscriptStatus.error:
#                 raise Exception(f"Transcription failed: {transcript.error}")
            
#             # Convert to structured format
#             segments = []
#             for utterance in transcript.utterances:
#                 # Map speaker labels to roles
#                 speaker = "Interviewer" if utterance.speaker == "A" else "Candidate"
#                 segments.append({
#                     "start_time": self._ms_to_timestamp(utterance.start),
#                     "end_time": self._ms_to_timestamp(utterance.end),
#                     "speaker": speaker,
#                     "text": utterance.text,
#                     "confidence": utterance.confidence
#                 })
            
#             return {
#                 "segments": segments,
#                 "summary": transcript.auto_highlights.results if transcript.auto_highlights else [],
#                 "sentiment": transcript.sentiment_analysis_results if transcript.sentiment_analysis_results else []
#             }
            
#         except Exception as e:
#             st.error(f"AssemblyAI transcription failed: {str(e)}")
#             return {"segments": [], "summary": [], "sentiment": []}
    
#     def _ms_to_timestamp(self, ms: int) -> str:
#         """Convert milliseconds to HH:MM:SS format"""
#         seconds = ms // 1000
#         hours = seconds // 3600
#         minutes = (seconds % 3600) // 60
#         seconds = seconds % 60
#         return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# class ReviewerAgent:
#     """Comprehensive reviewer agent using LangChain framework"""
    
#     def __init__(self, google_api_key: str):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             temperature=0.3,
#             google_api_key=google_api_key
#         )
#         self._setup_tools()
#         self._setup_agent()
    
#     def _setup_tools(self):
#         """Setup tools for the reviewer agent"""
        
#         def technical_evaluator(input_text: str) -> str:
#             """Evaluate technical aspects of the answer"""
#             prompt = ChatPromptTemplate.from_template("""
#             As a technical interviewer, evaluate the technical accuracy and depth of this answer:
            
#             Question: {question}
#             Answer: {answer}
#             Reference: {reference}
            
#             Provide:
#             1. Technical accuracy score (1-5)
#             2. Depth of knowledge demonstrated
#             3. Areas of strength
#             4. Technical gaps identified
#             5. Specific improvement suggestions
            
#             Format as JSON with keys: score, accuracy, depth, strengths, gaps, suggestions
#             """)
            
#             try:
#                 parts = input_text.split("|||")
#                 if len(parts) >= 3:
#                     result = (prompt | self.llm).invoke({
#                         "question": parts[0],
#                         "answer": parts[1], 
#                         "reference": parts[2]
#                     })
#                     return result.content
#                 return "Invalid input format"
#             except Exception as e:
#                 return f"Technical evaluation failed: {str(e)}"
        
#         def communication_evaluator(input_text: str) -> str:
#             """Evaluate communication skills"""
#             prompt = ChatPromptTemplate.from_template("""
#             As a communication expert, evaluate the communication effectiveness:
            
#             Answer: {answer}
            
#             Assess:
#             1. Clarity and structure (1-5)
#             2. Professional language use
#             3. Confidence level
#             4. Engagement and enthusiasm
#             5. Areas for improvement
            
#             Format as JSON with keys: clarity_score, language_quality, confidence_level, engagement, improvements
#             """)
            
#             try:
#                 result = (prompt | self.llm).invoke({"answer": input_text})
#                 return result.content
#             except Exception as e:
#                 return f"Communication evaluation failed: {str(e)}"
        
#         def behavioral_evaluator(input_text: str) -> str:
#             """Evaluate behavioral aspects"""
#             prompt = ChatPromptTemplate.from_template("""
#             As a behavioral interviewer, evaluate the behavioral indicators:
            
#             Question: {question}
#             Answer: {answer}
            
#             Look for:
#             1. Leadership qualities
#             2. Problem-solving approach
#             3. Teamwork indicators
#             4. Adaptability
#             5. Cultural fit indicators
            
#             Format as JSON with keys: leadership_score, problem_solving, teamwork, adaptability, cultural_fit, examples
#             """)
            
#             try:
#                 parts = input_text.split("|||")
#                 if len(parts) >= 2:
#                     result = (prompt | self.llm).invoke({
#                         "question": parts[0],
#                         "answer": parts[1]
#                     })
#                     return result.content
#                 return "Invalid input format"
#             except Exception as e:
#                 return f"Behavioral evaluation failed: {str(e)}"
        
#         def improvement_advisor(input_text: str) -> str:
#             """Provide specific improvement recommendations"""
#             prompt = ChatPromptTemplate.from_template("""
#             Based on the evaluation results, provide specific, actionable improvement advice:
            
#             Evaluation Data: {eval_data}
            
#             Provide:
#             1. Top 3 priority areas for improvement
#             2. Specific action steps for each area
#             3. Practice exercises or techniques
#             4. Recommended resources (books, courses, websites)
#             5. Timeline for improvement
            
#             Format as structured advice with clear action items.
#             """)
            
#             try:
#                 result = (prompt | self.llm).invoke({"eval_data": input_text})
#                 return result.content
#             except Exception as e:
#                 return f"Improvement advice generation failed: {str(e)}"
        
#         self.tools = [
#             Tool(
#                 name="technical_evaluator",
#                 description="Evaluates technical aspects of interview answers. Input format: question|||answer|||reference",
#                 func=technical_evaluator
#             ),
#             Tool(
#                 name="communication_evaluator", 
#                 description="Evaluates communication effectiveness of answers. Input: answer text",
#                 func=communication_evaluator
#             ),
#             Tool(
#                 name="behavioral_evaluator",
#                 description="Evaluates behavioral indicators. Input format: question|||answer",
#                 func=behavioral_evaluator
#             ),
#             Tool(
#                 name="improvement_advisor",
#                 description="Provides improvement recommendations based on evaluation results",
#                 func=improvement_advisor
#             )
#         ]
    
#     def _setup_agent(self):
#         """Setup the ReAct agent"""
#         try:
#             # Get the ReAct prompt from hub
#             prompt = hub.pull("hwchase17/react")
            
#             # Create the agent
#             agent = create_react_agent(self.llm, self.tools, prompt)
#             self.agent_executor = AgentExecutor(
#                 agent=agent,
#                 tools=self.tools,
#                 verbose=True,
#                 handle_parsing_errors=True,
#                 max_iterations=3
#             )
#         except Exception as e:
#             st.warning(f"Agent setup failed, falling back to direct tool usage: {str(e)}")
#             self.agent_executor = None
    
#     def comprehensive_review(self, question: str, answer: str, reference: str = "") -> Dict:
#         """Conduct comprehensive review using agent framework"""
#         if self.agent_executor:
#             try:
#                 # Use agent for comprehensive evaluation
#                 agent_input = f"""
#                 Conduct a comprehensive interview evaluation for:
#                 Question: {question}
#                 Candidate Answer: {answer}
#                 Reference Answer: {reference}
                
#                 Use the available tools to:
#                 1. Evaluate technical aspects
#                 2. Assess communication skills
#                 3. Analyze behavioral indicators
#                 4. Provide improvement recommendations
                
#                 Compile a final comprehensive report.
#                 """
                
#                 result = self.agent_executor.invoke({"input": agent_input})
#                 return {"comprehensive_review": result["output"]}
                
#             except Exception as e:
#                 st.warning(f"Agent execution failed, using direct evaluation: {str(e)}")
        
#         # Fallback to direct tool usage
#         return self._direct_evaluation(question, answer, reference)
    
#     def _direct_evaluation(self, question: str, answer: str, reference: str) -> Dict:
#         """Direct evaluation using tools without agent framework"""
#         results = {}
        
#         try:
#             # Technical evaluation
#             tech_input = f"{question}|||{answer}|||{reference}"
#             results["technical"] = self.tools[0].func(tech_input)
            
#             # Communication evaluation
#             results["communication"] = self.tools[1].func(answer)
            
#             # Behavioral evaluation
#             behav_input = f"{question}|||{answer}"
#             results["behavioral"] = self.tools[2].func(behav_input)
            
#             # Improvement advice
#             eval_summary = f"Technical: {results['technical']}\nCommunication: {results['communication']}\nBehavioral: {results['behavioral']}"
#             results["improvement_advice"] = self.tools[3].func(eval_summary)
            
#         except Exception as e:
#             st.error(f"Direct evaluation failed: {str(e)}")
#             results["error"] = str(e)
        
#         return results


# class RAGManager:
#     """Enhanced RAG system using LangChain"""
    
#     def __init__(self, google_api_key: str):
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=google_api_key
#         )
#         self.vectorstore = None
#         self.documents = []
    
#     def build_knowledge_base(self, pdf_file) -> bool:
#         """Build knowledge base from PDF using LangChain"""
#         try:
#             # Save PDF temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#                 tmp.write(pdf_file.getvalue())
#                 tmp_path = tmp.name
            
#             # Load and process PDF
#             loader = PyPDFLoader(tmp_path)
#             pages = loader.load()
            
#             # Split into chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 separators=["\n\n", "\n", ". ", " ", ""]
#             )
            
#             self.documents = text_splitter.split_documents(pages)
            
#             if self.documents:
#                 # Create vector store
#                 self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
#                 return True
            
#             return False
            
#         except Exception as e:
#             st.error(f"Knowledge base creation failed: {str(e)}")
#             return False
    
#     def retrieve_relevant_context(self, question: str, k: int = 3) -> List[str]:
#         """Retrieve relevant context for a question"""
#         if not self.vectorstore:
#             return []
        
#         try:
#             docs = self.vectorstore.similarity_search(question, k=k)
#             return [doc.page_content for doc in docs]
#         except Exception as e:
#             st.warning(f"Context retrieval failed: {str(e)}")
#             return []


# def clean_transcript_segments(segments: List[Dict], llm) -> List[Dict]:
#     """Clean transcript segments using LangChain"""
#     prompt = ChatPromptTemplate.from_template("""
#     Clean the following interview transcript segments by:
#     1. Removing filler words and irrelevant content
#     2. Keeping only meaningful questions and substantive answers
#     3. Maintaining the JSON structure
    
#     Segments: {segments}
    
#     Return only the cleaned segments in JSON format.
#     """)
    
#     try:
#         result = (prompt | llm).invoke({"segments": json.dumps(segments)})
#         cleaned_text = result.content.strip()
        
#         # Remove code block formatting if present
#         if cleaned_text.startswith("```json"):
#             cleaned_text = cleaned_text[7:-3].strip()
#         elif cleaned_text.startswith("```"):
#             cleaned_text = cleaned_text[3:-3].strip()
        
#         cleaned_segments = json.loads(cleaned_text)
#         return cleaned_segments if isinstance(cleaned_segments, list) else segments
        
#     except Exception as e:
#         st.warning(f"Transcript cleaning failed: {str(e)}")
#         return segments


# def extract_qa_pairs(segments: List[Dict]) -> List[Dict]:
#     """Extract Q&A pairs from transcript segments"""
#     qa_pairs = []
#     current_question = None
#     current_response = []
    
#     for segment in segments:
#         if segment["speaker"] == "Interviewer":
#             # Save previous Q&A pair if exists
#             if current_question and current_response:
#                 qa_pairs.append({
#                     "question": current_question,
#                     "response": " ".join(current_response).strip(),
#                     "confidence": segment.get("confidence", 0.0)
#                 })
            
#             current_question = segment["text"]
#             current_response = []
#         else:
#             if current_question:  # Only collect responses if we have a question
#                 current_response.append(segment["text"])
    
#     # Add final Q&A pair
#     if current_question and current_response:
#         qa_pairs.append({
#             "question": current_question,
#             "response": " ".join(current_response).strip(),
#             "confidence": segments[-1].get("confidence", 0.0)
#         })
    
#     return qa_pairs


# def main():
#     st.set_page_config(
#         page_title="AI Mock Interview Feedback System", 
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     st.title("🎤 AI Mock Interview Feedback System")
#     st.markdown("""
#     **Enhanced Features:**
#     - 🎯 AssemblyAI for accurate audio transcription with speaker diarization
#     - 🤖 LangChain-based agent framework for comprehensive evaluation
#     - 📚 Advanced RAG system for contextual feedback
#     - 📊 Multi-dimensional assessment (Technical, Communication, Behavioral)
#     """)
    
#     # Initialize components
#     if "transcription_agent" not in st.session_state:
#         st.session_state.transcription_agent = AudioTranscriptionAgent(ASSEMBLYAI_API_KEY)
    
#     if "reviewer_agent" not in st.session_state:
#         st.session_state.reviewer_agent = ReviewerAgent(GOOGLE_API_KEY)
    
#     if "rag_manager" not in st.session_state:
#         st.session_state.rag_manager = RAGManager(GOOGLE_API_KEY)
    
#     # Sidebar configuration
#     with st.sidebar:
#         st.header("📋 Configuration")
        
#         # PDF upload
#         pdf_file = st.file_uploader("Upload Reference Q&A PDF", type=["pdf"])
        
#         if pdf_file:
#             if st.button("🔨 Build Knowledge Base"):
#                 with st.spinner("Building knowledge base..."):
#                     success = st.session_state.rag_manager.build_knowledge_base(pdf_file)
#                     if success:
#                         st.success("✅ Knowledge base built successfully!")
#                     else:
#                         st.error("❌ Failed to build knowledge base")
        
#         st.divider()
        
#         # Settings
#         st.subheader("⚙️ Settings")
#         evaluation_depth = st.selectbox(
#             "Evaluation Depth",
#             ["Quick", "Comprehensive", "Detailed"],
#             index=1
#         )
        
#         include_sentiment = st.checkbox("Include Sentiment Analysis", True)
#         include_highlights = st.checkbox("Include Key Highlights", True)
    
#     # Main content area
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # Audio upload
#         uploaded_file = st.file_uploader(
#             "🎵 Upload Interview Recording", 
#             type=["mp3", "wav", "m4a", "flac"]
#         )
        
#         if uploaded_file and st.session_state.rag_manager.vectorstore:
#             process_interview(
#                 uploaded_file, 
#                 st.session_state.transcription_agent,
#                 st.session_state.reviewer_agent,
#                 st.session_state.rag_manager,
#                 evaluation_depth,
#                 include_sentiment,
#                 include_highlights
#             )
#         elif uploaded_file and not st.session_state.rag_manager.vectorstore:
#             st.warning("⚠️ Please upload and build knowledge base from PDF first!")
    
#     with col2:
#         st.subheader("📊 System Status")
        
#         # Status indicators
#         kb_status = "✅ Ready" if st.session_state.rag_manager.vectorstore else "❌ Not Built"
#         st.metric("Knowledge Base", kb_status)
        
#         transcription_status = "✅ Ready" if st.session_state.transcription_agent else "❌ Not Ready"
#         st.metric("Transcription Service", transcription_status)
        
#         reviewer_status = "✅ Ready" if st.session_state.reviewer_agent else "❌ Not Ready"
#         st.metric("Reviewer Agent", reviewer_status)


# def process_interview(audio_file, transcription_agent, reviewer_agent, rag_manager, 
#                      evaluation_depth, include_sentiment, include_highlights):
#     """Process the interview recording"""
    
#     with st.spinner("🎯 Processing interview recording..."):
#         # Save audio file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(audio_file.getvalue())
#             audio_path = tmp.name
        
#         # Step 1: Transcription with AssemblyAI
#         st.subheader("📝 Transcription Results")
        
#         transcript_data = transcription_agent.transcribe_with_speakers(audio_path)
#         print(transcript_data, 1069)
        
#         if transcript_data["segments"]:
#             # Display transcription
#             with st.expander("View Full Transcript", expanded=False):
#                 for i, segment in enumerate(transcript_data["segments"]):
#                     speaker_emoji = "👨‍💼" if segment["speaker"] == "Interviewer" else "👤"
#                     st.write(f"{speaker_emoji} **{segment['speaker']}** ({segment['start_time']}): {segment['text']}")
            
#             # Show highlights if available
#             if include_highlights and transcript_data.get("summary"):
#                 st.subheader("🎯 Key Highlights")
#                 for highlight in transcript_data["summary"][:5]:
#                     st.info(f"💡 {highlight.text}")
            
#             # Clean transcript
#             llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
#             cleaned_segments = clean_transcript_segments(transcript_data["segments"], llm)
            
#             # Extract Q&A pairs
#             qa_pairs = extract_qa_pairs(cleaned_segments)
            
#             if qa_pairs:
#                 st.subheader("🔍 Interview Analysis")
#                 st.write(f"Found {len(qa_pairs)} question-answer pairs")
                
#                 # Process each Q&A pair
#                 for i, pair in enumerate(qa_pairs):
#                     with st.expander(f"Question {i+1}: {pair['question'][:80]}...", expanded=True):
#                         st.write("**Question:**")
#                         st.write(pair["question"])
                        
#                         st.write("**Candidate Response:**")
#                         st.write(pair["response"])
                        
#                         # Retrieve relevant context using RAG
#                         relevant_contexts = rag_manager.retrieve_relevant_context(pair["question"])
#                         reference_answer = relevant_contexts[0] if relevant_contexts else ""
                        
#                         if reference_answer:
#                             st.write("**Reference Context:**")
#                             st.info(reference_answer[:500] + "..." if len(reference_answer) > 500 else reference_answer)
                        
#                         # Get comprehensive review from agent
#                         with st.spinner("Generating comprehensive feedback..."):
#                             review_results = reviewer_agent.comprehensive_review(
#                                 pair["question"], 
#                                 pair["response"], 
#                                 reference_answer
#                             )
                        
#                         # Display results
#                         if "comprehensive_review" in review_results:
#                             st.write("**🤖 Agent Review:**")
#                             st.write(review_results["comprehensive_review"])
#                         else:
#                             # Display individual evaluations
#                             col1, col2, col3 = st.columns(3)
                            
#                             with col1:
#                                 st.write("**🔧 Technical Evaluation**")
#                                 st.text_area("Technical Feedback", review_results.get("technical", "N/A"), height=100)
                            
#                             with col2:
#                                 st.write("**💬 Communication Assessment**") 
#                                 st.text_area("Communication Feedback", review_results.get("communication", "N/A"), height=100)
                            
#                             with col3:
#                                 st.write("**🎭 Behavioral Analysis**")
#                                 st.text_area("Behavioral Feedback", review_results.get("behavioral", "N/A"), height=100)
                            
#                             if "improvement_advice" in review_results:
#                                 st.write("**📈 Improvement Recommendations**")
#                                 st.success(review_results["improvement_advice"])
                        
#                         st.divider()
#             else:
#                 st.warning("No clear question-answer pairs found in the transcript.")
#         else:
#             st.error("Transcription failed. Please check your audio file and try again.")


# if __name__ == "__main__":
#     main()



# import os
# import streamlit as st
# import json
# import tempfile
# import warnings
# from pypdf import PdfReader
# import numpy as np
# import faiss
# import pickle
# import re
# import assemblyai as aai
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores.faiss import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import BaseMessage
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.tools import Tool
# from langchain import hub
# from typing import Dict, List, Optional
# import asyncio

# # Configuration
# GOOGLE_API_KEY = "AIzaSyBUgD1d0p8SNP73gbGTMb5mGCZajuHQ4Mo"  # Replace with your actual API key
# ASSEMBLYAI_API_KEY = "c4d1580904294010b36eff07a1dbc992"  # Replace with your actual API key

# # Set environment variables
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# aai.settings.api_key = ASSEMBLYAI_API_KEY

# # Suppress warnings
# warnings.filterwarnings("ignore")


# class AudioTranscriptionAgent:
#     """Agent for handling audio transcription using AssemblyAI"""
    
#     def __init__(self, api_key: str):
#         aai.settings.api_key = api_key
        
#     def transcribe_with_speakers(self, audio_path: str) -> Dict:
#         """Transcribe audio with speaker diarization using AssemblyAI"""
#         try:
#             st.info("🎵 Starting transcription with AssemblyAI...")
            
#             config = aai.TranscriptionConfig(
#                 speaker_labels=True,
#                 speakers_expected=2,  # Interviewer and Candidate
#                 auto_highlights=True,
#                 sentiment_analysis=True,
#                 entity_detection=True,
#                 punctuate=True,
#                 format_text=True
#             )
            
#             transcriber = aai.Transcriber()
            
#             # Upload and transcribe
#             st.info("📤 Uploading audio file...")
#             transcript = transcriber.transcribe(audio_path, config)
            
#             # Wait for completion with progress
#             st.info("⏳ Processing transcription...")
#             while transcript.status not in [aai.TranscriptStatus.completed, aai.TranscriptStatus.error]:
#                 transcript = transcriber.get_transcript(transcript.id)
#                 st.write(f"Status: {transcript.status}")
                
#             if transcript.status == aai.TranscriptStatus.error:
#                 raise Exception(f"Transcription failed: {transcript.error}")
            
#             st.success("✅ Transcription completed!")
            
#             # Convert to structured format
#             segments = []
            
#             # Check if we have speaker labels (utterances)
#             if hasattr(transcript, 'utterances') and transcript.utterances:
#                 for utterance in transcript.utterances:
#                     # Map speaker labels to roles (you may need to adjust this logic)
#                     speaker = "Interviewer" if utterance.speaker == "A" else "Candidate"
#                     segments.append({
#                         "start_time": self._ms_to_timestamp(utterance.start),
#                         "end_time": self._ms_to_timestamp(utterance.end),
#                         "speaker": speaker,
#                         "text": utterance.text,
#                         "confidence": getattr(utterance, 'confidence', 0.9)
#                     })
#             else:
#                 # Fallback: use the full transcript without speaker separation
#                 st.warning("Speaker diarization not available, using full transcript")
#                 segments.append({
#                     "start_time": "00:00:00",
#                     "end_time": self._ms_to_timestamp(getattr(transcript, 'audio_duration', 0) * 1000),
#                     "speaker": "Unknown",
#                     "text": transcript.text,
#                     "confidence": getattr(transcript, 'confidence', 0.9)
#                 })
            
#             # Get additional features safely
#             summary = []
#             if hasattr(transcript, 'auto_highlights') and transcript.auto_highlights:
#                 summary = [highlight.text for highlight in transcript.auto_highlights.results]
            
#             sentiment = []
#             if hasattr(transcript, 'sentiment_analysis_results') and transcript.sentiment_analysis_results:
#                 sentiment = transcript.sentiment_analysis_results
            
#             return {
#                 "segments": segments,
#                 "summary": summary,
#                 "sentiment": sentiment,
#                 "full_text": transcript.text
#             }
            
#         except Exception as e:
#             error_msg = f"AssemblyAI transcription failed: {str(e)}"
#             st.error(error_msg)
#             print(f"DEBUG: {error_msg}")
            
#             # Try a simpler transcription as fallback
#             return self._fallback_transcription(audio_path)
    
#     def _fallback_transcription(self, audio_path: str) -> Dict:
#         """Fallback transcription without advanced features"""
#         try:
#             st.warning("🔄 Trying simplified transcription...")
            
#             config = aai.TranscriptionConfig(
#                 speaker_labels=False,  # Disable speaker labels
#                 punctuate=True,
#                 format_text=True
#             )
            
#             transcriber = aai.Transcriber()
#             transcript = transcriber.transcribe(audio_path, config)
            
#             # Wait for completion
#             while transcript.status not in [aai.TranscriptStatus.completed, aai.TranscriptStatus.error]:
#                 transcript = transcriber.get_transcript(transcript.id)
                
#             if transcript.status == aai.TranscriptStatus.error:
#                 raise Exception(f"Fallback transcription failed: {transcript.error}")
            
#             # Return simple format
#             segments = [{
#                 "start_time": "00:00:00",
#                 "end_time": "00:00:00",
#                 "speaker": "Speaker",
#                 "text": transcript.text,
#                 "confidence": 0.9
#             }]
            
#             st.success("✅ Fallback transcription completed!")
            
#             return {
#                 "segments": segments,
#                 "summary": [],
#                 "sentiment": [],
#                 "full_text": transcript.text
#             }
            
#         except Exception as e:
#             st.error(f"All transcription methods failed: {str(e)}")
#             return {"segments": [], "summary": [], "sentiment": [], "full_text": ""}
    
#     def _ms_to_timestamp(self, ms: int) -> str:
#         """Convert milliseconds to HH:MM:SS format"""
#         if ms == 0:
#             return "00:00:00"
#         seconds = ms // 1000
#         hours = seconds // 3600
#         minutes = (seconds % 3600) // 60
#         seconds = seconds % 60
#         return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# class ReviewerAgent:
#     """Comprehensive reviewer agent using LangChain framework"""
    
#     def __init__(self, google_api_key: str):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             temperature=0.3,
#             google_api_key=google_api_key
#         )
#         self._setup_tools()
#         self._setup_agent()
    
#     def _setup_tools(self):
#         """Setup tools for the reviewer agent"""
        
#         def technical_evaluator(input_text: str) -> str:
#             """Evaluate technical aspects of the answer"""
#             prompt = ChatPromptTemplate.from_template("""
#             As a technical interviewer, evaluate the technical accuracy and depth of this answer:
            
#             Question: {question}
#             Answer: {answer}
#             Reference: {reference}
            
#             Provide:
#             1. Technical accuracy score (1-5)
#             2. Depth of knowledge demonstrated
#             3. Areas of strength
#             4. Technical gaps identified
#             5. Specific improvement suggestions
            
#             Format as JSON with keys: score, accuracy, depth, strengths, gaps, suggestions
#             """)
            
#             try:
#                 parts = input_text.split("|||")
#                 if len(parts) >= 3:
#                     result = (prompt | self.llm).invoke({
#                         "question": parts[0],
#                         "answer": parts[1], 
#                         "reference": parts[2]
#                     })
#                     return result.content
#                 return "Invalid input format"
#             except Exception as e:
#                 return f"Technical evaluation failed: {str(e)}"
        
#         def communication_evaluator(input_text: str) -> str:
#             """Evaluate communication skills"""
#             prompt = ChatPromptTemplate.from_template("""
#             As a communication expert, evaluate the communication effectiveness:
            
#             Answer: {answer}
            
#             Assess:
#             1. Clarity and structure (1-5)
#             2. Professional language use
#             3. Confidence level
#             4. Engagement and enthusiasm
#             5. Areas for improvement
            
#             Format as JSON with keys: clarity_score, language_quality, confidence_level, engagement, improvements
#             """)
            
#             try:
#                 result = (prompt | self.llm).invoke({"answer": input_text})
#                 return result.content
#             except Exception as e:
#                 return f"Communication evaluation failed: {str(e)}"
        
#         def behavioral_evaluator(input_text: str) -> str:
#             """Evaluate behavioral aspects"""
#             prompt = ChatPromptTemplate.from_template("""
#             As a behavioral interviewer, evaluate the behavioral indicators:
            
#             Question: {question}
#             Answer: {answer}
            
#             Look for:
#             1. Leadership qualities
#             2. Problem-solving approach
#             3. Teamwork indicators
#             4. Adaptability
#             5. Cultural fit indicators
            
#             Format as JSON with keys: leadership_score, problem_solving, teamwork, adaptability, cultural_fit, examples
#             """)
            
#             try:
#                 parts = input_text.split("|||")
#                 if len(parts) >= 2:
#                     result = (prompt | self.llm).invoke({
#                         "question": parts[0],
#                         "answer": parts[1]
#                     })
#                     return result.content
#                 return "Invalid input format"
#             except Exception as e:
#                 return f"Behavioral evaluation failed: {str(e)}"
        
#         def improvement_advisor(input_text: str) -> str:
#             """Provide specific improvement recommendations"""
#             prompt = ChatPromptTemplate.from_template("""
#             Based on the evaluation results, provide specific, actionable improvement advice:
            
#             Evaluation Data: {eval_data}
            
#             Provide:
#             1. Top 3 priority areas for improvement
#             2. Specific action steps for each area
#             3. Practice exercises or techniques
#             4. Recommended resources (books, courses, websites)
#             5. Timeline for improvement
            
#             Format as structured advice with clear action items.
#             """)
            
#             try:
#                 result = (prompt | self.llm).invoke({"eval_data": input_text})
#                 return result.content
#             except Exception as e:
#                 return f"Improvement advice generation failed: {str(e)}"
        
#         self.tools = [
#             Tool(
#                 name="technical_evaluator",
#                 description="Evaluates technical aspects of interview answers. Input format: question|||answer|||reference",
#                 func=technical_evaluator
#             ),
#             Tool(
#                 name="communication_evaluator", 
#                 description="Evaluates communication effectiveness of answers. Input: answer text",
#                 func=communication_evaluator
#             ),
#             Tool(
#                 name="behavioral_evaluator",
#                 description="Evaluates behavioral indicators. Input format: question|||answer",
#                 func=behavioral_evaluator
#             ),
#             Tool(
#                 name="improvement_advisor",
#                 description="Provides improvement recommendations based on evaluation results",
#                 func=improvement_advisor
#             )
#         ]
    
#     def _setup_agent(self):
#         """Setup the ReAct agent"""
#         try:
#             # Get the ReAct prompt from hub
#             prompt = hub.pull("hwchase17/react")
            
#             # Create the agent
#             agent = create_react_agent(self.llm, self.tools, prompt)
#             self.agent_executor = AgentExecutor(
#                 agent=agent,
#                 tools=self.tools,
#                 verbose=True,
#                 handle_parsing_errors=True,
#                 max_iterations=3
#             )
#         except Exception as e:
#             st.warning(f"Agent setup failed, falling back to direct tool usage: {str(e)}")
#             self.agent_executor = None
    
#     def comprehensive_review(self, question: str, answer: str, reference: str = "") -> Dict:
#         """Conduct comprehensive review using agent framework"""
#         if self.agent_executor:
#             try:
#                 # Use agent for comprehensive evaluation
#                 agent_input = f"""
#                 Conduct a comprehensive interview evaluation for:
#                 Question: {question}
#                 Candidate Answer: {answer}
#                 Reference Answer: {reference}
                
#                 Use the available tools to:
#                 1. Evaluate technical aspects
#                 2. Assess communication skills
#                 3. Analyze behavioral indicators
#                 4. Provide improvement recommendations
                
#                 Compile a final comprehensive report.
#                 """
                
#                 result = self.agent_executor.invoke({"input": agent_input})
#                 return {"comprehensive_review": result["output"]}
                
#             except Exception as e:
#                 st.warning(f"Agent execution failed, using direct evaluation: {str(e)}")
        
#         # Fallback to direct tool usage
#         return self._direct_evaluation(question, answer, reference)
    
#     def _direct_evaluation(self, question: str, answer: str, reference: str) -> Dict:
#         """Direct evaluation using tools without agent framework"""
#         results = {}
        
#         try:
#             # Technical evaluation
#             tech_input = f"{question}|||{answer}|||{reference}"
#             results["technical"] = self.tools[0].func(tech_input)
            
#             # Communication evaluation
#             results["communication"] = self.tools[1].func(answer)
            
#             # Behavioral evaluation
#             behav_input = f"{question}|||{answer}"
#             results["behavioral"] = self.tools[2].func(behav_input)
            
#             # Improvement advice
#             eval_summary = f"Technical: {results['technical']}\nCommunication: {results['communication']}\nBehavioral: {results['behavioral']}"
#             results["improvement_advice"] = self.tools[3].func(eval_summary)
            
#         except Exception as e:
#             st.error(f"Direct evaluation failed: {str(e)}")
#             results["error"] = str(e)
        
#         return results


# class RAGManager:
#     """Enhanced RAG system using LangChain"""
    
#     def __init__(self, google_api_key: str):
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=google_api_key
#         )
#         self.vectorstore = None
#         self.documents = []
    
#     def build_knowledge_base(self, pdf_file) -> bool:
#         """Build knowledge base from PDF using LangChain"""
#         try:
#             # Save PDF temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#                 tmp.write(pdf_file.getvalue())
#                 tmp_path = tmp.name
            
#             # Load and process PDF
#             loader = PyPDFLoader(tmp_path)
#             pages = loader.load()
            
#             # Split into chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 separators=["\n\n", "\n", ". ", " ", ""]
#             )
            
#             self.documents = text_splitter.split_documents(pages)
            
#             if self.documents:
#                 # Create vector store
#                 self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
#                 return True
            
#             return False
            
#         except Exception as e:
#             st.error(f"Knowledge base creation failed: {str(e)}")
#             return False
    
#     def retrieve_relevant_context(self, question: str, k: int = 3) -> List[str]:
#         """Retrieve relevant context for a question"""
#         if not self.vectorstore:
#             return []
        
#         try:
#             docs = self.vectorstore.similarity_search(question, k=k)
#             return [doc.page_content for doc in docs]
#         except Exception as e:
#             st.warning(f"Context retrieval failed: {str(e)}")
#             return []


# def clean_transcript_segments(segments: List[Dict], llm) -> List[Dict]:
#     """Clean transcript segments using LangChain"""
#     prompt = ChatPromptTemplate.from_template("""
#     Clean the following interview transcript segments by:
#     1. Removing filler words and irrelevant content
#     2. Keeping only meaningful questions and substantive answers
#     3. Maintaining the JSON structure
    
#     Segments: {segments}
    
#     Return only the cleaned segments in JSON format.
#     """)
    
#     try:
#         result = (prompt | llm).invoke({"segments": json.dumps(segments)})
#         cleaned_text = result.content.strip()
        
#         # Remove code block formatting if present
#         if cleaned_text.startswith("```json"):
#             cleaned_text = cleaned_text[7:-3].strip()
#         elif cleaned_text.startswith("```"):
#             cleaned_text = cleaned_text[3:-3].strip()
        
#         cleaned_segments = json.loads(cleaned_text)
#         return cleaned_segments if isinstance(cleaned_segments, list) else segments
        
#     except Exception as e:
#         st.warning(f"Transcript cleaning failed: {str(e)}")
#         return segments


# def extract_qa_pairs(segments: List[Dict]) -> List[Dict]:
#     """Extract Q&A pairs from transcript segments"""
#     qa_pairs = []
#     current_question = None
#     current_response = []
    
#     for segment in segments:
#         if segment["speaker"] == "Interviewer":
#             # Save previous Q&A pair if exists
#             if current_question and current_response:
#                 qa_pairs.append({
#                     "question": current_question,
#                     "response": " ".join(current_response).strip(),
#                     "confidence": segment.get("confidence", 0.0)
#                 })
            
#             current_question = segment["text"]
#             current_response = []
#         else:
#             if current_question:  # Only collect responses if we have a question
#                 current_response.append(segment["text"])
    
#     # Add final Q&A pair
#     if current_question and current_response:
#         qa_pairs.append({
#             "question": current_question,
#             "response": " ".join(current_response).strip(),
#             "confidence": segments[-1].get("confidence", 0.0)
#         })
    
#     return qa_pairs


# def main():
#     st.set_page_config(
#         page_title="AI Mock Interview Feedback System", 
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     st.title("🎤 AI Mock Interview Feedback System")
#     st.markdown("""
#     **Enhanced Features:**
#     - 🎯 AssemblyAI for accurate audio transcription with speaker diarization
#     - 🤖 LangChain-based agent framework for comprehensive evaluation
#     - 📚 Advanced RAG system for contextual feedback
#     - 📊 Multi-dimensional assessment (Technical, Communication, Behavioral)
#     """)
    
#     # Initialize components
#     if "transcription_agent" not in st.session_state:
#         st.session_state.transcription_agent = AudioTranscriptionAgent(ASSEMBLYAI_API_KEY)
    
#     if "reviewer_agent" not in st.session_state:
#         st.session_state.reviewer_agent = ReviewerAgent(GOOGLE_API_KEY)
    
#     if "rag_manager" not in st.session_state:
#         st.session_state.rag_manager = RAGManager(GOOGLE_API_KEY)
    
#     # Sidebar configuration
#     with st.sidebar:
#         st.header("📋 Configuration")
        
#         # API Key Configuration
#         st.subheader("🔑 API Keys")
#         assemblyai_key = st.text_input("AssemblyAI API Key", type="password", 
#                                      value=ASSEMBLYAI_API_KEY if ASSEMBLYAI_API_KEY != "your_assemblyai_api_key_here" else "")
#         google_key = st.text_input("Google API Key", type="password",
#                                  value=GOOGLE_API_KEY if GOOGLE_API_KEY != "your_google_api_key_here" else "")
        
#         if assemblyai_key and google_key:
#             # Update API keys
#             if assemblyai_key != ASSEMBLYAI_API_KEY:
#                 aai.settings.api_key = assemblyai_key
#                 st.session_state.transcription_agent = AudioTranscriptionAgent(assemblyai_key)
            
#             if google_key != GOOGLE_API_KEY:
#                 os.environ["GOOGLE_API_KEY"] = google_key
#                 st.session_state.reviewer_agent = ReviewerAgent(google_key)
#                 st.session_state.rag_manager = RAGManager(google_key)
        
#         st.divider()
        
#         # PDF upload
#         pdf_file = st.file_uploader("Upload Reference Q&A PDF", type=["pdf"])
        
#         if pdf_file:
#             if st.button("🔨 Build Knowledge Base"):
#                 with st.spinner("Building knowledge base..."):
#                     success = st.session_state.rag_manager.build_knowledge_base(pdf_file)
#                     if success:
#                         st.success("✅ Knowledge base built successfully!")
#                     else:
#                         st.error("❌ Failed to build knowledge base")
        
#         st.divider()
        
#         # Settings
#         st.subheader("⚙️ Settings")
#         evaluation_depth = st.selectbox(
#             "Evaluation Depth",
#             ["Quick", "Comprehensive", "Detailed"],
#             index=1
#         )
        
#         include_sentiment = st.checkbox("Include Sentiment Analysis", True)
#         include_highlights = st.checkbox("Include Key Highlights", True)
        
#         # Debug mode
#         debug_mode = st.checkbox("Debug Mode", False)
#         if debug_mode:
#             st.subheader("🐛 Debug Info")
#             st.write(f"AssemblyAI Key Set: {'✅' if assemblyai_key else '❌'}")
#             st.write(f"Google Key Set: {'✅' if google_key else '❌'}")
#             st.write(f"Audio Processing: {'✅' if 'transcription_agent' in st.session_state else '❌'}")
    
#     # Main content area
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # Audio upload
#         uploaded_file = st.file_uploader(
#             "🎵 Upload Interview Recording", 
#             type=["mp3", "wav", "m4a", "flac"]
#         )
        
#         if uploaded_file and st.session_state.rag_manager.vectorstore:
#             process_interview(
#                 uploaded_file, 
#                 st.session_state.transcription_agent,
#                 st.session_state.reviewer_agent,
#                 st.session_state.rag_manager,
#                 evaluation_depth,
#                 include_sentiment,
#                 include_highlights
#             )
#         elif uploaded_file and not st.session_state.rag_manager.vectorstore:
#             st.warning("⚠️ Please upload and build knowledge base from PDF first!")
    
#     with col2:
#         st.subheader("📊 System Status")
        
#         # Status indicators
#         kb_status = "✅ Ready" if st.session_state.rag_manager.vectorstore else "❌ Not Built"
#         st.metric("Knowledge Base", kb_status)
        
#         transcription_status = "✅ Ready" if st.session_state.transcription_agent else "❌ Not Ready"
#         st.metric("Transcription Service", transcription_status)
        
#         reviewer_status = "✅ Ready" if st.session_state.reviewer_agent else "❌ Not Ready"
#         st.metric("Reviewer Agent", reviewer_status)


# def test_transcription_agent(api_key: str):
#     """Test function to verify AssemblyAI connection"""
#     try:
#         aai.settings.api_key = api_key
#         # Test with a simple API call
#         transcriber = aai.Transcriber()
#         st.success("✅ AssemblyAI connection verified!")
#         return True
#     except Exception as e:
#         st.error(f"❌ AssemblyAI connection failed: {str(e)}")
#         return False


# def process_interview(audio_file, transcription_agent, reviewer_agent, rag_manager, 
#                      evaluation_depth, include_sentiment, include_highlights):
#     """Process the interview recording"""
    
#     with st.spinner("🎯 Processing interview recording..."):
#         # Save audio file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(audio_file.getvalue())
#             audio_path = tmp.name
        
#         # Debug: Show file info
#         st.info(f"📁 Audio file saved: {audio_path}")
#         st.info(f"📊 File size: {len(audio_file.getvalue())} bytes")
        
#         # Step 1: Transcription with AssemblyAI
#         st.subheader("📝 Transcription Results")
        
#         transcript_data = transcription_agent.transcribe_with_speakers(audio_path)
#         print(f"DEBUG: Transcript data: {transcript_data}")
        
#         if transcript_data["segments"]:
#             # Display transcription
#             with st.expander("View Full Transcript", expanded=True):
#                 for i, segment in enumerate(transcript_data["segments"]):
#                     speaker_emoji = "👨‍💼" if segment["speaker"] == "Interviewer" else "👤"
#                     st.write(f"{speaker_emoji} **{segment['speaker']}** ({segment['start_time']}): {segment['text']}")
            
#             # Show full text if available
#             if transcript_data.get("full_text"):
#                 st.subheader("📄 Full Transcript Text")
#                 st.text_area("Full Text", transcript_data["full_text"], height=200)
            
#             # Show highlights if available
#             if include_highlights and transcript_data.get("summary"):
#                 st.subheader("🎯 Key Highlights")
#                 for highlight in transcript_data["summary"][:5]:
#                     st.info(f"💡 {highlight}")
            
#             # Clean transcript
#             llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
#             cleaned_segments = clean_transcript_segments(transcript_data["segments"], llm)
            
#             # Extract Q&A pairs
#             qa_pairs = extract_qa_pairs(cleaned_segments)
            
#             if qa_pairs:
#                 st.subheader("🔍 Interview Analysis")
#                 st.write(f"Found {len(qa_pairs)} question-answer pairs")
                
#                 # Process each Q&A pair
#                 for i, pair in enumerate(qa_pairs):
#                     with st.expander(f"Question {i+1}: {pair['question'][:80]}...", expanded=True):
#                         st.write("**Question:**")
#                         st.write(pair["question"])
                        
#                         st.write("**Candidate Response:**")
#                         st.write(pair["response"])
                        
#                         # Retrieve relevant context using RAG
#                         relevant_contexts = rag_manager.retrieve_relevant_context(pair["question"])
#                         reference_answer = relevant_contexts[0] if relevant_contexts else ""
                        
#                         if reference_answer:
#                             st.write("**Reference Context:**")
#                             st.info(reference_answer[:500] + "..." if len(reference_answer) > 500 else reference_answer)
                        
#                         # Get comprehensive review from agent
#                         with st.spinner("Generating comprehensive feedback..."):
#                             review_results = reviewer_agent.comprehensive_review(
#                                 pair["question"], 
#                                 pair["response"], 
#                                 reference_answer
#                             )
                        
#                         # Display results
#                         if "comprehensive_review" in review_results:
#                             st.write("**🤖 Agent Review:**")
#                             st.write(review_results["comprehensive_review"])
#                         else:
#                             # Display individual evaluations
#                             col1, col2, col3 = st.columns(3)
                            
#                             with col1:
#                                 st.write("**🔧 Technical Evaluation**")
#                                 st.text_area("Technical Feedback", review_results.get("technical", "N/A"), height=100, key=f"tech_{i}")
                            
#                             with col2:
#                                 st.write("**💬 Communication Assessment**") 
#                                 st.text_area("Communication Feedback", review_results.get("communication", "N/A"), height=100, key=f"comm_{i}")
                            
#                             with col3:
#                                 st.write("**🎭 Behavioral Analysis**")
#                                 st.text_area("Behavioral Feedback", review_results.get("behavioral", "N/A"), height=100, key=f"behav_{i}")
                            
#                             if "improvement_advice" in review_results:
#                                 st.write("**📈 Improvement Recommendations**")
#                                 st.success(review_results["improvement_advice"])
                        
#                         st.divider()
#             else:
#                 st.warning("No clear question-answer pairs found in the transcript.")
                
#                 # Show the full transcript as fallback
#                 if transcript_data.get("full_text"):
#                     st.subheader("💬 Full Conversation")
#                     st.text(transcript_data["full_text"])
                    
#                     # Try to get basic feedback on the full text
#                     st.subheader("📊 General Feedback")
#                     with st.spinner("Analyzing full conversation..."):
#                         review_results = reviewer_agent.comprehensive_review(
#                             "General interview conversation", 
#                             transcript_data["full_text"], 
#                             ""
#                         )
                    
#                     if review_results:
#                         st.write("**🤖 General Analysis:**")
#                         st.write(review_results.get("comprehensive_review", "Analysis completed"))
#         else:
#             st.error("Transcription failed. Please check your audio file and API key.")
            
#             # Show debug info
#             st.write("**Debug Information:**")
#             st.write(f"- Audio file path: {audio_path}")
#             st.write(f"- Audio file size: {len(audio_file.getvalue())} bytes")
#             st.write(f"- Transcript data: {transcript_data}")
            
#             # Clean up temp file
#             try:
#                 os.unlink(audio_path)
#             except:
#                 pass


# if __name__ == "__main__":
#     main()


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
from typing import Dict, List, Optional
import asyncio

# Configuration
GOOGLE_API_KEY = "AIzaSyBUgD1d0p8SNP73gbGTMb5mGCZajuHQ4Mo"  # Replace with your actual API key
ASSEMBLYAI_API_KEY = "c4d1580904294010b36eff07a1dbc992"  # Replace with your actual API key

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Suppress warnings
warnings.filterwarnings("ignore")


class AudioTranscriptionAgent:
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
            
            # Safely get highlights
            highlights = []
            try:
                if hasattr(transcript, 'auto_highlights') and transcript.auto_highlights:
                    highlights = [h.text for h in transcript.auto_highlights.results]
            except AttributeError:
                pass
            
            return {
                "segments": segments,
                "summary": highlights,
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


class ReviewerAgent:
    """Comprehensive reviewer agent using LangChain framework"""
    
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=google_api_key
        )
        self._setup_tools()
        self._setup_agent()
    
    def _setup_tools(self):
        """Setup tools for the reviewer agent"""
        
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
        
        self.tools = [
            Tool(
                name="technical_evaluator",
                description="Evaluates technical aspects of interview answers. Input format: question|||answer|||reference",
                func=technical_evaluator
            ),
            Tool(
                name="communication_evaluator", 
                description="Evaluates communication effectiveness of answers. Input: answer text",
                func=communication_evaluator
            ),
            Tool(
                name="behavioral_evaluator",
                description="Evaluates behavioral indicators. Input format: question|||answer",
                func=behavioral_evaluator
            ),
            Tool(
                name="improvement_advisor",
                description="Provides improvement recommendations based on evaluation results",
                func=improvement_advisor
            ),
            Tool(
                name="grammar_corrector",
                description="Corrects grammar and improves phrasing. Input: response text",
                func=grammar_corrector
            ),
            Tool(
                name="answer_rater",
                description="Rates answer against reference. Input format: question|||answer|||reference",
                func=answer_rater
            )
        ]
    
    def _setup_agent(self):
        """Setup the ReAct agent"""
        try:
            # Get the ReAct prompt from hub
            prompt = hub.pull("hwchase17/react")
            
            # Create the agent
            agent = create_react_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=6  # Increased for more complex evaluations
            )
        except Exception as e:
            st.warning(f"Agent setup failed, falling back to direct tool usage: {str(e)}")
            self.agent_executor = None
    
    def comprehensive_review(self, question: str, answer: str, reference: str = "") -> Dict:
        """Conduct comprehensive review using agent framework"""
        if self.agent_executor:
            try:
                # Use agent for comprehensive evaluation
                agent_input = f"""
                Conduct a comprehensive interview evaluation for ONLY this specific question:
                Question: {question}
                Candidate Answer: {answer}
                Reference Answer: {reference}
                
                Use the available tools to:
                1. Evaluate technical aspects
                2. Assess communication skills
                3. Analyze behavioral indicators
                4. Rate the answer against the reference
                5. Provide improvement recommendations
                
                Compile a final comprehensive report.
                """
                
                # Add delay to prevent rate limiting
                time.sleep(2)
                result = self.agent_executor.invoke({"input": agent_input})
                return {"comprehensive_review": result["output"]}
                
            except Exception as e:
                st.warning(f"Agent execution failed, using direct evaluation: {str(e)}")
        
        # Fallback to direct tool usage
        return self._direct_evaluation(question, answer, reference)
    
    def _direct_evaluation(self, question: str, answer: str, reference: str) -> Dict:
        """Direct evaluation using tools without agent framework"""
        results = {}
        
        try:
            # Add delays to prevent rate limiting
            time.sleep(1)
            # Technical evaluation
            tech_input = f"{question}|||{answer}|||{reference}"
            results["technical"] = self.tools[0].func(tech_input)
            
            time.sleep(1)
            # Communication evaluation
            results["communication"] = self.tools[1].func(answer)
            
            time.sleep(1)
            # Behavioral evaluation
            behav_input = f"{question}|||{answer}"
            results["behavioral"] = self.tools[2].func(behav_input)
            
            time.sleep(1)
            # Answer rating
            rating_input = f"{question}|||{answer}|||{reference}"
            results["rating"] = self.tools[5].func(rating_input)
            
            time.sleep(1)
            # Improvement advice
            eval_summary = f"Technical: {results['technical']}\nCommunication: {results['communication']}\nBehavioral: {results['behavioral']}\nRating: {results['rating']}"
            results["improvement_advice"] = self.tools[3].func(eval_summary)
            
            time.sleep(1)
            # Grammar correction
            results["grammar_correction"] = self.tools[4].func(answer)
            
        except Exception as e:
            st.error(f"Direct evaluation failed: {str(e)}")
            results["error"] = str(e)
        
        return results


class RAGManager:
    """Enhanced RAG system using LangChain"""
    
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


def clean_transcript_segments(segments: List[Dict], llm) -> List[Dict]:
    """Clean transcript segments using LangChain"""
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


def extract_qa_pairs(segments: List[Dict]) -> List[Dict]:
    """Extract Q&A pairs from transcript segments"""
    qa_pairs = []
    current_question = None
    current_response = []
    
    for segment in segments:
        if segment["speaker"] == "Interviewer":
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


def main():
    st.set_page_config(
        page_title="AI Mock Interview Feedback System", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎤 AI Mock Interview Feedback System")
    st.markdown("""
    **Enhanced Features:**
    - 🎯 AssemblyAI for accurate audio transcription with speaker diarization
    - 🤖 LangChain-based agent framework for comprehensive evaluation
    - 📚 Advanced RAG system for contextual feedback
    - 📊 Multi-dimensional assessment (Technical, Communication, Behavioral)
    - ⭐ Answer rating against reference standards
    - ✍️ Grammar correction and improvement suggestions
    """)
    
    # Initialize components
    if "transcription_agent" not in st.session_state:
        st.session_state.transcription_agent = AudioTranscriptionAgent(ASSEMBLYAI_API_KEY)
    
    if "reviewer_agent" not in st.session_state:
        st.session_state.reviewer_agent = ReviewerAgent(GOOGLE_API_KEY)
    
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = RAGManager(GOOGLE_API_KEY)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Configuration")
        
        # PDF upload
        pdf_file = st.file_uploader("Upload Reference Q&A PDF", type=["pdf"])
        
        if pdf_file:
            if st.button("🔨 Build Knowledge Base"):
                with st.spinner("Building knowledge base..."):
                    success = st.session_state.rag_manager.build_knowledge_base(pdf_file)
                    if success:
                        st.success("✅ Knowledge base built successfully!")
                    else:
                        st.error("❌ Failed to build knowledge base")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        evaluation_depth = st.selectbox(
            "Evaluation Depth",
            ["Quick", "Comprehensive", "Detailed"],
            index=1
        )
        
        include_sentiment = st.checkbox("Include Sentiment Analysis", True)
        include_highlights = st.checkbox("Include Key Highlights", True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Audio upload
        uploaded_file = st.file_uploader(
            "🎵 Upload Interview Recording", 
            type=["mp3", "wav", "m4a", "flac"]
        )
        
        if uploaded_file and st.session_state.rag_manager.vectorstore:
            process_interview(
                uploaded_file, 
                st.session_state.transcription_agent,
                st.session_state.reviewer_agent,
                st.session_state.rag_manager,
                evaluation_depth,
                include_sentiment,
                include_highlights
            )
        elif uploaded_file and not st.session_state.rag_manager.vectorstore:
            st.warning("⚠️ Please upload and build knowledge base from PDF first!")
    
    with col2:
        st.subheader("📊 System Status")
        
        # Status indicators
        kb_status = "✅ Ready" if st.session_state.rag_manager.vectorstore else "❌ Not Built"
        st.metric("Knowledge Base", kb_status)
        
        transcription_status = "✅ Ready" if st.session_state.transcription_agent else "❌ Not Ready"
        st.metric("Transcription Service", transcription_status)
        
        reviewer_status = "✅ Ready" if st.session_state.reviewer_agent else "❌ Not Ready"
        st.metric("Reviewer Agent", reviewer_status)


def process_interview(audio_file, transcription_agent, reviewer_agent, rag_manager, 
                     evaluation_depth, include_sentiment, include_highlights):
    """Process the interview recording"""
    
    with st.spinner("🎯 Processing interview recording..."):
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.getvalue())
            audio_path = tmp.name
        
        # Step 1: Transcription with AssemblyAI
        st.subheader("📝 Transcription Results")
        
        transcript_data = transcription_agent.transcribe_with_speakers(audio_path)
        
        if transcript_data["segments"]:
            # Display transcription
            with st.expander("View Full Transcript", expanded=False):
                for i, segment in enumerate(transcript_data["segments"]):
                    speaker_emoji = "👨‍💼" if segment["speaker"] == "Interviewer" else "👤"
                    st.write(f"{speaker_emoji} **{segment['speaker']}** ({segment['start_time']}): {segment['text']}")
            
            # Show highlights if available
            if include_highlights and transcript_data.get("summary"):
                st.subheader("🎯 Key Highlights")
                for highlight in transcript_data["summary"][:5]:
                    st.info(f"💡 {highlight}")
            
            # Clean transcript
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            cleaned_segments = clean_transcript_segments(transcript_data["segments"], llm)
            
            # Extract Q&A pairs
            qa_pairs = extract_qa_pairs(cleaned_segments)
            
            if qa_pairs:
                st.subheader("🔍 Interview Analysis")
                st.write(f"Found {len(qa_pairs)} question-answer pairs")
                
                # Process each Q&A pair
                for i, pair in enumerate(qa_pairs):
                    with st.expander(f"Question {i+1}: {pair['question'][:80]}...", expanded=True):
                        st.write("**Question:**")
                        st.write(pair["question"])
                        
                        st.write("**Candidate Response:**")
                        st.write(pair["response"])
                        
                        # Retrieve relevant context using RAG
                        relevant_contexts = rag_manager.retrieve_relevant_context(pair["question"])
                        reference_answer = rag_manager.extract_focused_reference(pair["question"], relevant_contexts)
                        
                        if reference_answer:
                            st.write("**Reference Context:**")
                            st.info(reference_answer)
                        
                        # Get comprehensive review from agent
                        with st.spinner("Generating comprehensive feedback..."):
                            # Add delay to prevent rate limiting
                            time.sleep(1)
                            review_results = reviewer_agent.comprehensive_review(
                                pair["question"], 
                                pair["response"], 
                                reference_answer
                            )
                        
                        # Display results
                        if "comprehensive_review" in review_results:
                            st.write("**🤖 Agent Review:**")
                            st.write(review_results["comprehensive_review"])
                        else:
                            # Display individual evaluations
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**🔧 Technical Evaluation**")
                                st.text_area("Technical Feedback", review_results.get("technical", "N/A"), height=100, key=f"tech_{i}")
                            
                            with col2:
                                st.write("**💬 Communication Assessment**") 
                                st.text_area("Communication Feedback", review_results.get("communication", "N/A"), height=100, key=f"comm_{i}")
                            
                            with col3:
                                st.write("**🎭 Behavioral Analysis**")
                                st.text_area("Behavioral Feedback", review_results.get("behavioral", "N/A"), height=100, key=f"behav_{i}")
                            
                            if "rating" in review_results:
                                st.write("**⭐ Answer Rating**")
                                st.text_area("Rating Feedback", review_results.get("rating", "N/A"), height=100, key=f"rating_{i}")
                            
                            if "grammar_correction" in review_results:
                                st.write("**✍️ Grammar Correction**")
                                st.text_area("Improved Response", review_results.get("grammar_correction", "N/A"), height=100, key=f"grammar_{i}")
                            
                            if "improvement_advice" in review_results:
                                st.write("**📈 Improvement Recommendations**")
                                st.success(review_results["improvement_advice"], icon="💡")
                        
                        st.divider()
                        
                        # Add delay between questions to prevent rate limiting
                        if i < len(qa_pairs) - 1:
                            time.sleep(3)
            else:
                st.warning("No clear question-answer pairs found in the transcript.")
        else:
            st.error("Transcription failed. Please check your audio file and try again.")


if __name__ == "__main__":
    main()