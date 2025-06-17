import os
import streamlit as st
import json
import tempfile
import google.generativeai as genai
from pydub import AudioSegment
import warnings
from pypdf import PdfReader
import numpy as np
import faiss
import pickle
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate

# Suppress Pydub warning about ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

# Configuration - FIX: Use your actual API key in both places
API_KEY = "AIzaSyD-ammIVL6zKq2ay7F35e7sSp8fKGQRqAg"  # Your actual API key
MODEL_NAME = "gemini-1.5-flash"

# Set environment variable for LangChain components
os.environ["GOOGLE_API_KEY"] = API_KEY

# Initialize Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


class InterviewTranscriberAgent:
    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio to text with timestamps"""
        try:
            # Upload the audio file
            audio_file = genai.upload_file(audio_path)

            prompt = """
            You are a professional interview transcription system. Transcribe the following interview audio 
            and provide the output in valid JSON format with time-stamped segments:
            
            {
                "segments": [
                    {
                        "start_time": "00:00:00",
                        "speaker": "Interviewer" or "Candidate",
                        "text": "transcribed text here"
                    },
                    ...
                ]
            }
            
            Important:
            1. Ensure the output is valid JSON that can be parsed with json.loads()
            2. Don't include any markdown or code formatting
            3. Be precise with speaker identification
            """

            response = self.model.generate_content([prompt, audio_file])

            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            return json.loads(response_text)
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            return {"segments": []}


class ResponseEvaluatorAgent:
    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)

    def evaluate_response(self, question: str, response: str, role: str) -> dict:
        """Evaluate a single Q&A pair"""
        prompt = f"""
        Evaluate this interview response for a {role} position. Provide feedback in valid JSON format:
        
        Question: {question}
        Response: {response}
        
        Use this JSON structure:
        {{
            "technical": {{
                "score": 1-5,
                "feedback": "technical evaluation here"
            }},
            "communication": {{
                "score": 1-5,
                "feedback": "communication evaluation here"
            }},
            "behavioral": {{
                "score": 1-5,
                "feedback": "behavioral evaluation here"
            }},
            "overall_feedback": "summary feedback here"
        }}
        
        Important: Only return valid JSON, no additional text or formatting.
        """

        try:
            result = self.model.generate_content(prompt)
            return json.loads(result.text)
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            return {}


def convert_audio_format(input_path, output_format="wav"):
    """Convert audio to WAV format for better compatibility"""
    sound = AudioSegment.from_file(input_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp:
        sound.export(tmp.name, format=output_format)
        return tmp.name


def extract_qa_from_text(text):
    lines = text.splitlines()
    qa_pairs = []
    question = None
    answer_lines = []
    for line in lines:
        if line.strip().endswith("?"):
            if question and answer_lines:
                qa_pairs.append(
                    {"question": question, "answer": " ".join(answer_lines).strip()}
                )
            question = line.strip()
            answer_lines = []
        else:
            if line.strip():
                answer_lines.append(line.strip())
    if question and answer_lines:
        qa_pairs.append(
            {"question": question, "answer": " ".join(answer_lines).strip()}
        )
    return qa_pairs


def get_gemini_embedding(text):
    """Get embedding for text using Gemini's embed_content API."""
    try:
        response = genai.embed_content(
            model="models/embedding-001", content=text, task_type="retrieval_document"
        )
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        st.error(f"Embedding failed: {str(e)}")
        return None


def retrieve_closest_qa(question, faiss_index, qa_pairs_pdf):
    """Retrieve the closest Q&A pair from FAISS for a given question."""
    if faiss_index is None or not qa_pairs_pdf:
        return None
    q_emb = get_gemini_embedding(question)
    if q_emb is None:
        return None
    q_emb = np.expand_dims(q_emb, axis=0)
    D, I = faiss_index.search(q_emb, 1)  # 1 nearest neighbor
    idx = I[0][0]
    if idx < len(qa_pairs_pdf):
        return qa_pairs_pdf[idx]
    return None


def save_faiss_index(index, qa_pairs, path_prefix):
    faiss.write_index(index, path_prefix + ".faiss")
    with open(path_prefix + ".pkl", "wb") as f:
        pickle.dump(qa_pairs, f)


def load_faiss_index(path_prefix):
    try:
        index = faiss.read_index(path_prefix + ".faiss")
        with open(path_prefix + ".pkl", "rb") as f:
            qa_pairs = pickle.load(f)
        return index, qa_pairs
    except Exception:
        return None, None


def clean_transcript_with_gemini_lc(transcript_segments):
    """Use LangChain Gemini LLM to clean transcript segments, keeping only meaningful Q&A, and post-filtering for short/filler segments."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert at cleaning interview transcripts. Given the following list of transcript segments (as JSON), do the following:
- Remove all segments that are not actual interview questions (from the interviewer) or substantive answers (from the candidate).
- Remove all short or irrelevant greetings, acknowledgments, and filler words (e.g., 'Yes, Lakshman.', 'Yeah.', 'Okay.', 'Hi.', 'Hello.', 'Thank you.', 'Thanks.', 'Okay.', 'Sure.', 'Yes.', 'No.', 'Alright.', 'Uh', 'Um', 'Okay. Thank you.', etc.).
- Only keep segments where:
    - The interviewer is asking a real question (usually ends with a question mark or starts with 'What', 'How', 'Why', 'Explain', etc.).
    - The candidate is providing a meaningful answer (more than 5 words, not just 'Yes', 'No', 'Thank you', etc.).
    - If there is a meaningful greeting (e.g., 'Good morning', 'Welcome to the interview'), keep it, but remove all other short greetings.
Return the cleaned list in the same JSON format (list of segments, each with start_time, speaker, text).

Segments:
{segments}

Important: Only return valid JSON, no extra text or formatting.
"""
    )
    chain = prompt | llm
    try:
        result = chain.invoke({"segments": transcript_segments})
        cleaned_text = result.content.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:-3].strip()
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:-3].strip()
        cleaned_segments = json.loads(cleaned_text)
        if not cleaned_segments or not isinstance(cleaned_segments, list):
            st.warning("Transcript cleaning returned no segments. Using original transcript.")
            return transcript_segments
        # Python post-filter: remove any segment with <5 words unless it's a valid greeting or question
        valid_greetings = ["good morning", "welcome", "good afternoon", "good evening"]
        def is_valid(segment):
            text = segment["text"].strip().lower()
            if len(text.split()) >= 5:
                return True
            if any(greet in text for greet in valid_greetings):
                return True
            if segment["speaker"].lower() == "interviewer" and (text.endswith("?") or text.startswith(("what", "how", "why", "explain"))):
                return True
            return False
        filtered_segments = [seg for seg in cleaned_segments if is_valid(seg)]
        if not filtered_segments:
            st.warning("All segments were filtered out. Using original transcript.")
            return transcript_segments
        return filtered_segments
    except Exception as e:
        st.warning(
            f"Transcript cleaning failed: {str(e)}. Proceeding with original transcript."
        )
        return transcript_segments


def process_pdf_with_langchain(pdf_file):
    """Load PDF, split into chunks, embed with Gemini, and store in FAISS using LangChain."""
    # Save PDF to a temp file for loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    # Load PDF with LangChain
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    # Combine all text for debugging purposes (optional, could be removed)
    all_text = "\n".join([p.page_content for p in pages])
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust chunk size as needed
        chunk_overlap=200,  # Adjust chunk overlap as needed
    )
    lc_docs = text_splitter.split_documents(pages)
    # Embed with Gemini (GoogleGenerativeAIEmbeddings)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if lc_docs:
        vectorstore = FAISS.from_documents(lc_docs, embeddings)
        return vectorstore, lc_docs, all_text  # Return all_text for debugging
    else:
        return None, lc_docs, all_text


def extract_relevant_answer_from_chunk(question, chunk, llm):
    """Use Gemini LLM to extract the most relevant answer from a chunk for a given question."""
    prompt = ChatPromptTemplate.from_template(
        """
Given the following chunk of text and a specific interview question, extract only the answer that best matches the question. If the chunk contains multiple Q&A pairs, return only the answer for the given question. If the answer is not present, return an empty string.

Chunk:
{chunk}

Question:
{question}

Important: Only return the answer text, no extra formatting or explanation.
"""
    )
    chain = prompt | llm
    try:
        result = chain.invoke({"chunk": chunk, "question": question})
        answer = result.content.strip()
        return answer
    except Exception as e:
        st.warning(f"Failed to extract relevant answer: {str(e)}. Using full chunk.")
        return chunk


def main():
    st.set_page_config(page_title="AI Mock Interview Feedback", layout="wide")
    st.title("AI Mock Interview Feedback Generator")
    st.markdown(
        """
    **Instructions:**
    1. Upload a PDF with Q&A pairs (format: Q: ... A: ...).
    2. Upload an interview audio file (mp3, wav, m4a).
    3. The app will extract, retrieve, and evaluate answers using AI.
    """
    )

    # Initialize agents
    transcriber = InterviewTranscriberAgent()
    evaluator = ResponseEvaluatorAgent()

    # Sidebar configuration
    with st.sidebar:
        st.header("Interview Details")
        pdf_file = st.file_uploader("Upload Q&A PDF", type=["pdf"])

    # PDF Q&A extraction and embedding
    qa_pairs_pdf = []
    faiss_index = None
    qa_texts = []
    pdf_id = None
    if pdf_file is not None:
        # Check file size (4MB = 4 * 1024 * 1024 bytes)
        if len(pdf_file.getvalue()) > 4 * 1024 * 1024:
            st.error("PDF file is too large. Please upload a file smaller than 4MB.")
            return
        # Check number of pages (max 8)
        try:
            reader = PdfReader(pdf_file)
            num_pages = len(reader.pages)
            if num_pages > 200:
                st.error(
                    f"PDF has {num_pages} pages. Please upload a PDF with no more than 8 pages."
                )
                return
        except Exception as e:
            st.error(f"Failed to read PDF: {str(e)}")
            return
        pdf_id = str(hash(pdf_file.getvalue()))
        with st.spinner(
            "Extracting and chunking PDF content, building vector store with LangChain..."
        ):
            try:
                vectorstore, lc_docs, all_text = process_pdf_with_langchain(pdf_file)
                if not lc_docs:
                    st.warning(
                        "No text chunks extracted from PDF. This might indicate an issue with PDF text extraction. Here is the raw extracted text for debugging:"
                    )
                    st.code(all_text)
                else:
                    st.success(f"Extracted {len(lc_docs)} text chunks from PDF.")
                    st.subheader("First 5 Extracted Chunks from PDF (for verification)")
                    for i, doc in enumerate(lc_docs[:5]):
                        st.markdown(f"**Chunk {i+1}:** {doc.page_content[:200]}...")
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["lc_docs"] = lc_docs
            except Exception as e:
                st.error(f"Failed to process PDF with LangChain: {str(e)}")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Interview Recording", type=["mp3", "wav", "m4a"]
    )

    if uploaded_file is not None:
        vectorstore = st.session_state.get("vectorstore")
        lc_docs = st.session_state.get("lc_docs")
        if not vectorstore or not lc_docs:
            st.error(
                "Please upload a valid Q&A PDF and ensure the index is built before uploading audio."
            )
            return
        with st.spinner("Processing your interview..."):
            # Save and convert audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_path = tmp.name
            try:
                audio_path = convert_audio_format(audio_path)
            except Exception as e:
                st.warning(f"Audio conversion warning: {str(e)}")
            # Step 1: Transcription
            st.subheader("Interview Transcription")
            transcript = transcriber.transcribe(audio_path)
            if transcript and "segments" in transcript:
                st.json(transcript)
                # Step 1.5: Clean transcript using LangChain Gemini LLM
                with st.spinner("Cleaning transcript..."):
                    cleaned_segments = clean_transcript_with_gemini_lc(
                        transcript["segments"]
                    )
                st.subheader("Cleaned Transcript Segments")
                st.json(cleaned_segments)
                # Step 2: Extract Q&A pairs from cleaned transcript
                st.subheader("Question & Answer Analysis (RAG)")
                qa_pairs = []
                current_question = None
                current_response = []
                for segment in cleaned_segments:
                    if segment["speaker"] == "Interviewer":
                        if current_question and current_response:
                            qa_pairs.append(
                                {
                                    "question": current_question,
                                    "response": " ".join(current_response),
                                }
                            )
                        current_question = segment["text"]
                        current_response = []
                    else:
                        current_response.append(segment["text"])
                if current_question and current_response:
                    qa_pairs.append(
                        {
                            "question": current_question,
                            "response": " ".join(current_response),
                        }
                    )
                # Step 3: RAG retrieval and feedback using LangChain
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                for i, pair in enumerate(qa_pairs):
                    with st.expander(f"Q{i+1}: {pair['question'][:50]}..."):
                        st.write(f"**User Response:** {pair['response']}")
                        # Retrieve closest chunk from vectorstore (RAG)
                        retr_res = vectorstore.similarity_search(pair["question"], k=1)
                        if retr_res:
                            # Use the entire retrieved chunk as the reference answer
                            reference_chunk_content = retr_res[0].page_content
                            # Extract only the most relevant answer for the current question
                            relevant_answer = extract_relevant_answer_from_chunk(pair["question"], reference_chunk_content, llm)
                            st.write(f"**Reference Answer:**")
                            st.info(relevant_answer)
                        else:
                            st.warning(
                                "No relevant reference content found for this question."
                            )
                        # Generate structured feedback using LangChain Gemini LLM
                        if retr_res:
                            feedback_prompt = ChatPromptTemplate.from_template(
                                """
You are an expert interview evaluator. Compare the user's answer to the provided reference answer.
Return feedback in this JSON format:
{{
  "technical": {{"score": 1-5, "feedback": "..."}},
  "communication": {{"score": 1-5, "feedback": "..."}},
  "behavioral": {{"score": 1-5, "feedback": "..."}},
  "improvement_tips": "...",
  "resource_links": ["https://...", "..."]
}}
Question: {question}
Reference Answer: {reference_answer}
User Answer: {user_answer}
Important: Only return valid JSON, no extra text.
"""
                            )
                            chain = feedback_prompt | llm
                            try:
                                result = chain.invoke(
                                    {
                                        "question": pair["question"],
                                        "reference_answer": relevant_answer,
                                        "user_answer": pair["response"],
                                    }
                                )
                                feedback_text = result.content.strip()
                                if feedback_text.startswith("```json"):
                                    feedback_text = feedback_text[7:-3].strip()
                                elif feedback_text.startswith("```"):
                                    feedback_text = feedback_text[3:-3].strip()
                                feedback = json.loads(feedback_text)
                                cols = st.columns(3)
                                with cols[0]:
                                    st.metric(
                                        "Technical",
                                        feedback.get("technical", {}).get(
                                            "score", "N/A"
                                        ),
                                        "out of 5",
                                    )
                                    st.caption(
                                        feedback.get("technical", {}).get(
                                            "feedback", ""
                                        )
                                    )
                                with cols[1]:
                                    st.metric(
                                        "Communication",
                                        feedback.get("communication", {}).get(
                                            "score", "N/A"
                                        ),
                                        "out of 5",
                                    )
                                    st.caption(
                                        feedback.get("communication", {}).get(
                                            "feedback", ""
                                        )
                                    )
                                with cols[2]:
                                    st.metric(
                                        "Behavioral",
                                        feedback.get("behavioral", {}).get(
                                            "score", "N/A"
                                        ),
                                        "out of 5",
                                    )
                                    st.caption(
                                        feedback.get("behavioral", {}).get(
                                            "feedback", ""
                                        )
                                    )
                                st.write("**Improvement Tips:**")
                                st.info(feedback.get("improvement_tips", ""))
                                resource_links = feedback.get("resource_links", [])
                                if resource_links:
                                    st.write("**Curated Resources:**")
                                    for link in resource_links:
                                        st.markdown(f"- [{link}]({link})")
                            except Exception as e:
                                st.error(f"Feedback generation failed: {str(e)}")
            else:
                st.error("Failed to generate transcript. Please try again.")


if __name__ == "__main__":
    main()
