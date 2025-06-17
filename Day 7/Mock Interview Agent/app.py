# import os
# import tempfile
# import streamlit as st
# import assemblyai as aai
# from dotenv import load_dotenv

# from langchain.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.document_loaders import TextLoader

# # -------------------- Setup --------------------
# load_dotenv()
# aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.set_page_config(page_title="Mock Interview RAG", layout="centered")
# st.title("🎙️ AI Mock Interview Feedback System")

# # -------------------- Upload Audio --------------------
# audio_file = st.file_uploader("Upload your mock interview audio (.mp3/.wav)", type=["mp3", "wav"])

# if audio_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
#         tmp.write(audio_file.read())
#         audio_path = tmp.name
#     st.audio(audio_path)

#     # -------------------- Transcribe with AssemblyAI --------------------
#     st.info("🔍 Transcribing...")
#     transcriber = aai.Transcriber()
#     transcript = transcriber.transcribe(audio_path, config=aai.TranscriptionConfig(speaker_labels=True))

#     st.success("✅ Transcription complete.")
#     os.remove(audio_path)

#     # -------------------- Extract Candidate Responses --------------------
#     st.subheader("🎤 Transcript")
#     candidate_texts = []

#     # Heuristic: Assume Speaker 1 is the candidate
#     candidate_speaker = "1"

#     for utt in transcript.utterances:
#         speaker = utt.speaker
#         text = utt.text
#         st.markdown(f"**Speaker {speaker}:** {text}")
#         if speaker == candidate_speaker:
#             candidate_texts.append(text)

#     # -------------------- Vector DB Setup (RAG) --------------------
#     st.info("📚 Setting up vector DB from ideal answers...")

#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     loader = TextLoader("ideal_answers.txt")
#     docs = loader.load()
#     vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
#     retriever = vectorstore.as_retriever()

#     # -------------------- Gemini LLM Setup --------------------
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

#     feedback_prompt = PromptTemplate(
#         input_variables=["response", "ideal"],
#         template="""
# You are an expert technical interviewer.

# Compare the candidate's answer with the ideal answer and generate feedback under 3 categories:
# 1. Technical Accuracy
# 2. Communication Clarity
# 3. Behavioral Fit

# Candidate's Response:
# {response}

# Ideal Answer:
# {ideal}

# Give a score (1–5) for each category and clear improvement tips.
# """
#     )

#     feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)

#     # -------------------- Generate Feedback --------------------
#     st.subheader("📊 AI Feedback")

#     for i, candidate_response in enumerate(candidate_texts):
#         docs = retriever.get_relevant_documents(candidate_response)
#         ideal = docs[0].page_content if docs else "No ideal answer found."
#         feedback = feedback_chain.run({"response": candidate_response, "ideal": ideal})
#         st.markdown(f"### 💬 Candidate Response {i+1}")
#         st.markdown(candidate_response)
#         st.markdown("#### 📌 Feedback")
#         st.markdown(feedback)


import os
import streamlit as st
import json
import tempfile
import google.generativeai as genai
from pydub import AudioSegment
import warnings

# Suppress Pydub warning about ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

# Configuration
MODEL_NAME = "gemini-1.5-flash"
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"  # Replace with your actual API key

# Initialize Gemini
genai.configure(api_key="AIzaSyBUgD1d0p8SNP73gbGTMb5mGCZajuHQ4Mo")
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

def main():
    st.set_page_config(page_title="AI Mock Interview Feedback", layout="wide")
    st.title("AI Mock Interview Feedback Generator")
    
    # Initialize agents
    transcriber = InterviewTranscriberAgent()
    evaluator = ResponseEvaluatorAgent()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Interview Details")
        role = st.selectbox("Target Role", 
                           ["Software Engineer", "Data Scientist", "Product Manager"])
        experience = st.selectbox("Experience Level", 
                                ["Entry", "Mid", "Senior"])
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Interview Recording", 
                                   type=["mp3", "wav", "m4a"])
    
    if uploaded_file is not None:
        with st.spinner("Processing your interview..."):
            # Save and convert audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_path = tmp.name
            
            # Convert to WAV for better compatibility
            try:
                audio_path = convert_audio_format(audio_path)
            except Exception as e:
                st.warning(f"Audio conversion warning: {str(e)}")
            
            # Step 1: Transcription
            st.subheader("Interview Transcription")
            transcript = transcriber.transcribe(audio_path)
            
            if transcript and "segments" in transcript:
                st.json(transcript)
                
                # Step 2: Extract Q&A pairs
                st.subheader("Question & Answer Analysis")
                qa_pairs = []
                current_question = None
                current_response = []
                
                for segment in transcript["segments"]:
                    if segment["speaker"] == "Interviewer":
                        if current_question and current_response:
                            qa_pairs.append({
                                "question": current_question,
                                "response": " ".join(current_response)
                            })
                        current_question = segment["text"]
                        current_response = []
                    else:
                        current_response.append(segment["text"])
                
                if current_question and current_response:
                    qa_pairs.append({
                        "question": current_question,
                        "response": " ".join(current_response)
                    })
                
                # Step 3: Evaluate responses
                for i, pair in enumerate(qa_pairs):
                    with st.expander(f"Q{i+1}: {pair['question'][:50]}..."):
                        st.write(f"**Response:** {pair['response']}")
                        
                        evaluation = evaluator.evaluate_response(
                            pair["question"], 
                            pair["response"], 
                            role
                        )
                        
                        if evaluation:
                            st.subheader("Evaluation")
                            cols = st.columns(3)
                            
                            with cols[0]:
                                st.metric("Technical", evaluation["technical"]["score"])
                                st.caption(evaluation["technical"]["feedback"])
                            
                            with cols[1]:
                                st.metric("Communication", evaluation["communication"]["score"])
                                st.caption(evaluation["communication"]["feedback"])
                            
                            with cols[2]:
                                st.metric("Behavioral", evaluation["behavioral"]["score"])
                                st.caption(evaluation["behavioral"]["feedback"])
                            
                            st.write("**Overall Feedback:**")
                            st.info(evaluation["overall_feedback"])
            else:
                st.error("Failed to generate transcript. Please try again.")

if __name__ == "__main__":
    main()
