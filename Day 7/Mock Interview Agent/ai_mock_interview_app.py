import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import assemblyai as aai
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import queue

# Load environment variables
load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up Streamlit page
st.set_page_config(page_title="AI Mock Interview", layout="centered")
st.title("🎤 AI-Powered Mock Interview")

# Step 1: User enters interview topic
topic = st.text_input("🎯 Enter Interview Topic (e.g., Node.js, React, System Design):")

# Step 2: AI generates questions
if topic:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate.from_template("""
    You are an expert interviewer. Generate 5 technical interview questions on the topic: {topic}.
    Keep questions concise and beginner-to-intermediate level.
    """)
    chain = prompt | llm
    questions = chain.invoke({"topic": topic}).content.split("\n")
    st.session_state["questions"] = questions

    st.subheader("💡 Interview Questions")
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}:** {q}")

    # Step 3: Audio Recording using streamlit-webrtc
    st.subheader("🎙️ Record Your Interview Responses")
    audio_queue = queue.Queue()

    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        audio_queue.put(frame.to_ndarray().tobytes())
        return frame

    webrtc_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        in_audio=True,
        client_settings=ClientSettings(
            media_stream_constraints={"video": False, "audio": True},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        ),
        audio_frame_callback=audio_frame_callback,
    )

    if st.button("💾 Save Recording"):
        if not audio_queue.empty():
            audio_data = b"".join(list(audio_queue.queue))
            with open("interview_recording.wav", "wb") as f:
                f.write(audio_data)
            st.success("✅ Recording saved as 'interview_recording.wav'")
            st.audio("interview_recording.wav")
        else:
            st.warning("⚠️ No audio data recorded yet.")

# Step 4: Upload recorded file (optionally after download)
st.markdown("---")
st.subheader("📁 Upload a previously recorded interview")
uploaded_file = st.file_uploader("Upload interview audio file", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name
    st.audio(audio_path)

    # Step 5: Transcription with AssemblyAI
    st.info("🔍 Transcribing audio...")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=aai.TranscriptionConfig(speaker_labels=True))
    os.remove(audio_path)

    st.success("✅ Transcription complete.")
    st.subheader("📝 Transcript")
    candidate_texts = []
    candidate_speaker = "1"  # Assumption

    for utt in transcript.utterances:
        speaker = utt.speaker
        text = utt.text
        st.markdown(f"**Speaker {speaker}:** {text}")
        if speaker == candidate_speaker:
            candidate_texts.append(text)

    # Step 6: RAG + Feedback
    st.info("📚 Setting up vector DB and LLM feedback")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = TextLoader("ideal_answers.txt")
    docs = loader.load()
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)
    retriever = vectorstore.as_retriever()

    feedback_prompt = PromptTemplate(
        input_variables=["response", "ideal"],
        template="""
You are an expert technical interviewer.
Compare the candidate's answer with the ideal answer and generate feedback under 3 categories:
1. Technical Accuracy
2. Communication Clarity
3. Behavioral Fit

Candidate's Response:
{response}

Ideal Answer:
{ideal}

Give a score (1–5) for each category and clear improvement tips.
"""
    )

    feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)

    st.subheader("📊 AI Feedback")
    for i, candidate_response in enumerate(candidate_texts):
        docs = retriever.get_relevant_documents(candidate_response)
        ideal = docs[0].page_content if docs else "No ideal answer found."
        feedback = feedback_chain.run({"response": candidate_response, "ideal": ideal})
        st.markdown(f"### 💬 Candidate Response {i+1}")
        st.markdown(candidate_response)
        st.markdown("#### 📌 Feedback")
        st.markdown(feedback)

# Note: Put your ideal answers in ideal_answers.txt in the root directory.