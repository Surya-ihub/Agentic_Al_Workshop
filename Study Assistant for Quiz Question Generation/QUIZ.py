# study_assistant.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import json
import re

# Configure Streamlit
st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Study Assistant")
st.caption("Upload a PDF to get a summary and interactive quiz")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key="AIzaSyDG-0xIaprzdT70VTf-LnMt62_s-F8SJqA",
    temperature=0.3,
    max_output_tokens=2048
)

def extract_text(pdf_file):
    """Extract text from uploaded PDF"""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def generate_summary(text):
    """Generate bullet point summary using Gemini"""
    summary_prompt = """
    You are an expert study assistant. Create a concise bullet point summary 
    of the following study material. Follow these guidelines:
    - Extract key concepts and important facts
    - Use clear academic language
    - Organize information logically
    - Keep each bullet point to 1-2 sentences
    - Focus on the most critical 10-15 points
    
    Study Material:
    {text}

    Summary in bullet points:
    """
    prompt = ChatPromptTemplate.from_template(summary_prompt)
    chain = (
        {"text": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke(text)

def generate_quiz(summary):
    """Generate MCQs based on summary using Gemini with JSON output"""
    quiz_prompt = """
    Create 5 high-quality multiple choice questions based on the summary. 
    Follow these rules:
    1. Questions must test understanding of key concepts
    2. Provide 4 plausible options (A-D) per question
    3. Clearly indicate the correct answer
    4. Format your output as valid JSON with this structure:
    {{
        "questions": [
            {{
                "question": "Question text here",
                "options": {{
                    "A": "Option A text",
                    "B": "Option B text",
                    "C": "Option C text",
                    "D": "Option D text"
                }},
                "correct_answer": "A"
            }}
        ]
    }}
    
    Summary:
    {summary}
    """
    prompt = ChatPromptTemplate.from_template(quiz_prompt)
    chain = (
        {"summary": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = chain.invoke(summary)
    
    # Clean and parse the JSON
    try:
        # Remove markdown code block if present
        cleaned = re.sub(r'```json|```', '', result).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse quiz questions: {str(e)}")
        st.text("Raw output for debugging:")
        st.text(result)
        return {"questions": []}

# Initialize session state
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = {}
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# File upload section
uploaded_file = st.file_uploader("Upload study PDF", type="pdf")

if uploaded_file:
    # Reset state if new file uploaded
    if st.session_state.processed_file != uploaded_file.name:
        st.session_state.quiz_data = {}
        st.session_state.user_answers = {}
        st.session_state.submitted = False
        st.session_state.processed_file = uploaded_file.name
    
    # Extract text if not already done
    if not st.session_state.summary:
        with st.spinner("Reading PDF content..."):
            text = extract_text(uploaded_file)
        
        if not text.strip():
            st.error("Failed to extract text from PDF. Please try a different file.")
            st.stop()
        
        # Generate summary
        with st.spinner("Generating summary using Gemini 1.5 Flash..."):
            st.session_state.summary = generate_summary(text)
    
    # Display summary
    st.subheader("📝 Summary")
    st.markdown(st.session_state.summary)
    
    # Generate quiz if not already generated
    if not st.session_state.quiz_data:
        with st.spinner("Creating interactive quiz questions..."):
            quiz_data = generate_quiz(st.session_state.summary)
            st.session_state.quiz_data = quiz_data
    
    # Display quiz
    if st.session_state.quiz_data.get('questions'):
        st.subheader("❓ Interactive Quiz")
        st.caption("Select your answers and click Submit to check your understanding")
        
        with st.form(key='quiz_form'):
            for i, q in enumerate(st.session_state.quiz_data['questions']):
                st.markdown(f"**Q{i+1}:** {q['question']}")
                
                # Display options as radio buttons
                options = list(q['options'].items())
                user_choice = st.radio(
                    label="Select your answer:",
                    options=[opt[0] for opt in options],
                    format_func=lambda x: f"{x}) {q['options'][x]}",
                    key=f"question_{i}",
                    index=None
                )
                st.session_state.user_answers[i] = user_choice
                st.divider()
            
            submit_button = st.form_submit_button("Submit Answers")
        
        # Handle form submission
        if submit_button:
            st.session_state.submitted = True
        
        # Show results after submission
        if st.session_state.submitted:
            st.subheader("📊 Quiz Results")
            score = 0
            
            for i, q in enumerate(st.session_state.quiz_data['questions']):
                user_answer = st.session_state.user_answers.get(i)
                correct_answer = q['correct_answer']
                
                if user_answer == correct_answer:
                    score += 1
                    st.success(f"Q{i+1}: Correct! ✅ (Your answer: {user_answer})")
                else:
                    st.error(
                        f"Q{i+1}: Incorrect ❌ "
                        f"(Your answer: {user_answer or 'N/A'}, "
                        f"Correct answer: {correct_answer})"
                    )
                
                # Show explanation
                with st.expander(f"See explanation for Q{i+1}"):
                    st.markdown(f"**Question:** {q['question']}")
                    for opt, text in q['options'].items():
                        prefix = "✓ " if opt == correct_answer else "• "
                        st.markdown(f"{prefix}**{opt}:** {text}")
            
            # Display score
            st.divider()
            total = len(st.session_state.quiz_data['questions'])
            st.success(f"🎯 Your score: **{score}/{total}** ({score/total:.0%})")
            
            # Reset button
            if st.button("Take Quiz Again"):
                st.session_state.submitted = False
                st.session_state.user_answers = {}
                st.rerun()
    
    st.success("Processing complete!")
    st.caption(f"Filename: {uploaded_file.name} | Pages: {len(PdfReader(uploaded_file).pages)}")
    st.divider()
    st.info("💡 Study tip: Review the summary before taking the quiz for better results!")