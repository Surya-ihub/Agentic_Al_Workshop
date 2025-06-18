# import streamlit as st
# import PyPDF2
# import io
# import numpy as np
# import google.generativeai as genai
# import re
# from typing import List, Dict, Tuple
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# import string

# from dotenv import load_dotenv
# import os

# load_dotenv() 


# api_key = os.getenv("GEMINI_API_KEY")

# print("API Key:", api_key)


# # Download required NLTK data with updated punkt tokenizer
# def download_nltk_data():
#     """Download required NLTK data with proper error handling"""
#     try:
#         # Try the new punkt_tab first
#         try:
#             nltk.data.find('tokenizers/punkt_tab')
#         except LookupError:
#             try:
#                 nltk.download('punkt_tab', quiet=True)
#             except:
#                 # Fall back to punkt if punkt_tab is not available
#                 try:
#                     nltk.data.find('tokenizers/punkt')
#                 except LookupError:
#                     nltk.download('punkt', quiet=True)
        
#         # Download stopwords
#         try:
#             nltk.data.find('corpora/stopwords')
#         except LookupError:
#             nltk.download('stopwords', quiet=True)
            
#     except Exception as e:
#         st.warning(f"NLTK download issue: {e}. Will use basic text splitting.")

# # Initialize NLTK data
# download_nltk_data()

# # Page config
# st.set_page_config(
#     page_title="Interview Prep Assistant",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'pdf_processed' not in st.session_state:
#     st.session_state.pdf_processed = False
# if 'extracted_questions' not in st.session_state:
#     st.session_state.extracted_questions = []
# if 'selected_questions' not in st.session_state:
#     st.session_state.selected_questions = []
# if 'num_questions_selected' not in st.session_state:
#     st.session_state.num_questions_selected = False
# if 'current_question' not in st.session_state:
#     st.session_state.current_question = 0
# if 'user_answers' not in st.session_state:
#     st.session_state.user_answers = []
# if 'feedback_generated' not in st.session_state:
#     st.session_state.feedback_generated = False

# class InterviewPrepRAG:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(
#             max_features=1000,
#             stop_words='english',
#             ngram_range=(1, 2)
#         )
#         self.chunks = []
#         self.chunk_vectors = None
#         self.is_fitted = False
        
#     def extract_text_from_pdf(self, pdf_file) -> str:
#         """Extract text from uploaded PDF file"""
#         try:
#             pdf_reader = PyPDF2.PdfReader(pdf_file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text() + "\n"
#             return text
#         except Exception as e:
#             st.error(f"Error extracting text from PDF: {str(e)}")
#             return ""
    
#     def preprocess_text(self, text: str) -> str:
#         """Clean and preprocess text"""
#         # Remove extra whitespace and newlines
#         text = re.sub(r'\s+', ' ', text)
#         # Remove special characters but keep basic punctuation
#         text = re.sub(r'[^\w\s\.\,\?\!\-\(\)]', '', text)
#         return text.strip()
    
#     def chunk_text_safe(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
#         """Split text into overlapping chunks with fallback method"""
#         try:
#             # Try NLTK sentence tokenization first
#             sentences = sent_tokenize(text)
            
#             chunks = []
#             current_chunk = ""
            
#             for sentence in sentences:
#                 # If adding this sentence would exceed chunk size, save current chunk
#                 if len(current_chunk.split()) + len(sentence.split()) > chunk_size:
#                     if current_chunk:
#                         chunks.append(current_chunk.strip())
#                     current_chunk = sentence
#                 else:
#                     current_chunk += " " + sentence if current_chunk else sentence
            
#             # Add the last chunk
#             if current_chunk:
#                 chunks.append(current_chunk.strip())
            
#             # If NLTK fails or no sentences found, use basic splitting
#             if not chunks:
#                 raise Exception("NLTK tokenization failed")
                
#         except Exception:
#             # Fallback to basic word-based chunking
#             st.info("Using basic text chunking (NLTK not available)")
#             words = text.split()
#             chunks = []
#             for i in range(0, len(words), chunk_size - overlap):
#                 chunk = " ".join(words[i:i + chunk_size])
#                 chunks.append(chunk)
        
#         return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter out very short chunks
    
#     def create_vector_db(self, chunks: List[str]):
#         """Create vector database from text chunks using TF-IDF"""
#         try:
#             self.chunks = [self.preprocess_text(chunk) for chunk in chunks]
            
#             if not self.chunks:
#                 raise Exception("No valid chunks to process")
            
#             # Fit vectorizer and transform chunks
#             self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
#             self.is_fitted = True
            
#             # Save to session state
#             st.session_state.vector_db = {
#                 'chunks': self.chunks,
#                 'vectorizer': self.vectorizer,
#                 'chunk_vectors': self.chunk_vectors,
#                 'is_fitted': True
#             }
            
#         except Exception as e:
#             st.error(f"Error creating vector database: {str(e)}")
#             raise e
    
#     def retrieve_answer_for_question(self, question: str, k: int = 5) -> str:
#         """Retrieve the most relevant chunks that contain the answer to the question"""
#         if not self.is_fitted or self.chunk_vectors is None:
#             return "No relevant content found."
        
#         try:
#             # Vectorize the question
#             query_vector = self.vectorizer.transform([self.preprocess_text(question)])
            
#             # Calculate cosine similarity
#             similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
#             # Get top k most similar chunks
#             top_indices = similarities.argsort()[-k:][::-1]
            
#             # Filter chunks with reasonable similarity scores
#             relevant_chunks = []
#             similarity_scores = []
            
#             for idx in top_indices:
#                 if similarities[idx] > 0.05:  # Lower threshold for better recall
#                     chunk_text = self.chunks[idx]
#                     relevant_chunks.append(chunk_text)
#                     similarity_scores.append(similarities[idx])
            
#             if relevant_chunks:
#                 # Combine relevant chunks
#                 combined_text = "\n\n".join(relevant_chunks)
                
#                 # Use Gemini to extract the specific answer from the retrieved content
#                 return self.extract_answer_with_gemini(question, combined_text)
#             else:
#                 return "No sufficiently relevant content found in the document."
            
#         except Exception as e:
#             st.error(f"Error retrieving answer: {str(e)}")
#             return "Error retrieving answer from document."
    
#     def extract_answer_with_gemini(self, question: str, context: str) -> str:
#         """Use Gemini to extract the specific answer from the retrieved context"""
#         try:
#             model = genai.GenerativeModel('gemini-1.5-flash')
            
#             prompt = f"""
#             Given the following question and context from a document, extract ONLY the direct answer to the question from the provided context. 
            
#             Do not generate additional content, explanations, or information not present in the context.
#             Do not add your own knowledge or assumptions.
#             Simply extract the relevant answer text that directly addresses the question.
            
#             QUESTION: {question}
            
#             CONTEXT FROM DOCUMENT:
#             {context}
            
#             INSTRUCTIONS:
#             - Extract only the text that directly answers the question
#             - Keep the answer concise and factual
#             - If no clear answer exists in the context, respond with "Answer not found in the provided context"
#             - Do not add introductory phrases like "The answer is" or "According to the document"
#             - Return only the factual content that answers the question
#             """
            
#             response = model.generate_content(prompt)
#             answer = response.text.strip()
            
#             # Clean up common AI-generated prefixes if they slip through
#             prefixes_to_remove = [
#                 "the answer is",
#                 "according to the document",
#                 "based on the context",
#                 "from the document",
#                 "the document states",
#                 "as mentioned in the context"
#             ]
            
#             answer_lower = answer.lower()
#             for prefix in prefixes_to_remove:
#                 if answer_lower.startswith(prefix):
#                     # Remove the prefix and clean up
#                     answer = answer[len(prefix):].strip()
#                     if answer.startswith(':'):
#                         answer = answer[1:].strip()
#                     break
            
#             return answer if answer else "Answer not found in the provided context"
            
#         except Exception as e:
#             st.error(f"Error extracting answer with Gemini: {str(e)}")
#             return "Error extracting answer from document."

# def initialize_gemini():
#     """Initialize Gemini API"""
    
#     if api_key:
#         try:
#             genai.configure(api_key=api_key)
#             return True
#         except Exception as e:
#             st.sidebar.error(f"Error configuring Gemini: {str(e)}")
#             return False
#     return False

# def extract_questions_from_pdf(content: str) -> List[str]:
#     """Extract questions that already exist in the PDF content using Gemini"""
#     try:
#         model = genai.GenerativeModel('gemini-1.5-flash')
        
#         # Use content in chunks to avoid token limits
#         content_sample = content[:6000] if len(content) > 6000 else content
        
#         prompt = f"""
#         Analyze the following text and extract ALL questions that already exist in the document.
#         Look for questions that:
#         1. End with question marks (?)
#         2. Start with question words (What, How, Why, When, Where, Which, Who, etc.)
#         3. Are structured as questions within the text
#         4. Include review questions, study questions, or any interrogative sentences
        
#         Text content:
#         {content_sample}
        
#         Please extract and list only the questions found in the text, one per line, without any additional formatting or numbering. 
#         If no questions are found, respond with "NO_QUESTIONS_FOUND".
#         Do not create new questions, only extract existing ones.
#         """
        
#         response = model.generate_content(prompt)
#         questions_text = response.text.strip()
        
#         if questions_text == "NO_QUESTIONS_FOUND" or not questions_text:
#             return []
        
#         # Parse questions from response
#         questions = []
#         lines = questions_text.split('\n')
#         for line in lines:
#             line = line.strip()
#             if line and len(line) > 10 and ('?' in line or any(line.lower().startswith(qw) for qw in ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can', 'does', 'is', 'are', 'will', 'would', 'could', 'should'])):
#                 # Clean up the question
#                 cleaned = re.sub(r'^\d+\.?\s*', '', line)
#                 cleaned = re.sub(r'^\*\s*', '', cleaned)
#                 cleaned = re.sub(r'^-\s*', '', cleaned)
#                 if cleaned and len(cleaned) > 10:
#                     questions.append(cleaned)
        
#         return questions
    
#     except Exception as e:
#         st.error(f"Error extracting questions: {str(e)}")
#         return []

# def get_comparative_feedback(question: str, user_answer: str, actual_answer: str) -> str:
#     """Generate feedback by comparing user's answer with the actual answer from PDF"""
#     try:
#         model = genai.GenerativeModel('gemini-1.5-flash')
        
#         prompt = f"""
#         You are an expert evaluator comparing a student's answer with the actual answer from source material.
        
#         QUESTION: {question}
        
#         STUDENT'S ANSWER: {user_answer}
        
#         ACTUAL ANSWER FROM SOURCE MATERIAL: {actual_answer}
        
#         Please provide detailed feedback that includes:
        
#         1. **ACCURACY SCORE**: Rate the student's answer from 1-10 based on how well it matches the actual answer
#         2. **WHAT THEY GOT RIGHT**: Specific points that align with the source material
#         3. **WHAT THEY MISSED**: Key information from the actual answer that the student didn't mention
#         4. **WHAT THEY GOT WRONG**: Any incorrect information in the student's answer
#         5. **COMPLETENESS**: How complete their answer was compared to the source
#         6. **SUGGESTIONS**: Specific recommendations to improve their answer
        
#         Be constructive but thorough in your evaluation. Focus on factual accuracy and completeness.
#         """
        
#         response = model.generate_content(prompt)
#         return response.text
    
#     except Exception as e:
#         return f"Error generating feedback: {str(e)}"

# # Main app
# def main():
#     st.title("📚 Interview Preparation Assistant with RAG")
#     st.markdown("Upload a PDF document to extract questions and practice with answers from the source material!")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Settings")
        
#         # Gemini API setup
#         if not initialize_gemini():
#             st.warning("⚠️ Please enter your Gemini API key to continue")
#             st.info("Get your API key from: https://makersuite.google.com/app/apikey")
#             st.markdown("---")
#             st.markdown("### How to get Gemini API Key:")
#             st.markdown("1. Visit Google AI Studio")
#             st.markdown("2. Sign in with your Google account")
#             st.markdown("3. Create a new API key")
#             st.markdown("4. Copy and paste it above")
#             return
        
#         st.success("✅ Gemini API configured")
        
#         st.markdown("---")
#         st.markdown("### 📋 RAG Process:")
#         st.markdown("1. **Upload PDF** - Extract text content")
#         st.markdown("2. **Create Vector DB** - Store document chunks")
#         st.markdown("3. **Extract Questions** - Find questions IN the PDF")
#         st.markdown("4. **Select Questions** - Choose how many to practice")
#         st.markdown("5. **Answer Questions** - Practice with selected questions")
#         st.markdown("6. **RAG Retrieval** - Get actual answers from PDF")
#         st.markdown("7. **AI Comparison** - Compare your answer with source")
    
#     # Initialize RAG system
#     rag_system = InterviewPrepRAG()
    
#     # Step 1: PDF Upload and Processing
#     if not st.session_state.pdf_processed:
#         st.header("📄 Step 1: Upload PDF and Extract Questions")
        
#         uploaded_file = st.file_uploader(
#             "Choose a PDF file with questions", 
#             type="pdf",
#             help="Upload a PDF that contains questions (like study materials, textbooks, or question banks)"
#         )
        
#         if uploaded_file is not None:
#             with st.spinner("🔄 Processing PDF and extracting questions..."):
#                 try:
#                     # Extract text
#                     text = rag_system.extract_text_from_pdf(uploaded_file)
                    
#                     if text and len(text.strip()) > 100:
#                         st.info(f"📊 Extracted {len(text)} characters from PDF")
                        
#                         # Chunk text and create vector database
#                         chunks = rag_system.chunk_text_safe(text)
#                         st.info(f"📝 Created {len(chunks)} text chunks")
                        
#                         if chunks:
#                             # Create vector database
#                             rag_system.create_vector_db(chunks)
#                             st.info("🗄️ Vector database created")
                            
#                             # Extract questions from PDF content
#                             extracted_questions = extract_questions_from_pdf(text)
                            
#                             if extracted_questions:
#                                 st.session_state.extracted_questions = extracted_questions
#                                 st.session_state.pdf_processed = True
#                                 st.session_state.pdf_content = text
#                                 st.success(f"✅ Found {len(extracted_questions)} questions in the PDF!")
                                
#                                 # Show preview of extracted questions
#                                 with st.expander("🔍 Preview of Extracted Questions"):
#                                     for i, q in enumerate(extracted_questions[:5], 1):
#                                         st.write(f"{i}. {q}")
#                                     if len(extracted_questions) > 5:
#                                         st.write(f"... and {len(extracted_questions) - 5} more questions")
                                
#                                 st.balloons()
#                                 st.rerun()
#                             else:
#                                 st.error("❌ No questions found in the PDF. Please upload a PDF that contains questions.")
#                                 st.info("💡 The PDF should contain questions with question marks (?) or starting with question words (What, How, Why, etc.)")
#                         else:
#                             st.error("❌ Could not create text chunks from PDF. Please try a different file.")
#                     else:
#                         st.error("❌ Could not extract sufficient text from PDF. Please try a different file.")
                        
#                 except Exception as e:
#                     st.error(f"❌ Error processing PDF: {str(e)}")
    
#     # Step 2: Question Selection
#     elif not st.session_state.num_questions_selected:
#         st.header("🎯 Step 2: Select Number of Questions")
        
#         total_questions = len(st.session_state.extracted_questions)
#         st.info(f"📊 Total questions found in PDF: {total_questions}")
        
#         # Question selection slider
#         num_questions = st.slider(
#             "How many questions would you like to practice?",
#             min_value=1,
#             max_value=total_questions,
#             value=min(5, total_questions),
#             help="Select the number of questions you want to practice from the extracted questions"
#         )
        
#         # Selection method
#         selection_method = st.radio(
#             "Question selection method:",
#             ["First N questions", "Random selection", "Last N questions"],
#             help="Choose how to select the questions from the extracted list"
#         )
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if st.button("🎯 Start Practice Session", type="primary"):
#                 # Select questions based on method
#                 if selection_method == "First N questions":
#                     selected_questions = st.session_state.extracted_questions[:num_questions]
#                 elif selection_method == "Random selection":
#                     import random
#                     selected_questions = random.sample(st.session_state.extracted_questions, num_questions)
#                 else:  # Last N questions
#                     selected_questions = st.session_state.extracted_questions[-num_questions:]
                
#                 st.session_state.selected_questions = selected_questions
#                 st.session_state.num_questions_selected = True
#                 st.success(f"✅ Selected {num_questions} questions for practice!")
#                 st.rerun()
        
#         with col2:
#             if st.button("🔄 Reset Session"):
#                 for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'pdf_content', 'vector_db', 'selected_questions', 'num_questions_selected']:
#                     if key in st.session_state:
#                         del st.session_state[key]
#                 st.rerun()
        
#         # Preview selected questions
#         if num_questions <= 10:  # Only show preview for reasonable number of questions
#             with st.expander("🔍 Preview of Questions to Practice"):
#                 if selection_method == "First N questions":
#                     preview_questions = st.session_state.extracted_questions[:num_questions]
#                 elif selection_method == "Last N questions":
#                     preview_questions = st.session_state.extracted_questions[-num_questions:]
#                 else:
#                     st.write("Questions will be randomly selected when you start the practice session.")
#                     preview_questions = []
                
#                 for i, q in enumerate(preview_questions, 1):
#                     st.write(f"{i}. {q}")
    
#     # Step 3: Answer Questions
#     elif st.session_state.current_question < len(st.session_state.selected_questions):
#         current_q = st.session_state.current_question
#         total_q = len(st.session_state.selected_questions)
        
#         st.header(f"❓ Step 3: Question {current_q + 1} of {total_q}")
        
#         # Progress bar
#         progress = (current_q) / total_q
#         st.progress(progress)
        
#         # Display question
#         st.subheader("Question from PDF:")
#         st.info(st.session_state.selected_questions[current_q])
        
#         # Answer input
#         with st.form(f"question_{current_q}"):
#             user_answer = st.text_area(
#                 "Your Answer:", 
#                 height=150,
#                 placeholder="Provide your detailed answer based on what you know or remember from the document...",
#                 help="Answer the question as completely as possible. Your answer will be compared with the actual content from the PDF."
#             )
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 if st.form_submit_button("✅ Submit Answer", type="primary"):
#                     if user_answer.strip():
#                         st.session_state.user_answers.append({
#                             'question': st.session_state.selected_questions[current_q],
#                             'user_answer': user_answer
#                         })
#                         st.session_state.current_question += 1
#                         st.success("Answer submitted! Retrieving actual answer from PDF...")
#                         st.rerun()
#                     else:
#                         st.error("⚠️ Please provide an answer before submitting.")
            
#             with col2:
#                 if st.form_submit_button("⏭️ Skip Question"):
#                     st.session_state.user_answers.append({
#                         'question': st.session_state.selected_questions[current_q],
#                         'user_answer': "[Skipped]"
#                     })
#                     st.session_state.current_question += 1
#                     st.rerun()
            
#             with col3:
#                 if st.form_submit_button("🔄 Reset Session"):
#                     for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'pdf_content', 'vector_db', 'selected_questions', 'num_questions_selected']:
#                         if key in st.session_state:
#                             del st.session_state[key]
#                     st.rerun()
    
#     # Step 4: RAG Retrieval and Feedback Generation
#     elif not st.session_state.feedback_generated:
#         st.header("🤖 Step 4: RAG Retrieval and AI Feedback Generation")
        
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Load vector database from session state
#         if 'vector_db' in st.session_state:
#             rag_system.chunks = st.session_state.vector_db['chunks']
#             rag_system.vectorizer = st.session_state.vector_db['vectorizer']
#             rag_system.chunk_vectors = st.session_state.vector_db['chunk_vectors']
#             rag_system.is_fitted = st.session_state.vector_db['is_fitted']
        
#         feedbacks = []
#         actual_answers = []
        
#         for i, qa in enumerate(st.session_state.user_answers):
#             status_text.text(f"🔍 Retrieving answer {i + 1} from vector database...")
#             progress_bar.progress((i + 0.5) / len(st.session_state.user_answers))
            
#             # RAG: Always retrieve actual answer from vector database, regardless of skip status
#             actual_answer = rag_system.retrieve_answer_for_question(qa['question'])
#             actual_answers.append(actual_answer)
            
#             if qa['user_answer'] != "[Skipped]":
#                 status_text.text(f"🤖 Generating comparative feedback for answer {i + 1}...")
                
#                 # Generate comparative feedback
#                 feedback = get_comparative_feedback(qa['question'], qa['user_answer'], actual_answer)
#                 feedbacks.append(feedback)
#             else:
#                 feedbacks.append("⏭️ This question was skipped - no feedback available, but you can see the actual answer from the PDF below.")
            
#             progress_bar.progress((i + 1) / len(st.session_state.user_answers))
        
#         st.session_state.actual_answers = actual_answers
#         st.session_state.feedbacks = feedbacks
#         st.session_state.feedback_generated = True
#         status_text.text("✅ RAG retrieval and feedback generation complete!")
#         st.rerun()
    
#     # Step 5: Display Results
#     else:
#         st.header("📊 Interview Results & Comparative Analysis")
        
#         # Overall summary
#         answered_questions = sum(1 for qa in st.session_state.user_answers if qa['user_answer'] != "[Skipped]")
#         total_questions = len(st.session_state.selected_questions)
        
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Questions Practiced", total_questions)
#         with col2:
#             st.metric("Questions Answered", answered_questions)
#         with col3:
#             completion_rate = (answered_questions / total_questions) * 100 if total_questions > 0 else 0
#             st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
#         st.markdown("---")
        
#         # Detailed feedback for each question
#         st.subheader("📝 Question-by-Question Analysis")
        
#         for i, (qa, actual_answer, feedback) in enumerate(zip(st.session_state.user_answers, st.session_state.actual_answers, st.session_state.feedbacks)):
#             with st.expander(f"Question {i + 1}: {qa['question'][:60]}..."):
                
#                 st.markdown("**❓ Question (from PDF):**")
#                 st.write(qa['question'])
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("**💭 Your Answer:**")
#                     if qa['user_answer'] == "[Skipped]":
#                         st.write("*Question was skipped*")
#                     else:
#                         st.write(qa['user_answer'])
                
#                 with col2:
#                     st.markdown("**📖 Actual Answer (from PDF via Gemini):**")
#                     st.write(actual_answer)
                
#                 st.markdown("**🤖 Comparative Feedback:**")
#                 st.write(feedback)
#                 st.markdown("---")
        
#         # Action buttons
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔄 Start New Session", type="primary"):
#                 for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'actual_answers', 'pdf_content', 'vector_db', 'selected_questions', 'num_questions_selected']:
#                     if key in st.session_state:
#                         del st.session_state[key]
#                 st.rerun()
        
#         with col2:
#             if st.button("📥 Export Results"):
#                 results = {
#                     'total_questions_in_pdf': len(st.session_state.extracted_questions),
#                     'selected_questions': st.session_state.selected_questions,
#                     'user_answers': st.session_state.user_answers,
#                     'actual_answers': st.session_state.actual_answers,
#                     'comparative_feedbacks': st.session_state.feedbacks,
#                     'summary': {
#                         'total_questions_practiced': total_questions,
#                         'answered_questions': answered_questions,
#                         'completion_rate': completion_rate
#                     }
#                 }
                
#                 st.download_button(
#                     label="Download JSON",
#                     data=json.dumps(results, indent=2),
#                     file_name="interview_rag_results.json",
#                     mime="application/json"
#                 )

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import PyPDF2
# import io
# import numpy as np
# from typing import List, Dict, Tuple
# import json
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
# import string
# import re

# from dotenv import load_dotenv
# import os

# # LangChain imports
# from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.documents import Document

# load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")

# # Download required NLTK data
# def download_nltk_data():
#     """Download required NLTK data with proper error handling"""
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt', quiet=True)
    
#     try:
#         nltk.data.find('corpora/stopwords')
#     except LookupError:
#         nltk.download('stopwords', quiet=True)

# # Initialize NLTK data
# download_nltk_data()

# # Page config
# st.set_page_config(
#     page_title="Interview Prep Assistant",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'pdf_processed' not in st.session_state:
#     st.session_state.pdf_processed = False
# if 'extracted_questions' not in st.session_state:
#     st.session_state.extracted_questions = []
# if 'selected_questions' not in st.session_state:
#     st.session_state.selected_questions = []
# if 'num_questions_selected' not in st.session_state:
#     st.session_state.num_questions_selected = False
# if 'current_question' not in st.session_state:
#     st.session_state.current_question = 0
# if 'user_answers' not in st.session_state:
#     st.session_state.user_answers = []
# if 'feedback_generated' not in st.session_state:
#     st.session_state.feedback_generated = False
# if 'vectorstore' not in st.session_state:
#     st.session_state.vectorstore = None

# class InterviewPrepRAG:
#     def __init__(self):
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=api_key
#         )
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=api_key,
#             temperature=0.3
#         )
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#             is_separator_regex=False,
#         )
    
#     def extract_text_from_pdf(self, pdf_file) -> str:
#         """Extract text from uploaded PDF file"""
#         try:
#             pdf_reader = PyPDF2.PdfReader(pdf_file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text() + "\n"
#             return text
#         except Exception as e:
#             st.error(f"Error extracting text from PDF: {str(e)}")
#             return ""
    
#     def preprocess_text(self, text: str) -> str:
#         """Clean and preprocess text"""
#         # Remove extra whitespace and newlines
#         text = re.sub(r'\s+', ' ', text)
#         # Remove special characters but keep basic punctuation
#         text = re.sub(r'[^\w\s\.\,\?\!\-\(\)]', '', text)
#         return text.strip()
    
#     def create_vector_db(self, text: str):
#         """Create vector database from text using LangChain and FAISS"""
#         try:
#             # Split text into chunks
#             chunks = self.text_splitter.split_text(text)
            
#             if not chunks:
#                 raise Exception("No valid chunks to process")
            
#             # Create documents
#             docs = [Document(page_content=chunk) for chunk in chunks]
            
#             # Create vector store
#             vectorstore = FAISS.from_documents(
#                 documents=docs,
#                 embedding=self.embeddings
#             )
            
#             # Save to session state
#             st.session_state.vectorstore = vectorstore
            
#             return vectorstore
            
#         except Exception as e:
#             st.error(f"Error creating vector database: {str(e)}")
#             raise e
    
#     def retrieve_answer_for_question(self, question: str) -> str:
#         """Retrieve the most relevant answer using RAG pipeline"""
#         if st.session_state.vectorstore is None:
#             return "No vector database available for retrieval."
        
#         try:
#             # Create retriever
#             retriever = st.session_state.vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={"k": 4}
#             )
            
#             # Create prompt template
#             template = """Given the following question and context from a document, extract ONLY the direct answer to the question from the provided context. 
            
#             Do not generate additional content, explanations, or information not present in the context.
#             Do not add your own knowledge or assumptions.
#             Simply extract the relevant answer text that directly addresses the question.
            
#             QUESTION: {question}
            
#             CONTEXT FROM DOCUMENT:
#             {context}
            
#             INSTRUCTIONS:
#             - Extract only the text that directly answers the question
#             - Keep the answer concise and factual
#             - If no clear answer exists in the context, respond with "Answer not found in the provided context"
#             - Do not add introductory phrases like "The answer is" or "According to the document"
#             - Return only the factual content that answers the question
#             """
            
#             prompt = ChatPromptTemplate.from_template(template)
            
#             # Create RAG chain
#             rag_chain = (
#                 {"context": retriever, "question": RunnablePassthrough()}
#                 | prompt
#                 | self.llm
#                 | StrOutputParser()
#             )
            
#             # Invoke the chain
#             answer = rag_chain.invoke(question)
            
#             # Clean up common AI-generated prefixes if they slip through
#             prefixes_to_remove = [
#                 "the answer is",
#                 "according to the document",
#                 "based on the context",
#                 "from the document",
#                 "the document states",
#                 "as mentioned in the context"
#             ]
            
#             answer_lower = answer.lower()
#             for prefix in prefixes_to_remove:
#                 if answer_lower.startswith(prefix):
#                     # Remove the prefix and clean up
#                     answer = answer[len(prefix):].strip()
#                     if answer.startswith(':'):
#                         answer = answer[1:].strip()
#                     break
            
#             return answer if answer else "Answer not found in the provided context"
            
#         except Exception as e:
#             st.error(f"Error retrieving answer: {str(e)}")
#             return f"Error retrieving answer: {str(e)}"

# def extract_questions_from_pdf(content: str, llm: ChatGoogleGenerativeAI) -> List[str]:
#     """Extract questions that already exist in the PDF content using Gemini"""
#     try:
#         # Use content in chunks to avoid token limits
#         content_sample = content[:6000] if len(content) > 6000 else content
        
#         prompt = ChatPromptTemplate.from_template("""
#         Analyze the following text and extract ALL questions that already exist in the document.
#         Look for questions that:
#         1. End with question marks (?)
#         2. Start with question words (What, How, Why, When, Where, Which, Who, etc.)
#         3. Are structured as questions within the text
#         4. Include review questions, study questions, or any interrogative sentences
        
#         Text content:
#         {content}
        
#         Please extract and list only the questions found in the text, one per line, without any additional formatting or numbering. 
#         If no questions are found, respond with "NO_QUESTIONS_FOUND".
#         Do not create new questions, only extract existing ones.
#         """)
        
#         chain = prompt | llm | StrOutputParser()
#         questions_text = chain.invoke({"content": content_sample})
        
#         if questions_text == "NO_QUESTIONS_FOUND" or not questions_text:
#             return []
        
#         # Parse questions from response
#         questions = []
#         lines = questions_text.split('\n')
#         for line in lines:
#             line = line.strip()
#             if line and len(line) > 10 and ('?' in line or any(line.lower().startswith(qw) for qw in ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can', 'does', 'is', 'are', 'will', 'would', 'could', 'should'])):
#                 # Clean up the question
#                 cleaned = re.sub(r'^\d+\.?\s*', '', line)
#                 cleaned = re.sub(r'^\*\s*', '', cleaned)
#                 cleaned = re.sub(r'^-\s*', '', cleaned)
#                 if cleaned and len(cleaned) > 10:
#                     questions.append(cleaned)
        
#         return questions
    
#     except Exception as e:
#         st.error(f"Error extracting questions: {str(e)}")
#         return []

# def get_comparative_feedback(question: str, user_answer: str, actual_answer: str, llm: ChatGoogleGenerativeAI) -> str:
#     """Generate feedback by comparing user's answer with the actual answer from PDF"""
#     try:
#         prompt = ChatPromptTemplate.from_template("""
#         You are an expert evaluator comparing a student's answer with the actual answer from source material.
        
#         QUESTION: {question}
        
#         STUDENT'S ANSWER: {user_answer}
        
#         ACTUAL ANSWER FROM SOURCE MATERIAL: {actual_answer}
        
#         Please provide detailed feedback that includes:
        
#         1. **ACCURACY SCORE**: Rate the student's answer from 1-10 based on how well it matches the actual answer
#         2. **WHAT THEY GOT RIGHT**: Specific points that align with the source material
#         3. **WHAT THEY MISSED**: Key information from the actual answer that the student didn't mention
#         4. **WHAT THEY GOT WRONG**: Any incorrect information in the student's answer
#         5. **COMPLETENESS**: How complete their answer was compared to the source
#         6. **SUGGESTIONS**: Specific recommendations to improve their answer
        
#         Be constructive but thorough in your evaluation. Focus on factual accuracy and completeness.
#         """)
        
#         chain = prompt | llm | StrOutputParser()
#         feedback = chain.invoke({
#             "question": question,
#             "user_answer": user_answer,
#             "actual_answer": actual_answer
#         })
        
#         return feedback
    
#     except Exception as e:
#         return f"Error generating feedback: {str(e)}"

# # Main app
# def main():
#     st.title("📚 Interview Preparation Assistant with RAG (LangChain)")
#     st.markdown("Upload a PDF document to extract questions and practice with answers from the source material!")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Settings")
        
#         if not api_key:
#             st.warning("⚠️ Please set your Gemini API key in the .env file")
#             st.info("Get your API key from: https://makersuite.google.com/app/apikey")
#             st.markdown("---")
#             st.markdown("### How to get Gemini API Key:")
#             st.markdown("1. Visit Google AI Studio")
#             st.markdown("2. Sign in with your Google account")
#             st.markdown("3. Create a new API key")
#             st.markdown("4. Add it to your .env file as GEMINI_API_KEY")
#             return
        
#         st.success("✅ Gemini API key found")
        
#         st.markdown("---")
#         st.markdown("### 📋 RAG Process:")
#         st.markdown("1. **Upload PDF** - Extract text content")
#         st.markdown("2. **Create Vector DB** - Store document chunks")
#         st.markdown("3. **Extract Questions** - Find questions IN the PDF")
#         st.markdown("4. **Select Questions** - Choose how many to practice")
#         st.markdown("5. **Answer Questions** - Practice with selected questions")
#         st.markdown("6. **RAG Retrieval** - Get actual answers from PDF")
#         st.markdown("7. **AI Comparison** - Compare your answer with source")
    
#     # Initialize RAG system
#     rag_system = InterviewPrepRAG()
    
#     # Step 1: PDF Upload and Processing
#     if not st.session_state.pdf_processed:
#         st.header("📄 Step 1: Upload PDF and Extract Questions")
        
#         uploaded_file = st.file_uploader(
#             "Choose a PDF file with questions", 
#             type="pdf",
#             help="Upload a PDF that contains questions (like study materials, textbooks, or question banks)"
#         )
        
#         if uploaded_file is not None:
#             with st.spinner("🔄 Processing PDF and extracting questions..."):
#                 try:
#                     # Extract text
#                     text = rag_system.extract_text_from_pdf(uploaded_file)
                    
#                     if text and len(text.strip()) > 100:
#                         st.info(f"📊 Extracted {len(text)} characters from PDF")
                        
#                         # Create vector database
#                         vectorstore = rag_system.create_vector_db(text)
#                         st.info("🗄️ Vector database created")
                        
#                         # Extract questions from PDF content
#                         extracted_questions = extract_questions_from_pdf(text, rag_system.llm)
                        
#                         if extracted_questions:
#                             st.session_state.extracted_questions = extracted_questions
#                             st.session_state.pdf_processed = True
#                             st.session_state.pdf_content = text
#                             st.success(f"✅ Found {len(extracted_questions)} questions in the PDF!")
                            
#                             # Show preview of extracted questions
#                             with st.expander("🔍 Preview of Extracted Questions"):
#                                 for i, q in enumerate(extracted_questions[:5], 1):
#                                     st.write(f"{i}. {q}")
#                                 if len(extracted_questions) > 5:
#                                     st.write(f"... and {len(extracted_questions) - 5} more questions")
                            
#                             st.balloons()
#                             st.rerun()
#                         else:
#                             st.error("❌ No questions found in the PDF. Please upload a PDF that contains questions.")
#                             st.info("💡 The PDF should contain questions with question marks (?) or starting with question words (What, How, Why, etc.)")
#                     else:
#                         st.error("❌ Could not extract sufficient text from PDF. Please try a different file.")
                        
#                 except Exception as e:
#                     st.error(f"❌ Error processing PDF: {str(e)}")
    
#     # Step 2: Question Selection
#     elif not st.session_state.num_questions_selected:
#         st.header("🎯 Step 2: Select Number of Questions")
        
#         total_questions = len(st.session_state.extracted_questions)
#         st.info(f"📊 Total questions found in PDF: {total_questions}")
        
#         # Question selection slider
#         num_questions = st.slider(
#             "How many questions would you like to practice?",
#             min_value=1,
#             max_value=total_questions,
#             value=min(5, total_questions),
#             help="Select the number of questions you want to practice from the extracted questions"
#         )
        
#         # Selection method
#         selection_method = st.radio(
#             "Question selection method:",
#             ["First N questions", "Random selection", "Last N questions"],
#             help="Choose how to select the questions from the extracted list"
#         )
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if st.button("🎯 Start Practice Session", type="primary"):
#                 # Select questions based on method
#                 if selection_method == "First N questions":
#                     selected_questions = st.session_state.extracted_questions[:num_questions]
#                 elif selection_method == "Random selection":
#                     import random
#                     selected_questions = random.sample(st.session_state.extracted_questions, num_questions)
#                 else:  # Last N questions
#                     selected_questions = st.session_state.extracted_questions[-num_questions:]
                
#                 st.session_state.selected_questions = selected_questions
#                 st.session_state.num_questions_selected = True
#                 st.success(f"✅ Selected {num_questions} questions for practice!")
#                 st.rerun()
        
#         with col2:
#             if st.button("🔄 Reset Session"):
#                 for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'pdf_content', 'vectorstore', 'selected_questions', 'num_questions_selected']:
#                     if key in st.session_state:
#                         del st.session_state[key]
#                 st.rerun()
        
#         # Preview selected questions
#         if num_questions <= 10:  # Only show preview for reasonable number of questions
#             with st.expander("🔍 Preview of Questions to Practice"):
#                 if selection_method == "First N questions":
#                     preview_questions = st.session_state.extracted_questions[:num_questions]
#                 elif selection_method == "Last N questions":
#                     preview_questions = st.session_state.extracted_questions[-num_questions:]
#                 else:
#                     st.write("Questions will be randomly selected when you start the practice session.")
#                     preview_questions = []
                
#                 for i, q in enumerate(preview_questions, 1):
#                     st.write(f"{i}. {q}")
    
#     # Step 3: Answer Questions
#     elif st.session_state.current_question < len(st.session_state.selected_questions):
#         current_q = st.session_state.current_question
#         total_q = len(st.session_state.selected_questions)
        
#         st.header(f"❓ Step 3: Question {current_q + 1} of {total_q}")
        
#         # Progress bar
#         progress = (current_q) / total_q
#         st.progress(progress)
        
#         # Display question
#         st.subheader("Question from PDF:")
#         st.info(st.session_state.selected_questions[current_q])
        
#         # Answer input
#         with st.form(f"question_{current_q}"):
#             user_answer = st.text_area(
#                 "Your Answer:", 
#                 height=150,
#                 placeholder="Provide your detailed answer based on what you know or remember from the document...",
#                 help="Answer the question as completely as possible. Your answer will be compared with the actual content from the PDF."
#             )
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 if st.form_submit_button("✅ Submit Answer", type="primary"):
#                     if user_answer.strip():
#                         st.session_state.user_answers.append({
#                             'question': st.session_state.selected_questions[current_q],
#                             'user_answer': user_answer
#                         })
#                         st.session_state.current_question += 1
#                         st.success("Answer submitted! Retrieving actual answer from PDF...")
#                         st.rerun()
#                     else:
#                         st.error("⚠️ Please provide an answer before submitting.")
            
#             with col2:
#                 if st.form_submit_button("⏭️ Skip Question"):
#                     st.session_state.user_answers.append({
#                         'question': st.session_state.selected_questions[current_q],
#                         'user_answer': "[Skipped]"
#                     })
#                     st.session_state.current_question += 1
#                     st.rerun()
            
#             with col3:
#                 if st.form_submit_button("🔄 Reset Session"):
#                     for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'pdf_content', 'vectorstore', 'selected_questions', 'num_questions_selected']:
#                         if key in st.session_state:
#                             del st.session_state[key]
#                     st.rerun()
    
#     # Step 4: RAG Retrieval and Feedback Generation
#     elif not st.session_state.feedback_generated:
#         st.header("🤖 Step 4: RAG Retrieval and AI Feedback Generation")
        
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         feedbacks = []
#         actual_answers = []
        
#         for i, qa in enumerate(st.session_state.user_answers):
#             status_text.text(f"🔍 Retrieving answer {i + 1} from vector database...")
#             progress_bar.progress((i + 0.5) / len(st.session_state.user_answers))
            
#             # RAG: Always retrieve actual answer from vector database, regardless of skip status
#             actual_answer = rag_system.retrieve_answer_for_question(qa['question'])
#             actual_answers.append(actual_answer)
            
#             if qa['user_answer'] != "[Skipped]":
#                 status_text.text(f"🤖 Generating comparative feedback for answer {i + 1}...")
                
#                 # Generate comparative feedback
#                 feedback = get_comparative_feedback(
#                     qa['question'],
#                     qa['user_answer'],
#                     actual_answer,
#                     rag_system.llm
#                 )
#                 feedbacks.append(feedback)
#             else:
#                 feedbacks.append("⏭️ This question was skipped - no feedback available, but you can see the actual answer from the PDF below.")
            
#             progress_bar.progress((i + 1) / len(st.session_state.user_answers))
        
#         st.session_state.actual_answers = actual_answers
#         st.session_state.feedbacks = feedbacks
#         st.session_state.feedback_generated = True
#         status_text.text("✅ RAG retrieval and feedback generation complete!")
#         st.rerun()
    
#     # Step 5: Display Results
#     else:
#         st.header("📊 Interview Results & Comparative Analysis")
        
#         # Overall summary
#         answered_questions = sum(1 for qa in st.session_state.user_answers if qa['user_answer'] != "[Skipped]")
#         total_questions = len(st.session_state.selected_questions)
        
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Questions Practiced", total_questions)
#         with col2:
#             st.metric("Questions Answered", answered_questions)
#         with col3:
#             completion_rate = (answered_questions / total_questions) * 100 if total_questions > 0 else 0
#             st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
#         st.markdown("---")
        
#         # Detailed feedback for each question
#         st.subheader("📝 Question-by-Question Analysis")
        
#         for i, (qa, actual_answer, feedback) in enumerate(zip(st.session_state.user_answers, st.session_state.actual_answers, st.session_state.feedbacks)):
#             with st.expander(f"Question {i + 1}: {qa['question'][:60]}..."):
                
#                 st.markdown("**❓ Question (from PDF):**")
#                 st.write(qa['question'])
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("**💭 Your Answer:**")
#                     if qa['user_answer'] == "[Skipped]":
#                         st.write("*Question was skipped*")
#                     else:
#                         st.write(qa['user_answer'])
                
#                 with col2:
#                     st.markdown("**📖 Actual Answer (from PDF via Gemini):**")
#                     st.write(actual_answer)
                
#                 st.markdown("**🤖 Comparative Feedback:**")
#                 st.write(feedback)
#                 st.markdown("---")
        
#         # Action buttons
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔄 Start New Session", type="primary"):
#                 for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'actual_answers', 'pdf_content', 'vectorstore', 'selected_questions', 'num_questions_selected']:
#                     if key in st.session_state:
#                         del st.session_state[key]
#                 st.rerun()
        
#         with col2:
#             if st.button("📥 Export Results"):
#                 results = {
#                     'total_questions_in_pdf': len(st.session_state.extracted_questions),
#                     'selected_questions': st.session_state.selected_questions,
#                     'user_answers': st.session_state.user_answers,
#                     'actual_answers': st.session_state.actual_answers,
#                     'comparative_feedbacks': st.session_state.feedbacks,
#                     'summary': {
#                         'total_questions_practiced': total_questions,
#                         'answered_questions': answered_questions,
#                         'completion_rate': completion_rate
#                     }
#                 }
                
#                 st.download_button(
#                     label="Download JSON",
#                     data=json.dumps(results, indent=2),
#                     file_name="interview_rag_results.json",
#                     mime="application/json"
#                 )

# if __name__ == "__main__":
#     main()


import streamlit as st
import time
import PyPDF2
import io
import numpy as np
from typing import List, Dict, Tuple
import json
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import re

from dotenv import load_dotenv
import os

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

# Initialize NLTK data
download_nltk_data()

# Page config
st.set_page_config(
    page_title="Interview Prep Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'extracted_questions' not in st.session_state:
    st.session_state.extracted_questions = []
if 'selected_questions' not in st.session_state:
    st.session_state.selected_questions = []
if 'num_questions_selected' not in st.session_state:
    st.session_state.num_questions_selected = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []
if 'feedback_generated' not in st.session_state:
    st.session_state.feedback_generated = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

class InterviewPrepRAG:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\-\(\)]', '', text)
        return text.strip()
    
    def create_vector_db(self, text: str):
        """Create vector database from text using LangChain and FAISS"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                raise Exception("No valid chunks to process")
            
            # Create documents
            docs = [Document(page_content=chunk) for chunk in chunks]
            
            # Create vector store
            vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            # Save to session state
            st.session_state.vectorstore = vectorstore
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            raise e
    
    def retrieve_answer_for_question(self, question: str) -> str:
        """Retrieve the most relevant answer using RAG pipeline"""
        if st.session_state.vectorstore is None:
            return "No vector database available for retrieval."
        
        try:
            # Create retriever
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Create prompt template
            template = """Given the following question and context from a document, extract ONLY the direct answer to the question from the provided context. 
            
            Do not generate additional content, explanations, or information not present in the context.
            Do not add your own knowledge or assumptions.
            Simply extract the relevant answer text that directly addresses the question.
            
            QUESTION: {question}
            
            CONTEXT FROM DOCUMENT:
            {context}
            
            INSTRUCTIONS:
            - Extract only the text that directly answers the question
            - Keep the answer concise and factual
            - If no clear answer exists in the context, respond with "Answer not found in the provided context"
            - Do not add introductory phrases like "The answer is" or "According to the document"
            - Return only the factual content that answers the question
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create RAG chain
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Invoke the chain
            answer = rag_chain.invoke(question)
            
            # Clean up common AI-generated prefixes if they slip through
            prefixes_to_remove = [
                "the answer is",
                "according to the document",
                "based on the context",
                "from the document",
                "the document states",
                "as mentioned in the context"
            ]
            
            answer_lower = answer.lower()
            for prefix in prefixes_to_remove:
                if answer_lower.startswith(prefix):
                    # Remove the prefix and clean up
                    answer = answer[len(prefix):].strip()
                    if answer.startswith(':'):
                        answer = answer[1:].strip()
                    break
            
            return answer if answer else "Answer not found in the provided context"
            
        except Exception as e:
            st.error(f"Error retrieving answer: {str(e)}")
            return f"Error retrieving answer: {str(e)}"

def extract_questions_from_pdf(content: str, llm: ChatGoogleGenerativeAI) -> List[str]:
    """Extract questions that already exist in the PDF content using Gemini"""
    try:
        # Improved question extraction by processing in chunks
        chunk_size = 5000  # Process in chunks to handle large documents
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        all_questions = []
        
        for chunk in chunks:
            prompt = ChatPromptTemplate.from_template("""
            Analyze the following text and extract ALL questions that already exist in the document.
            Be very thorough and don't miss any questions. Include:
            1. Questions ending with question marks (?)
            2. Questions starting with question words (What, How, Why, When, Where, Which, Who, Can, Does, Is, Are, Will, Would, Could, Should)
            3. Questions in bullet points or numbered lists
            4. Questions in headings or subheadings
            5. Questions in bold or emphasized text
            6. Questions in review sections or practice sections
            
            IMPORTANT:
            - Extract ALL questions you can find, don't leave any out
            - Include questions even if they're not perfectly formatted
            - Preserve the original wording of the questions
            - Don't modify or rephrase the questions
            - Return each question on a new line
            
            Text content:
            {content}
            """)
            
            chain = prompt | llm | StrOutputParser()
            questions_text = chain.invoke({"content": chunk})
            
            if questions_text.strip().lower() == "no questions found" or not questions_text.strip():
                continue
                
            # Parse questions from response
            questions = []
            lines = questions_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Minimum question length
                    # Clean up the question
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove numbering
                    cleaned = re.sub(r'^[\*\-\•]\s*', '', cleaned)  # Remove bullets
                    cleaned = re.sub(r'^[Qq]\d*[\.\:]\s*', '', cleaned)  # Remove Q1:, Q2. etc
                    cleaned = cleaned.strip()
                    if cleaned and len(cleaned) > 10:
                        # Additional checks to confirm it's a question
                        if ('?' in cleaned or 
                            any(cleaned.lower().startswith(qw) for qw in 
                                ['what', 'how', 'why', 'when', 'where', 'which', 'who', 
                                 'can', 'does', 'is', 'are', 'will', 'would', 'could', 'should',
                                 'explain', 'describe', 'compare', 'contrast', 'list'])):
                            all_questions.append(cleaned)
            
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in all_questions:
            if q not in seen:
                seen.add(q)
                unique_questions.append(q)
                
        return unique_questions
    
    except Exception as e:
        st.error(f"Error extracting questions: {str(e)}")
        return []

def get_comparative_feedback(question: str, user_answer: str, actual_answer: str, llm: ChatGoogleGenerativeAI) -> str:
    """Generate feedback by comparing user's answer with the actual answer from PDF"""
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator comparing a student's answer with the actual answer from source material.
        
        QUESTION: {question}
        
        STUDENT'S ANSWER: {user_answer}
        
        ACTUAL ANSWER FROM SOURCE MATERIAL: {actual_answer}
        
        Please provide detailed feedback that includes:
        
        1. **ACCURACY SCORE**: Rate the student's answer from 1-10 based on how well it matches the actual answer
        2. **WHAT THEY GOT RIGHT**: Specific points that align with the source material
        3. **WHAT THEY MISSED**: Key information from the actual answer that the student didn't mention
        4. **WHAT THEY GOT WRONG**: Any incorrect information in the student's answer
        5. **COMPLETENESS**: How complete their answer was compared to the source
        6. **SUGGESTIONS**: Specific recommendations to improve their answer
        
        Be constructive but thorough in your evaluation. Focus on factual accuracy and completeness.
        """)
        
        chain = prompt | llm | StrOutputParser()
        feedback = chain.invoke({
            "question": question,
            "user_answer": user_answer,
            "actual_answer": actual_answer
        })
        
        return feedback
    
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

# Main app
def main():
    st.title("📚 Interview Preparation Assistant with RAG (Enhanced)")
    st.markdown("Upload a PDF document to extract questions and practice with answers from the source material!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if not api_key:
            st.warning("⚠️ Please set your Gemini API key in the .env file")
            st.info("Get your API key from: https://makersuite.google.com/app/apikey")
            st.markdown("---")
            st.markdown("### How to get Gemini API Key:")
            st.markdown("1. Visit Google AI Studio")
            st.markdown("2. Sign in with your Google account")
            st.markdown("3. Create a new API key")
            st.markdown("4. Add it to your .env file as GEMINI_API_KEY")
            return
        
        st.success("✅ Gemini API key found")
        
        st.markdown("---")
        st.markdown("### 📋 RAG Process:")
        st.markdown("1. **Upload PDF** - Extract text content")
        st.markdown("2. **Create Vector DB** - Store document chunks")
        st.markdown("3. **Extract Questions** - Find questions IN the PDF")
        st.markdown("4. **Select Questions** - Choose how many to practice")
        st.markdown("5. **Answer Questions** - Practice with selected questions")
        st.markdown("6. **RAG Retrieval** - Get actual answers from PDF")
        st.markdown("7. **AI Comparison** - Compare your answer with source")
    
    # Initialize RAG system
    rag_system = InterviewPrepRAG()
    
    # Step 1: PDF Upload and Processing
    if not st.session_state.pdf_processed:
        st.header("📄 Step 1: Upload PDF and Extract Questions")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file with questions", 
            type="pdf",
            help="Upload a PDF that contains questions (like study materials, textbooks, or question banks)"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Processing PDF and extracting questions..."):
                try:
                    # Extract text
                    text = rag_system.extract_text_from_pdf(uploaded_file)
                    
                    if text and len(text.strip()) > 100:
                        st.info(f"📊 Extracted {len(text)} characters from PDF")
                        
                        # Create vector database
                        vectorstore = rag_system.create_vector_db(text)
                        st.info("🗄️ Vector database created")
                        
                        # Extract questions from PDF content
                        extracted_questions = extract_questions_from_pdf(text, rag_system.llm)
                        
                        if extracted_questions:
                            st.session_state.extracted_questions = extracted_questions
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_content = text
                            st.success(f"✅ Found {len(extracted_questions)} questions in the PDF!")
                            
                            # Show preview of extracted questions
                            with st.expander("🔍 Preview of Extracted Questions"):
                                for i, q in enumerate(extracted_questions[:10], 1):
                                    st.write(f"{i}. {q}")
                                if len(extracted_questions) > 10:
                                    st.write(f"... and {len(extracted_questions) - 10} more questions")
                            
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ No questions found in the PDF. Please upload a PDF that contains questions.")
                            st.info("💡 The PDF should contain questions with question marks (?) or starting with question words (What, How, Why, etc.)")
                    else:
                        st.error("❌ Could not extract sufficient text from PDF. Please try a different file.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    # Step 2: Question Selection
    elif not st.session_state.num_questions_selected:
        st.header("🎯 Step 2: Select Number of Questions")
        
        total_questions = len(st.session_state.extracted_questions)
        st.info(f"📊 Total questions found in PDF: {total_questions}")
        
        # Question selection slider
        num_questions = st.slider(
            "How many questions would you like to practice?",
            min_value=1,
            max_value=total_questions,
            value=min(10, total_questions),
            help="Select the number of questions you want to practice from the extracted questions"
        )
        
        # Selection method
        selection_method = st.radio(
            "Question selection method:",
            ["First N questions", "Random selection", "Last N questions"],
            help="Choose how to select the questions from the extracted list"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Start Practice Session", type="primary"):
                # Select questions based on method
                if selection_method == "First N questions":
                    selected_questions = st.session_state.extracted_questions[:num_questions]
                elif selection_method == "Random selection":
                    import random
                    selected_questions = random.sample(st.session_state.extracted_questions, num_questions)
                else:  # Last N questions
                    selected_questions = st.session_state.extracted_questions[-num_questions:]
                
                st.session_state.selected_questions = selected_questions
                st.session_state.num_questions_selected = True
                st.success(f"✅ Selected {num_questions} questions for practice!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Session"):
                for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'pdf_content', 'vectorstore', 'selected_questions', 'num_questions_selected']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Preview selected questions
        if num_questions <= 20:  # Show preview for more questions now
            with st.expander("🔍 Preview of Questions to Practice"):
                if selection_method == "First N questions":
                    preview_questions = st.session_state.extracted_questions[:num_questions]
                elif selection_method == "Last N questions":
                    preview_questions = st.session_state.extracted_questions[-num_questions:]
                else:
                    st.write("Questions will be randomly selected when you start the practice session.")
                    preview_questions = []
                
                for i, q in enumerate(preview_questions, 1):
                    st.write(f"{i}. {q}")
    
    # Step 3: Answer Questions
    elif st.session_state.current_question < len(st.session_state.selected_questions):
        current_q = st.session_state.current_question
        total_q = len(st.session_state.selected_questions)
        
        st.header(f"❓ Step 3: Question {current_q + 1} of {total_q}")
        
        # Progress bar
        progress = (current_q) / total_q
        st.progress(progress)
        
        # Display question
        st.subheader("Question from PDF:")
        st.info(st.session_state.selected_questions[current_q])
        
        # Answer input
        with st.form(f"question_{current_q}"):
            user_answer = st.text_area(
                "Your Answer:", 
                height=200,
                placeholder="Provide your detailed answer based on what you know or remember from the document...",
                help="Answer the question as completely as possible. Your answer will be compared with the actual content from the PDF."
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.form_submit_button("✅ Submit Answer", type="primary"):
                    if user_answer.strip():
                        st.session_state.user_answers.append({
                            'question': st.session_state.selected_questions[current_q],
                            'user_answer': user_answer
                        })
                        st.session_state.current_question += 1
                        st.success("Answer submitted! Retrieving actual answer from PDF...")
                        st.rerun()
                    else:
                        st.error("⚠️ Please provide an answer before submitting.")
            
            with col2:
                if st.form_submit_button("⏭️ Skip Question"):
                    st.session_state.user_answers.append({
                        'question': st.session_state.selected_questions[current_q],
                        'user_answer': "[Skipped]"
                    })
                    st.session_state.current_question += 1
                    st.rerun()
            
            with col3:
                if st.form_submit_button("🔄 Reset Session"):
                    for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'pdf_content', 'vectorstore', 'selected_questions', 'num_questions_selected']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
    
    # Step 4: RAG Retrieval and Feedback Generation
    elif not st.session_state.feedback_generated:
        st.header("🤖 Step 4: RAG Retrieval and AI Feedback Generation")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        feedbacks = []
        actual_answers = []
        
        for i, qa in enumerate(st.session_state.user_answers):
            status_text.text(f"🔍 Retrieving answer {i + 1} from vector database...")
            progress_bar.progress((i + 0.5) / len(st.session_state.user_answers))
            
            # RAG: Always retrieve actual answer from vector database, regardless of skip status
            actual_answer = rag_system.retrieve_answer_for_question(qa['question'])
            actual_answers.append(actual_answer)
            
            if qa['user_answer'] != "[Skipped]":
                status_text.text(f"🤖 Generating comparative feedback for answer {i + 1}...")
                
                # Generate comparative feedback
                feedback = get_comparative_feedback(
                    qa['question'],
                    qa['user_answer'],
                    actual_answer,
                    rag_system.llm
                )
                feedbacks.append(feedback)
                time.sleep(15)
            else:
                feedbacks.append("⏭️ This question was skipped - no feedback available, but you can see the actual answer from the PDF below.")
            
            progress_bar.progress((i + 1) / len(st.session_state.user_answers))
        
        st.session_state.actual_answers = actual_answers
        st.session_state.feedbacks = feedbacks
        st.session_state.feedback_generated = True
        status_text.text("✅ RAG retrieval and feedback generation complete!")
        st.rerun()
    
    # Step 5: Display Results
    else:
        st.header("📊 Interview Results & Comparative Analysis")
        
        # Overall summary
        answered_questions = sum(1 for qa in st.session_state.user_answers if qa['user_answer'] != "[Skipped]")
        total_questions = len(st.session_state.selected_questions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Questions Practiced", total_questions)
        with col2:
            st.metric("Questions Answered", answered_questions)
        with col3:
            completion_rate = (answered_questions / total_questions) * 100 if total_questions > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        st.markdown("---")
        
        # Detailed feedback for each question
        st.subheader("📝 Question-by-Question Analysis")
        
        for i, (qa, actual_answer, feedback) in enumerate(zip(st.session_state.user_answers, st.session_state.actual_answers, st.session_state.feedbacks)):
            with st.expander(f"Question {i + 1}: {qa['question'][:60]}..."):
                
                st.markdown("**❓ Question (from PDF):**")
                st.write(qa['question'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**💭 Your Answer:**")
                    if qa['user_answer'] == "[Skipped]":
                        st.write("*Question was skipped*")
                    else:
                        st.write(qa['user_answer'])
                
                with col2:
                    st.markdown("**📖 Actual Answer (from PDF via Gemini):**")
                    st.write(actual_answer)
                
                st.markdown("**🤖 Comparative Feedback:**")
                st.write(feedback)
                st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Session", type="primary"):
                for key in ['pdf_processed', 'extracted_questions', 'current_question', 'user_answers', 'feedback_generated', 'feedbacks', 'actual_answers', 'pdf_content', 'vectorstore', 'selected_questions', 'num_questions_selected']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📥 Export Results"):
                results = {
                    'total_questions_in_pdf': len(st.session_state.extracted_questions),
                    'selected_questions': st.session_state.selected_questions,
                    'user_answers': st.session_state.user_answers,
                    'actual_answers': st.session_state.actual_answers,
                    'comparative_feedbacks': st.session_state.feedbacks,
                    'summary': {
                        'total_questions_practiced': total_questions,
                        'answered_questions': answered_questions,
                        'completion_rate': completion_rate
                    }
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(results, indent=2),
                    file_name="interview_rag_results.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()