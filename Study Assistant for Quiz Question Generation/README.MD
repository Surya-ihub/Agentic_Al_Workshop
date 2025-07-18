# 📚 AI Study Assistant

This Streamlit application allows you to upload a PDF file, receive a **concise summary**, and take an **AI-generated quiz** based on its content. It uses Google's **Gemini 1.5 Flash** model (via `langchain-google-genai`) to power the summarization and question generation.

---

## 🚀 Features

- 📝 PDF text extraction
- 📌 Bullet-point summary of key content
- ❓ Multiple Choice Questions (MCQs) quiz
- ✅ Real-time feedback and scoring
- 🔁 Option to retake the quiz

---

## 📂 How to Use

1. Upload a study PDF.
2. The app will:
   - Extract and summarize the content.
   - Generate 5 MCQs to test your understanding.
3. Answer the quiz and submit.
4. Get instant feedback and score.

---

## 🧠 Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 1.5 Flash via `langchain-google-genai`
- **PDF Parsing**: PyPDF2
- **Prompting & Chaining**: LangChain

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/study-assistant.git
cd study-assistant

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run study_assistant.py
