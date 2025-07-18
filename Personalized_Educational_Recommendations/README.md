# 🎓 Personalized Learning Assistant

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI Powered](https://img.shields.io/badge/AI%20Powered-Gemini%20%26%20CrewAI-purple.svg)](https://ai.google.dev/)

An intelligent, **AI-powered learning assistant** that generates personalized **learning materials**, **interactive quizzes**, and **hands-on project ideas** for any topic and skill level. Built with cutting-edge AI technology using **Google Gemini AI**, **Serper API**, and **CrewAI**.

---

## 🌟 **Demo & Screenshots**

### Main Interface
![Learning Assistant Interface](https://via.placeholder.com/800x400/667eea/ffffff?text=Learning+Assistant+Interface)

### Generated Learning Path
![Learning Path Results](https://via.placeholder.com/800x400/28a745/ffffff?text=Learning+Path+Results)

> *Replace with actual screenshots of your application*

---

## 🚀 **Key Features**

### 🔍 **Intelligent Content Curation**
- **Smart web search** for videos, articles, and exercises
- **AI-filtered results** ensuring quality and relevance
- **Multi-source aggregation** from trusted educational platforms

### 📝 **Interactive Assessment**
- **AI-generated quiz questions** with multiple-choice format
- **Immediate feedback** with correct answers
- **Knowledge testing** tailored to your learning level

### 🚧 **Project-Based Learning**
- **Skill-appropriate projects** (Beginner → Intermediate → Advanced)
- **Practical applications** to reinforce learning
- **Real-world scenarios** for hands-on experience

### 🤖 **Advanced AI Architecture**
- **Multi-agent system** using CrewAI framework
- **Specialized AI agents** for different learning aspects
- **Coordinated workflows** for comprehensive results

---

## 📦 **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive web interface |
| **AI Engine** | [Google Gemini 1.5 Flash](https://ai.google.dev/) | Content generation & analysis |
| **Web Search** | [Serper API](https://serper.dev/) | Real-time learning material search |
| **AI Framework** | [CrewAI](https://docs.crewai.com/) | Multi-agent orchestration |
| **LLM Integration** | [LangChain](https://www.langchain.com/) | Gemini LLM wrapper |
| **Backend** | Python 3.8+ | Core application logic |

---

## 📁 **Project Structure**

```
personalized-learning-assistant/
│
├── 📄 main.py                    # Main Streamlit application
├── 📄 requirements.txt           # Python dependencies
├── 📄 .env                      # Environment variables (not in repo)
├── 📄 .env.example              # Environment template
├── 📄 README.md                 # Project documentation
```

---

## 🔑 **Environment Setup**

### **Required API Keys**

1. **Google Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the generated key

2. **Serper API Key**
   - Go to [Serper.dev](https://serper.dev/)
   - Sign up for a free account
   - Navigate to API Keys section
   - Generate a new key

### **Environment Variables**

Create a `.env` file in the root directory:

```env
# Google Gemini API Configuration
GEMINI_API_KEY=your_google_gemini_api_key_here

# Serper API Configuration  
SERPER_API_KEY=your_serper_api_key_here

# Optional: Application Configuration
DEBUG=False
LOG_LEVEL=INFO
```

---

## ⚙️ **Installation & Setup**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### **Quick Start**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Surya-ihub/Agentic_Al_Workshop/tree/main/Automatic_code%20debugging%20Assistant
   cd personalized-learning-assistant
   ```

2. **Create virtual environment**
   ```bash
   # Using venv (recommended)
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your API keys
   nano .env  # or use your preferred editor
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Start creating your personalized learning paths!

---

## 📋 **Requirements**

```txt
streamlit>=1.28.0
google-generativeai>=0.3.0
langchain-google-genai>=0.0.8
crewai>=0.22.0
python-dotenv>=1.0.0
pydantic>=2.0.0
requests>=2.31.0
```

---

## 🧪 **Usage Examples**

### **Example 1: Learning Python Programming**
```
Topic: Python Programming
Level: Beginner
```
**Generated Output:**
- 📹 3 beginner-friendly Python tutorial videos
- 📚 3 comprehensive Python learning articles
- 💻 3 hands-on coding exercises
- ❓ 3 multiple-choice quiz questions
- 🚀 3 beginner projects (Calculator, To-Do List, Simple Game)

### **Example 2: Advanced Machine Learning**
```
Topic: Deep Learning Neural Networks
Level: Advanced
```
**Generated Output:**
- 📹 Advanced neural network architecture videos
- 📚 Research papers and advanced guides
- 💻 Complex implementation exercises
- ❓ Technical assessment questions
- 🚀 Advanced projects (Custom CNN, NLP Model, Reinforcement Learning)

---

## 🔧 **Configuration Options**

### **Skill Levels**
- **Beginner**: Basic concepts, guided tutorials, simple projects
- **Intermediate**: Applied knowledge, independent learning, moderate complexity
- **Advanced**: Expert-level content, research papers, complex implementations

### **Content Types**
- **Videos**: YouTube tutorials, course lectures, demonstrations
- **Articles**: Blog posts, documentation, guides, tutorials
- **Exercises**: Coding challenges, practice problems, hands-on labs
- **Projects**: Real-world applications, portfolio pieces, implementations

---

## 🚀 **Deployment**

### **Local Development**
```bash
streamlit run main.py --server.port 8501
```

### **Production Deployment**

#### **Streamlit Cloud**
1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

#### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Heroku Deployment**
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **API Key Error** | Verify keys in `.env` file and check API quotas |
| **Import Error** | Ensure all dependencies installed: `pip install -r requirements.txt` |
| **Search Results Empty** | Check Serper API key and internet connection |
| **Streamlit Not Found** | Install Streamlit: `pip install streamlit` |
| **Port Already in Use** | Change port: `streamlit run main.py --server.port 8502` |

### **Debug Mode**
```bash
# Run with debug information
streamlit run main.py --logger.level=debug
```

---

## 🤝 **Contributing**

We welcome contributions! Here's how you can help:

### **Ways to Contribute**
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit code improvements
- 🧪 Add tests and examples

### **Development Setup**
```bash


# Install development dependencies
pip install -r requirements-dev.txt


```

### **Code Style**
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where appropriate

---

## 📈 **Roadmap & Future Features**

### **Version 2.0 (Planned)**
- [ ] 👤 **User Authentication** - Save learning progress
- [ ] 📊 **Progress Tracking** - Visual learning analytics
- [ ] 🗂️ **Learning Path Management** - Save and organize paths
- [ ] 📱 **Mobile App** - React Native companion app

### **Version 2.1 (Planned)**
- [ ] 🔗 **Social Features** - Share learning paths
- [ ] 🏆 **Gamification** - Achievements and badges
- [ ] 📚 **Course Integration** - Connect with online courses
- [ ] 🎯 **Personalized Recommendations** - AI-driven suggestions

### **Version 3.0 (Future)**
- [ ] 🧠 **Advanced AI Tutoring** - Conversational learning assistant
- [ ] 🌐 **Multi-language Support** - Global accessibility
- [ ] 🔊 **Voice Integration** - Audio-based learning
- [ ] 🎨 **Custom Themes** - Personalized UI experience

---

## 📊 **Performance & Metrics**

### **System Requirements**
- **Memory**: 512MB RAM minimum
- **CPU**: 1 core minimum
- **Storage**: 100MB disk space
- **Network**: Stable internet connection

### **API Rate Limits**
- **Gemini API**: 60 requests/minute (free tier)
- **Serper API**: 100 searches/month (free tier)

---

## 🔒 **Security & Privacy**

### **Data Protection**
- ✅ No user data stored permanently
- ✅ API keys secured in environment variables
- ✅ HTTPS encryption for all API calls
- ✅ No tracking or analytics collection

### **Best Practices**
- Keep API keys secure and private
- Use environment variables for sensitive data
- Regularly update dependencies
- Follow security guidelines for deployment

---

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙌 **Acknowledgments**

### **Technologies & APIs**
- [Google Generative AI](https://ai.google.dev/) - Powering content generation
- [Serper API](https://serper.dev/) - Enabling intelligent web search
- [CrewAI Framework](https://docs.crewai.com/) - Multi-agent orchestration
- [Streamlit](https://streamlit.io/) - Beautiful web application framework
- [LangChain](https://www.langchain.com/) - LLM integration and tools

### **Community & Support**
- Stack Overflow community for troubleshooting
- GitHub open-source contributors
- Streamlit community for UI/UX inspiration
- AI/ML community for best practices



## ⭐ **Star History**



**If this project helped you, please consider giving it a ⭐!**

[⬆ Back to Top](#-personalized-learning-assistant)

</div>