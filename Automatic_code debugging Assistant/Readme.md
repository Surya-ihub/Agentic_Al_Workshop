# 🐍 AI-Powered Python Code Reviewer

An AI-powered tool that performs **static code analysis** and **auto-corrects Python code** — without executing it. Built using:

- 🧠 CrewAI agents (Analyzer, Fixer, Manager)
- 🤖 Google Gemini LLM (via LangChain)
- 🖥️ Streamlit UI for easy interaction

---

## ✨ Key Features

- 🔍 Analyzes Python code using the `ast` module
- ⚠️ Detects issues like:
  - `print()` usage in production
  - Bare `except:` blocks
- 🛠️ Automatically fixes issues using Gemini LLM
- 💬 Provides natural language explanations
- 🔒 Static analysis only — **no code execution**

---

## 🧠 Agent Workflow

1. **Analyzer Agent**  
   Parses the code using AST and identifies potential issues.

2. **Fixer Agent**  
   Suggests clean, PEP8-compliant corrections using Gemini.

3. **Manager Agent**  
   Coordinates the review and fix tasks.

---

## 🛠️ Installation Guide

pip install -r requirements.txt




