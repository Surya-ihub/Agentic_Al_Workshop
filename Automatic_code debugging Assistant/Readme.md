# Python Code Reviewer🔍👨‍💻

A lightweight **Python code analysis and correction tool** built using **Streamlit + CrewAI + Gemini 2.5 Flash**, without using any ONNX or runtime code execution.

## 🚀 Features

- Static code analysis using Python's `ast` module
- Identifies common issues (e.g. `print()` in production, broad `except:` blocks)
- Provides clean, corrected code suggestions
- Uses **CrewAI** agents to simulate a code review team
- Integrates **Gemini 2.5 Flash** via `LLM` API wrapper
- 100% safe: No actual code execution

## 🚫 No Dependencies on:

- ONNX Runtime
- `chromadb`
- LangChain Tools like CodeInterpreterTool

## 🔧 Setup Instructions

### 1. Install dependencies

```bash
pip install streamlit crewai google-generativeai langchain-google-genai
```

### 2. Set your Gemini API key

Replace in `app.py`:

```python
os.environ["GOOGLE_API_KEY"] = "your-api-key"
```

Or use `.env` with `python-dotenv`.

### 3. Run the app

```bash
streamlit run app.py
```

## 🔎 Agents Used

- **Python Static Analyzer**: Scans code with AST and flags bad patterns
- **Python Code Fixer**: Rewrites the code to resolve flagged issues
- **Code Review Manager**: Coordinates task execution (CrewAI manager)

## 🎓 Key Concepts

- **AST Parsing** for static code inspection
- **CrewAI Agents** simulate real-world development roles
- **Process.sequential** to ensure analysis precedes correction
- **LLM with Gemini 2.5 Flash** to generate intelligent fixes

## 🚨 Commonly Detected Issues

- Use of `print()` in production
- Bare `except:` blocks
- Syntax errors

## 📄 Example

### Input:

```python
def foo():
    try:
        print("Start")
        1/0
    except:
        print("Error")
```

### Output:

- Flags `print()` and bare `except:`
- Suggests using `logging` and specifying exceptions
- Returns corrected, clean Python code

## 🏆 Benefits

- Lightweight
- Agent-powered automation
- Fully customizable static analysis rules

---

Built with ✨ Streamlit, CrewAI, Gemini 2.5 Flash
