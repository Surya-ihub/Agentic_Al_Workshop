import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import ast

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAC1wHxDXyDGPdBfMvD6H76iA0L7cS7iU8"

# ----- Static Analysis Function -----
def analyze_python_code(code: str) -> str:
    try:
        tree = ast.parse(code)
        issues = []

        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print' 
               for node in ast.walk(tree)):
            issues.append("⚠️ Found `print()` - Use logging in production.")

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append("⚠️ Found bare `except:` - Specify exception types.")

        return "✅ No syntax errors found. Code looks good!" if not issues else "Found issues:\n" + "\n".join(issues)
    
    except SyntaxError as e:
        return f"❌ Syntax Error: {e.msg} (Line {e.lineno})"

# ----- LLM Setup -----
llm = LLM(
    api_key="AIzaSyAZtErluhP9-PX-Wd29D_QDWRG7V3xj6io",
    model="gemini/gemini-2.5-flash"
)

# ----- Agents -----
code_analyzer = Agent(
    role="Python Static Analyzer",
    goal="Find issues in Python code WITHOUT executing it",
    backstory="Expert in static code analysis using AST parsing.",
    llm=llm,
    verbose=True
)

code_corrector = Agent(
    role="Python Code Fixer",
    goal="Fix issues while keeping original functionality",
    backstory="Specializes in clean, PEP 8 compliant fixes.",
    llm=llm,
    verbose=True
)

manager = Agent(
    role="Code Review Manager",
    goal="Ensure smooth analysis & correction",
    backstory="Coordinates the review process.",
    llm=llm,
    verbose=True
)

# ----- Streamlit UI -----
st.set_page_config(page_title="Python Code Reviewer", page_icon="🐍", layout="centered")
st.title("🐍 AI-Powered Python Code Reviewer")
st.markdown("Analyze and fix your Python code using AI-powered agents without execution.")

# Input Section
with st.expander("📥 Paste Your Python Code", expanded=True):
    code_input = st.text_area("Write or paste your Python code below:", height=300, placeholder="# Your Python code here...")

# Action
if st.button("🚀 Analyze & Fix Code"):
    if not code_input.strip():
        st.warning("Please enter Python code to proceed.")
    else:
        with st.spinner("🤖 Running static analysis and corrections..."):

            # Tasks
            analysis_task = Task(
                description=f"Analyze this code:\n```python\n{code_input}\n```",
                agent=code_analyzer,
                expected_output="List of static analysis issues."
            )

            correction_task = Task(
                description="Fix all issues found.",
                agent=code_corrector,
                expected_output="Corrected Python code with explanations.",
                context=[analysis_task]
            )

            # CrewAI Execution
            crew = Crew(
                agents=[code_analyzer, code_corrector, manager],
                tasks=[analysis_task, correction_task],
                verbose=True,
                process=Process.sequential
            )

            result = crew.kickoff()

        # Output Section
        st.success("✅ Analysis and Fix Completed!")
        st.subheader("🔍 Code Review Result")
        st.code(result, language="python")

# Footer
st.markdown("---")
st.markdown("💡 _Built using [CrewAI](https://docs.crewai.com) and Gemini for static analysis and code correction._")
