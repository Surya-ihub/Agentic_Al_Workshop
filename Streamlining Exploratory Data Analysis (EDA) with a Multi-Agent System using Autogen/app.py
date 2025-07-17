import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from autogen.agentchat import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

def gemini_call(prompt, model_name="models/gemini-1.5-flash"):
    return genai.GenerativeModel(model_name).generate_content(prompt).text

# ===== Agent Definitions =====
class DataPrepAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        df = st.session_state["df"]
        prompt = f"""You are a Data Cleaning Agent.
- Handle missing values
- Fix data types
- Remove duplicates

Dataset head:
{df.head().to_string()}

Summary stats:
{df.describe(include='all').to_string()}

Return Python code for preprocessing and a short explanation."""
        return gemini_call(prompt)

class EDAAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        df = st.session_state["df"]
        prompt = f"""You are an EDA Agent.
- Provide summary statistics
- Extract at least 3 insights
- Suggest visualizations

Dataset head:
{df.head().to_string()}"""
        return gemini_call(prompt)

class ReportGeneratorAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        insights = st.session_state.get("eda_output", "")
        prompt = f"""You are a Report Generator.
Create a clean EDA report based on insights:

{insights}

Include:
- Overview
- Key Findings
- Visual Suggestions
- Summary conclusion."""
        return gemini_call(prompt)

class CriticAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        report = st.session_state.get("report_output", "")
        prompt = f"""You are a Critic Agent.
Review the EDA report:

{report}

Comment on clarity, accuracy, completeness, and suggest improvements."""
        return gemini_call(prompt)

class ExecutorAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        code = st.session_state.get("prep_output", "")
        prompt = f"""You are an Executor Agent.
Validate the following data preprocessing code:

{code}

- Is it runnable?
- Suggest corrections if needed."""
        return gemini_call(prompt)

# ===== Admin / Proxy Agent =====
admin_agent = UserProxyAgent(
    name="Admin",
    human_input_mode="NEVER",
    code_execution_config=False  # disables Docker requirement
)

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("🔍 Agentic EDA with Gemini + Autogen")
st.markdown("Upload a CSV file and let our multi-agent system analyze it step-by-step.")

uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["df"] = df
    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    if st.button("🚀 Run Agentic EDA"):
        with st.spinner("Initializing agents..."):
            agents = [
                admin_agent,
                DataPrepAgent(name="DataPrep"),
                EDAAgent(name="EDA"),
                ReportGeneratorAgent(name="ReportGen"),
                CriticAgent(name="Critic"),
                ExecutorAgent(name="Executor"),
            ]
            chat = GroupChat(agents=agents, messages=[])
            manager = GroupChatManager(groupchat=chat)

        with st.spinner("Running multi-agent system..."):

            # ===== Data Preparation Output =====
            prep = agents[1].generate_reply([], "Admin")
            print("\n===== 🧹 DataPrepAgent Output =====")
            print(prep)
            st.session_state["prep_output"] = prep
            with st.expander("🧹 Data Preparation Output", expanded=True):
                st.markdown("**Python Code:**")
                st.code(prep, language="python")

            # ===== EDA Agent Output =====
            eda_out = agents[2].generate_reply([], "Admin")
            print("\n===== 📊 EDAAgent Output =====")
            print(eda_out)
            st.session_state["eda_output"] = eda_out
            with st.expander("📊 EDA Insights", expanded=True):
                st.markdown(eda_out)

            # ===== Report Generation =====
            report = agents[3].generate_reply([], "Admin")
            print("\n===== 📄 ReportGeneratorAgent Output =====")
            print(report)
            st.session_state["report_output"] = report
            with st.expander("📄 EDA Report", expanded=True):
                st.markdown(report)

            # ===== Critic Feedback =====
            critique = agents[4].generate_reply([], "Admin")
            print("\n===== 🧐 CriticAgent Output =====")
            print(critique)
            with st.expander("🧐 Critic Agent Feedback", expanded=False):
                st.markdown(critique)

            # ===== Code Execution Check =====
            exec_feedback = agents[5].generate_reply([], "Admin")
            print("\n===== ✅ ExecutorAgent Output =====")
            print(exec_feedback)
            with st.expander("✅ Executor Agent Validation", expanded=False):
                st.markdown(exec_feedback)

        st.success("✔️ Agentic EDA completed successfully.")
else:
    st.info("Upload a CSV file above to begin.")
