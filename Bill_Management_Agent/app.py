import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import tempfile
import json
import google.generativeai as genai
from autogen.agentchat import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# --- UI CONFIG ---
st.set_page_config(page_title="🧾 Bill Management Agent", layout="wide")
st.markdown("""
    <style>
    .big-font { font-size: 22px !important; font-weight: 600; }
    .agent-box { border-radius: 15px; background-color: #f1f1f1; color: black; padding: 15px; margin: 10px 0; }
    .user { background-color: #e0f7fa; color: black; padding: 12px; border-radius: 10px; margin-bottom: 10px; }
    .agent { background-color: #f3e5f5; color: black; padding: 12px; border-radius: 10px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("💼 AI Bill Management Agent")
st.markdown("Upload a bill and let AI categorize and analyze your expenses.")

# --- Upload File ---
uploaded_file = st.file_uploader("📤 Upload your bill", type=["jpg", "jpeg", "png"])

chat_log = []

# --- Gemini Vision to extract expense categories ---
def process_bill_with_gemini(image_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(image_file.read())
        tmp_path = tmp.name

    image = Image.open(tmp_path)

    response = model.generate_content([
        "Extract all expenses from this bill image. Group them into categories: Groceries, Dining, Utilities, Shopping, Entertainment, Others. Return as JSON format like {category: [{item, cost}]}.",
        image
    ])

    try:
        text = response.text.strip()
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        data = json.loads(text[json_start:json_end])
        return data, response.text
    except Exception as e:
        return None, response.text

# --- Gemini Summary ---
def summarize_expenses_with_gemini(expenses):
    prompt = (
        f"Given the following categorized expenses: {expenses}, "
        "summarize the total expenditure, show each category total, and mention which category has the highest cost and why it could be unusual."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

# --- AutoGen Agents (no Docker, Gemini only) ---
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    llm_config=False
)

bill_processing_agent = AssistantAgent(
    name="BillProcessingAgent",
    llm_config=False,
    system_message="You categorize expenses from a bill into standard categories."
)

summary_agent = AssistantAgent(
    name="ExpenseSummarizationAgent",
    llm_config=False,
    system_message="You analyze categorized expenses and summarize trends."
)

group_chat = GroupChat(agents=[user_proxy, bill_processing_agent, summary_agent])
manager = GroupChatManager(groupchat=group_chat)

# --- Main Execution Flow ---
if uploaded_file:
    st.success("✅ File uploaded. Processing...")

    with st.spinner("🔍 Extracting expenses..."):
        categorized_data, raw_response = process_bill_with_gemini(uploaded_file)

    if not categorized_data:
        st.error("❌ Failed to extract expenses.")
        st.text(raw_response)
    else:
        # 1. User → Group Manager
        user_proxy.send("Bill uploaded", manager)
        chat_log.append(("UserProxy → chat_manager", "Bill uploaded"))

        # 2. User → BillProcessingAgent
        user_proxy.send(f"Categorized expenses: {categorized_data}", bill_processing_agent)
        chat_log.append(("UserProxy → BillProcessingAgent", json.dumps(categorized_data, indent=2)))

        # 3. Simulate BillProcessingAgent response
        bp_response = "Categorization complete. Expenses sorted into available categories."
        chat_log.append(("BillProcessingAgent", bp_response))

        # 4. User → ExpenseSummarizationAgent
        user_proxy.send("Summarize this data", summary_agent)
        chat_log.append(("UserProxy → ExpenseSummarizationAgent", "Summarize this data"))

        # 5. Generate and simulate response
        with st.spinner("📊 Generating spending summary..."):
            summary = summarize_expenses_with_gemini(categorized_data)

        chat_log.append(("ExpenseSummarizationAgent", summary))

        # --- Display Categorized Expenses ---
        st.markdown("## 📂 Categorized Expenses")
        for category, items in categorized_data.items():
            if items:
                st.markdown(f"### 🗂️ {category}")
                for i in items:
                    st.markdown(f"- **{i['item']}**: ₹{i['cost']}")

        st.markdown("---")
        st.markdown("## 📋 Spending Summary")
        st.markdown(f"<div class='agent-box'>{summary}</div>", unsafe_allow_html=True)

        # --- Agent Chat Logs ---
        st.markdown("---")
        st.markdown("## 💬 Agent Chat Logs")
        for sender, message in chat_log:
            style = "user" if "UserProxy" in sender else "agent"
            st.markdown(f"<div class='{style}'><strong>{sender}</strong><br>{message}</div>", unsafe_allow_html=True)
