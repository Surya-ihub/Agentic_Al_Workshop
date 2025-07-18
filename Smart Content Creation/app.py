import streamlit as st
import time
import google.generativeai as genai
from autogen import AssistantAgent, UserProxyAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import copy
import os

# Configure Gemini API
api_key = "AIzaSyDG-0xIaprzdT70VTf-LnMt62_s-F8SJqA"
genai.configure(api_key=api_key)

# System messages
CREATOR_SYSTEM_MESSAGE = """
You are a Content Creator Agent specializing in Generative AI. Your role is to:
1. Draft clear, concise, and technically accurate content
2. Revise content based on constructive feedback
3. Structure output in markdown format
4. Focus exclusively on content creation (no commentary)
"""

CRITIC_SYSTEM_MESSAGE = """
You are a Content Critic Agent evaluating Generative AI content. Your role is to:
1. Analyze technical accuracy and language clarity
2. Provide specific, constructive feedback
3. Identify both strengths and areas for improvement
4. Maintain professional, objective tone
"""

# Custom wrapper for deepcopy compatibility
class GeminiAgent:
    def __init__(self, model, system_message):
        self.model = model
        self.system_message = system_message
    
    def generate(self, prompt):
        full_prompt = self.system_message + "\n\n" + prompt
        try:
            response = self.model.invoke(full_prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def __deepcopy__(self, memo):
        # Create a new instance with same configuration
        return GeminiAgent(
            model=ChatGoogleGenerativeAI(model=self.model.model, google_api_key=api_key),
            system_message=self.system_message
        )

# Initialize Gemini models through LangChain
creator_model = GeminiAgent(
    model=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
    system_message=CREATOR_SYSTEM_MESSAGE
)

critic_model = GeminiAgent(
    model=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
    system_message=CRITIC_SYSTEM_MESSAGE
)

# Streamlit UI
st.title("🤖 Agentic AI Content Refinement")
st.caption("Simulated conversation between Content Creator and Content Critic agents")

# Controls
topic = st.text_input("Discussion Topic", "Agentic AI")
turns = st.slider("Conversation Turns", 3, 5, 3)
generate_btn = st.button("Start Simulation")

if generate_btn:
    # Create AutoGen agents with proper configuration
    creator = AssistantAgent(
        name="Creator",
        system_message=CREATOR_SYSTEM_MESSAGE,
        llm_config={
            "config_list": [
                {
                    "model": "gemini-1.5-flash",
                    "api_key": api_key,
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/models/"
                }
            ],
            "timeout": 120
        },
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )
    
    critic = AssistantAgent(
        name="Critic",
        system_message=CRITIC_SYSTEM_MESSAGE,
        llm_config={
            "config_list": [
                {
                    "model": "gemini-1.5-flash",
                    "api_key": api_key,
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/models/"
                }
            ],
            "timeout": 120
        },
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )
    
    # User proxy agent with Docker disabled
    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    
    # Initialize conversation state
    conversation_history = []
    creator_output = ""
    critic_feedback = ""
    
    # Start conversation
    for turn in range(1, turns + 1):
        with st.status(f"🚀 Turn {turn} in progress...", expanded=True):
            # Content Creator Turn (odd turns)
            if turn % 2 == 1:
                st.subheader(f"Turn {turn}: Content Creator")
                if turn == 1:
                    prompt = f"Draft comprehensive content about {topic} in markdown format covering:\n- Key concepts\n- Technical foundations\n- Real-world applications\n- Future implications"
                else:
                    prompt = f"Revise this content based on the critic's feedback:\n\n{critic_feedback}\n\nCurrent content:\n{creator_output}\n\nProvide improved markdown content:"
                
                st.markdown("**Prompt:**")
                st.write(prompt)
                
                # Generate content through creator model
                creator_output = creator_model.generate(prompt)
                
                st.markdown("**Generated Content:**")
                st.markdown(creator_output)
                conversation_history.append(("Creator", creator_output))
            
            # Content Critic Turn (even turns)
            else:
                st.subheader(f"Turn {turn}: Content Critic")
                prompt = f"Evaluate this content on:\n1. Technical accuracy\n2. Clarity of explanations\n3. Depth of coverage\n4. Improvement suggestions\n\nContent:\n{creator_output}"
                
                st.markdown("**Prompt:**")
                st.write(prompt)
                
                # Generate feedback through critic model
                critic_feedback = critic_model.generate(prompt)
                
                st.markdown("**Critical Feedback:**")
                st.write(critic_feedback)
                conversation_history.append(("Critic", critic_feedback))
            
            time.sleep(1)  # Avoid rate limiting
    
    # Display final output
    st.divider()
    st.subheader("✅ Final Content")
    st.markdown(creator_output)
    
    # Show conversation history
    st.divider()
    st.subheader("🗨️ Full Conversation Trace")
    for i, (role, content) in enumerate(conversation_history, 1):
        with st.expander(f"{role} - Turn {i}"):
            st.write(content)