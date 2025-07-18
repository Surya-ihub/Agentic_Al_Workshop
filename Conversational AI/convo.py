# competitor_analysis_agentic.py
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Callable
import copy

# Configure Streamlit
st.set_page_config(
    page_title="Clothing Store Competitor Intelligence",
    page_icon="👗",
    layout="centered"
)

# Create a function factory for generating replies
def create_gemini_reply_function(api_key: str) -> Callable:
    """Create a Gemini reply function that avoids deepcopy issues"""
    def generate_reply(messages: List[Dict], **kwargs) -> str:
        try:
            # Initialize LLM inside the function
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3,
                max_output_tokens=2048
            )
            
            # Convert AutoGen messages to LangChain format
            chat_history = []
            for msg in messages:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            response = llm.invoke(chat_history)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    return generate_reply

def create_agent_config(api_key: str, include_functions=True) -> Dict:
    """Create a properly configured llm_config"""
    config = {
        "config_list": [{
            "model": "gemini-1.5-flash",
            "api_type": "google",
            "api_key": api_key,
        }],
        "timeout": 600
    }
    
    if include_functions:
        config["functions"] = [create_gemini_reply_function(api_key)]
    
    return config

def main():
    # UI Setup
    st.title("👔 Clothing Store Competitor Intelligence")
    st.subheader("Agent-powered market analysis using Gemini 1.5 Flash")
    
    with st.sidebar:
        st.header("Configuration")
        gemini_api_key = st.text_input("🔑 Gemini API Key", type="password", value="AIzaSyDG-0xIaprzdT70VTf-LnMt62_s-F8SJqA")
        location = st.text_input("📍 Location", "Koramangala, Bangalore")
        competitors = st.slider("Number of competitors", 3, 10, 5)
        detail_level = st.selectbox("Detail Level", ["Summary", "Detailed", "Comprehensive"])
        generate_btn = st.button("Generate Report", type="primary")
    
    if generate_btn:
        if not gemini_api_key:
            st.error("Please enter your Gemini API key")
            return
            
        with st.spinner("Agent team analyzing competitors..."):
            try:
                # Create base agent config
                agent_config = create_agent_config(gemini_api_key, include_functions=True)
                manager_config = create_agent_config(gemini_api_key, include_functions=False)
                
                # Create agents with proper configuration
                research_analyst = AssistantAgent(
                    name="Research_Analyst",
                    llm_config=copy.deepcopy(agent_config),
                    system_message=f"""
                    As a retail market expert, analyze clothing stores in {location}.
                    Provide:
                    1. List of top {competitors} competitors
                    2. Their market positioning (luxury, mid-range, budget)
                    3. Foot traffic patterns (daily/weekly patterns)
                    4. Peak hours analysis (busiest times)
                    Present in clear bullet points. Detail level: {detail_level}
                    """
                )
                
                strategy_consultant = AssistantAgent(
                    name="Strategy_Consultant",
                    llm_config=copy.deepcopy(agent_config),
                    system_message=f"""
                    As a retail strategist, using Research_Analyst's data:
                    1. Compare price ranges and product offerings
                    2. Identify market gaps and opportunities
                    3. Recommend optimal strategies for:
                       - Operating hours
                       - Promotions timing
                       - Competitive differentiation
                    Make specific, actionable recommendations for {location}.
                    """
                )
                
                report_compiler = AssistantAgent(
                    name="Report_Compiler",
                    llm_config=copy.deepcopy(agent_config),
                    system_message=f"""
                    Compile a professional report with these sections:
                    ## Competitive Analysis: {location}
                    ### 1. Competitor Overview (table format)
                    ### 2. Market Analysis
                    ### 3. Strategic Recommendations
                    ### 4. Executive Summary
                    Format for business use with {detail_level} detail.
                    Use markdown formatting with tables where appropriate.
                    """
                )
                
                user_proxy = UserProxyAgent(
                    name="User_Proxy",
                    human_input_mode="NEVER",
                    code_execution_config=False,
                    max_consecutive_auto_reply=2,
                    default_auto_reply="Please continue the analysis..."
                )
                
                # Setup group chat
                groupchat = GroupChat(
                    agents=[user_proxy, research_analyst, strategy_consultant, report_compiler],
                    messages=[],
                    max_round=6,
                    speaker_selection_method="round_robin"
                )
                
                manager = GroupChatManager(
                    groupchat=groupchat,
                    llm_config=manager_config
                )
                
                # Initiate analysis
                user_proxy.initiate_chat(
                    manager,
                    message=f"""
                    Generate a {detail_level.lower()} competitor analysis for clothing stores in {location}.
                    Analyze {competitors} competitors including:
                    - Market positioning
                    - Foot traffic patterns
                    - Strategic recommendations
                    The final report should be comprehensive and business-ready.
                    """
                )
                
                # Display results
                st.success("Analysis Complete!")
                st.markdown("---")
                
                # Extract and display the final report
                final_report = None
                for msg in reversed(groupchat.messages):
                    if msg["name"] == "Report_Compiler" and "## Competitive Analysis" in msg.get("content", ""):
                        final_report = msg["content"]
                        break
                
                if final_report:
                    st.markdown(final_report)
                    st.download_button(
                        label="Download Report",
                        data=final_report,
                        file_name=f"competitor_analysis_{location.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.warning("Final report not found. Showing full conversation:")
                    for msg in groupchat.messages:
                        st.write(f"**{msg['name']}:**")
                        st.markdown(msg["content"])
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.info("Please check your API key and try again")

if __name__ == "__main__":
    main()