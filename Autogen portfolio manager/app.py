# import streamlit as st
# import os
# import json
# import autogen
# from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# # Page configuration
# st.set_page_config(
#     page_title="Financial Portfolio Manager",
#     page_icon="💰",
#     layout="wide"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         color: #2E8B57;
#         font-size: 2.5em;
#         margin-bottom: 30px;
#     }
#     .agent-box {
#         border: 2px solid #4CAF50;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#         background-color: #f8f9fa;
#     }
#     .portfolio-summary {
#         background-color: #e8f5e8;
#         border-radius: 10px;
#         padding: 20px;
#         margin: 15px 0;
#     }
#     .recommendation-box {
#         background-color: #fff3cd;
#         border: 1px solid #ffeaa7;
#         border-radius: 8px;
#         padding: 15px;
#         margin: 10px 0;
#     }
#     .final-report {
#         background-color: #d4edda;
#         border: 1px solid #c3e6cb;
#         border-radius: 8px;
#         padding: 20px;
#         margin: 20px 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'agents_configured' not in st.session_state:
#     st.session_state.agents_configured = False
# if 'analysis_complete' not in st.session_state:
#     st.session_state.analysis_complete = False
# if 'final_report' not in st.session_state:
#     st.session_state.final_report = None

# def configure_agents(api_key):
#     """Configure all agents with the provided API key"""
    
#     config_list_gemini = [{
#         "model": "gemini-1.5-flash",
#         "api_key": api_key,
#         "api_type": "google"
#     }]
    
#     # Portfolio Analysis Agent
#     portfolio_analyst = AssistantAgent(
#         name="PortfolioAnalyst",
#         llm_config={"config_list": config_list_gemini},
#         system_message="""
#         You are a Portfolio Analysis Agent. Analyze the user's financial portfolio and determine investment strategy.
        
#         Based on the user's:
#         - Current salary
#         - Existing investments (FD, Mutual Funds, Stocks, Real Estate)
#         - Risk profile
        
#         Determine if they should pursue:
#         1. "Growth" strategy - for younger investors with higher risk tolerance
#         2. "Value" strategy - for conservative investors seeking stability
        
#         Output ONLY in JSON format: 
#         {"strategy": "Growth" or "Value", "reason": "brief explanation", "portfolio_summary": "summary of current portfolio"}
#         """
#     )
    
#     # Growth Investment Agent
#     growth_strategist = AssistantAgent(
#         name="GrowthStrategist",
#         llm_config={"config_list": config_list_gemini},
#         system_message="""
#         You are a Growth Investment Agent. Suggest high-growth investments for users following a growth strategy.
        
#         Recommendations should include:
#         - Mid-cap and small-cap mutual funds
#         - Technology and emerging sector stocks
#         - International ETFs
#         - Growth-oriented SIPs
        
#         Output in JSON format:
#         {"recommendations": ["item1", "item2", "item3"], "rationale": "explanation", "risk_level": "High/Medium"}
#         """
#     )
    
#     # Value Investment Agent
#     value_strategist = AssistantAgent(
#         name="ValueStrategist",
#         llm_config={"config_list": config_list_gemini},
#         system_message="""
#         You are a Value Investment Agent. Suggest stable, long-term investments for users following a value strategy.
        
#         Recommendations should include:
#         - Blue-chip stocks
#         - Government bonds and schemes
#         - Large-cap mutual funds
#         - Fixed deposits and PPF
        
#         Output in JSON format:
#         {"recommendations": ["item1", "item2", "item3"], "rationale": "explanation", "risk_level": "Low/Medium"}
#         """
#     )
    
#     # Investment Advisor Agent
#     financial_advisor = AssistantAgent(
#         name="FinancialAdvisor",
#         llm_config={"config_list": config_list_gemini},
#         system_message="""
#         You are a Financial Advisor Agent. Compile a comprehensive financial report with:
        
#         1. Portfolio Analysis Summary
#         2. Recommended Strategy (Growth/Value)
#         3. Specific Investment Recommendations
#         4. Implementation Plan
#         5. Risk Assessment
#         6. Expected Returns
        
#         Ensure the report is professional, detailed, and actionable for the user.
#         """
#     )
    
#     # User Proxy Agent
#     user_proxy = UserProxyAgent(
#         name="UserProxy",
#         human_input_mode="NEVER",
#         max_consecutive_auto_reply=1,
#         is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
#         code_execution_config=False
#     )
    
#     return portfolio_analyst, growth_strategist, value_strategist, financial_advisor, user_proxy

# def format_user_message(salary, fd_amount, mf_amount, stocks_amount, real_estate_amount, age, risk_tolerance):
#     """Format user input into a message for the agents"""
#     return f"""
#     User Investment Portfolio Information:
    
#     Personal Details:
#     - Age: {age} years
#     - Annual Salary: ₹{salary:,}
#     - Risk Tolerance: {risk_tolerance}
    
#     Current Investment Portfolio:
#     - Fixed Deposits: ₹{fd_amount:,}
#     - Mutual Funds (SIP): ₹{mf_amount:,}
#     - Stocks: ₹{stocks_amount:,}
#     - Real Estate: ₹{real_estate_amount:,}
    
#     Total Portfolio Value: ₹{fd_amount + mf_amount + stocks_amount + real_estate_amount:,}
    
#     Please analyze this portfolio and provide investment recommendations.
#     """

# def extract_json_from_response(content):
#     """Extract JSON from agent response"""
#     try:
#         # Try to find JSON in the response
#         start_idx = content.find('{')
#         end_idx = content.rfind('}') + 1
#         if start_idx != -1 and end_idx != -1:
#             json_str = content[start_idx:end_idx]
#             return json.loads(json_str)
#     except:
#         pass
    
#     # Return default if parsing fails
#     return {"strategy": "Growth", "reason": "Default strategy", "recommendations": [], "rationale": "Analysis pending"}

# def run_portfolio_analysis():
#     """Main function to run the portfolio analysis workflow"""
    
#     if not st.session_state.agents_configured:
#         st.error("Please configure agents first by entering your API key.")
#         return
    
#     # Get user inputs
#     salary = st.session_state.get('salary', 0)
#     fd_amount = st.session_state.get('fd_amount', 0)
#     mf_amount = st.session_state.get('mf_amount', 0)
#     stocks_amount = st.session_state.get('stocks_amount', 0)
#     real_estate_amount = st.session_state.get('real_estate_amount', 0)
#     age = st.session_state.get('age', 30)
#     risk_tolerance = st.session_state.get('risk_tolerance', 'Medium')
    
#     # Get configured agents
#     portfolio_analyst, growth_strategist, value_strategist, financial_advisor, user_proxy = st.session_state.agents
    
#     # Format user message
#     user_message = format_user_message(salary, fd_amount, mf_amount, stocks_amount, real_estate_amount, age, risk_tolerance)
    
#     # Progress tracking
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         # Step 1: Portfolio Analysis
#         status_text.text("🔍 Analyzing your portfolio...")
#         progress_bar.progress(25)
        
#         analysis_result = user_proxy.initiate_chat(
#             portfolio_analyst,
#             message=user_message,
#             summary_method="last_msg",
#             silent=True
#         )
        
#         # Extract strategy from analysis
#         analysis_data = extract_json_from_response(analysis_result.chat_history[-1]["content"])
#         strategy = analysis_data.get("strategy", "Growth")
        
#         # Step 2: Get Investment Recommendations
#         status_text.text(f"🎯 Getting {strategy} investment recommendations...")
#         progress_bar.progress(50)
        
#         agent = growth_strategist if strategy == "Growth" else value_strategist
#         recommendations_result = user_proxy.initiate_chat(
#             agent,
#             message=f"Strategy: {strategy}\n{user_message}",
#             summary_method="last_msg",
#             silent=True
#         )
        
#         # Step 3: Generate Final Report
#         status_text.text("📊 Generating comprehensive financial report...")
#         progress_bar.progress(75)
        
#         final_report_result = user_proxy.initiate_chat(
#             financial_advisor,
#             message=f"""
#             Portfolio Analysis: {analysis_result.summary}
#             Strategy Recommendations: {recommendations_result.summary}
#             User Details: {user_message}
            
#             Please provide a comprehensive financial report.
#             """,
#             summary_method="last_msg",
#             silent=True
#         )
        
#         progress_bar.progress(100)
#         status_text.text("✅ Analysis complete!")
        
#         # Store results in session state
#         st.session_state.analysis_complete = True
#         st.session_state.analysis_data = analysis_data
#         st.session_state.recommendations_data = extract_json_from_response(recommendations_result.chat_history[-1]["content"])
#         st.session_state.final_report = final_report_result.summary
        
#     except Exception as e:
#         st.error(f"An error occurred during analysis: {str(e)}")
#         progress_bar.progress(0)
#         status_text.text("❌ Analysis failed")

# # Main App
# def main():
#     st.markdown('<h1 class="main-header">💰 Financial Portfolio Manager</h1>', unsafe_allow_html=True)
#     st.markdown("**Multi-Agent Investment Advisory System**")
    
#     # Sidebar for API Configuration
#     with st.sidebar:
#         st.header("🔧 Configuration")
#         api_key = st.text_input("Enter Gemini API Key:", type="password", help="Your Google Gemini API key")
        
#         if st.button("Configure Agents"):
#             if api_key:
#                 try:
#                     agents = configure_agents(api_key)
#                     st.session_state.agents = agents
#                     st.session_state.agents_configured = True
#                     st.success("✅ Agents configured successfully!")
#                 except Exception as e:
#                     st.error(f"❌ Configuration failed: {str(e)}")
#             else:
#                 st.warning("Please enter your API key")
    
#     # Main content area
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("📊 Portfolio Input")
        
#         # Personal Information
#         st.subheader("Personal Details")
#         age = st.slider("Age", 18, 80, 30)
#         salary = st.number_input("Annual Salary (₹)", min_value=0, value=1200000, step=50000)
#         risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        
#         # Investment Portfolio
#         st.subheader("Current Investment Portfolio")
#         fd_amount = st.number_input("Fixed Deposits (₹)", min_value=0, value=500000, step=10000)
#         mf_amount = st.number_input("Mutual Funds/SIP (₹)", min_value=0, value=300000, step=10000)
#         stocks_amount = st.number_input("Stocks (₹)", min_value=0, value=200000, step=10000)
#         real_estate_amount = st.number_input("Real Estate (₹)", min_value=0, value=1000000, step=50000)
        
#         # Store inputs in session state
#         st.session_state.salary = salary
#         st.session_state.fd_amount = fd_amount
#         st.session_state.mf_amount = mf_amount
#         st.session_state.stocks_amount = stocks_amount
#         st.session_state.real_estate_amount = real_estate_amount
#         st.session_state.age = age
#         st.session_state.risk_tolerance = risk_tolerance
        
#         # Portfolio Summary
#         total_portfolio = fd_amount + mf_amount + stocks_amount + real_estate_amount
#         st.markdown(f'<div class="portfolio-summary">', unsafe_allow_html=True)
#         st.markdown(f"**Total Portfolio Value: ₹{total_portfolio:,}**")
#         st.markdown(f"**Monthly Income: ₹{salary//12:,}**")
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Analysis Button
#         if st.button("🚀 Analyze Portfolio", type="primary", disabled=not st.session_state.agents_configured):
#             run_portfolio_analysis()
    
#     with col2:
#         st.header("📈 Analysis Results")
        
#         if st.session_state.analysis_complete:
#             # Display Strategy
#             strategy = st.session_state.analysis_data.get("strategy", "Growth")
#             st.markdown(f'<div class="agent-box">', unsafe_allow_html=True)
#             st.markdown(f"**🎯 Recommended Strategy: {strategy}**")
#             st.markdown(f"**Reason:** {st.session_state.analysis_data.get('reason', 'Analysis pending')}")
#             st.markdown('</div>', unsafe_allow_html=True)
            
#             # Display Recommendations
#             recommendations = st.session_state.recommendations_data.get("recommendations", [])
#             if recommendations:
#                 st.markdown(f'<div class="recommendation-box">', unsafe_allow_html=True)
#                 st.markdown("**💡 Investment Recommendations:**")
#                 for i, rec in enumerate(recommendations, 1):
#                     st.markdown(f"{i}. {rec}")
#                 st.markdown(f"**Risk Level:** {st.session_state.recommendations_data.get('risk_level', 'Medium')}")
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             # Display Final Report
#             if st.session_state.final_report:
#                 st.markdown(f'<div class="final-report">', unsafe_allow_html=True)
#                 st.markdown("**📋 Comprehensive Financial Report**")
#                 st.markdown(st.session_state.final_report)
#                 st.markdown('</div>', unsafe_allow_html=True)
#         else:
#             st.info("👆 Enter your portfolio details and click 'Analyze Portfolio' to get started!")
            
#             # Show workflow steps
#             st.markdown("**🔄 Analysis Workflow:**")
#             st.markdown("""
#             1. **Portfolio Analysis Agent** - Analyzes current portfolio
#             2. **Strategy Selection** - Determines Growth vs Value approach
#             3. **Investment Agent** - Provides specific recommendations
#             4. **Financial Advisor** - Compiles comprehensive report
#             """)

# if __name__ == "__main__":
#     main()



# app.py

# import streamlit as st
# import json
# import autogen
# from autogen import AssistantAgent, UserProxyAgent
# import time

# # --- Configuration ---
# api_key = "AIzaSyAwqS_3Ag48RxDMgVWL_oVm-iKOJZzSorI"  # Replace with your actual Gemini key

# config_list_gemini = [{
#     "model": "gemini-2.5-flash",
#     "api_key": api_key,
#     "api_type": "google"
# }]

# # --- Define agents ---
# portfolio_analyst = AssistantAgent(
#     name="PortfolioAnalyst",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Analyze the user's portfolio and determine investment strategy. 
#     Output ONLY in JSON format: {"strategy": "Growth" or "Value", "reason": "brief explanation"}
#     """
# )

# growth_strategist = AssistantAgent(
#     name="GrowthStrategist",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Suggest high-growth investments: mid-cap mutual funds, global ETFs, tech stocks, or crypto.
#     Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
#     """
# )

# value_strategist = AssistantAgent(
#     name="ValueStrategist",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Suggest stable investments: bonds, blue-chip stocks, or government schemes.
#     Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
#     """
# )

# financial_advisor = AssistantAgent(
#     name="FinancialAdvisor",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Compile a comprehensive financial report with:
#     1. Portfolio Analysis Summary
#     2. Recommended Strategy
#     3. Specific Investment Recommendations
#     4. Implementation Plan
#     5. Risk Assessment
#     Ensure the report is professional and detailed.
#     """
# )

# user_proxy = UserProxyAgent(
#     name="UserProxy",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=3,
#     is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
#     code_execution_config=False
# )

# # --- Helper ---
# def extract_strategy(content):
#     try:
#         data = json.loads(content.strip())
#         return data.get("strategy", "Growth"), json.dumps(data)
#     except:
#         return "Growth", '{"strategy": "Growth", "reason": "Fallback"}'

# # --- Workflow ---
# def run_analysis(salary, portfolio_text):
#     user_message = f"""
# Hi! I want help managing my investments.
# My salary is ₹{salary}/year.
# My portfolio: 
# {portfolio_text}
# """
#     st.info("🔍 Step 1: Analyzing your portfolio...")
#     analysis_result = user_proxy.initiate_chat(
#         portfolio_analyst,
#         message=user_message,
#         summary_method="last_msg",
#         silent=True
#     )
#     time.sleep(1)
#     strategy, analysis_json = extract_strategy(analysis_result.chat_history[-1]["content"])
#     st.success(f"🎯 Strategy Identified: {strategy}")

#     st.info("📈 Step 2: Getting investment suggestions...")
#     recommender = growth_strategist if strategy == "Growth" else value_strategist
#     recs = user_proxy.initiate_chat(
#         recommender,
#         message=f"Strategy: {strategy}\n{user_message}",
#         summary_method="last_msg",
#         silent=True
#     )
#     time.sleep(1)

#     st.info("📊 Step 3: Compiling financial report...")
#     final = user_proxy.initiate_chat(
#         financial_advisor,
#         message=f"Portfolio Analysis: {analysis_result.chat_history[-1]['content']}\nRecommendations: {recommendations.chat_history[-1]['content']}",
#         summary_method="last_msg",
#         silent=True
#     )
#     return final.summary

# # --- Streamlit UI ---
# st.set_page_config(page_title="AI Investment Advisor", page_icon="💸")

# st.title("💸 AI-Powered Investment Advisor")
# st.markdown("Get personalized financial recommendations based on your salary and portfolio.")

# with st.form("input_form"):
#     salary = st.text_input("Annual Salary (INR)", placeholder="e.g., 1200000")
#     portfolio = st.text_area("Your Portfolio", placeholder="e.g., ₹5L FD, ₹3L Mutual Funds, ₹2L Stocks, ₹10L Real Estate")
#     submitted = st.form_submit_button("Analyze My Investments")

# if submitted:
#     if not salary or not portfolio:
#         st.error("Please fill in both your salary and portfolio.")
#     else:
#         report = run_analysis(salary, portfolio)
#         st.divider()
#         st.markdown("## 💼 Your Personalized Financial Report")
#         st.write(report)



# 💼 Financial Portfolio Manager - Streamlit + AutoGen + Gemini
# import streamlit as st
# import json
# import autogen
# from autogen import AssistantAgent, UserProxyAgent

# # 🗝️ Gemini API Key (You can also set it from secrets)
# api_key = 'AIzaSyAwqS_3Ag48RxDMgVWL_oVm-iKOJZzSorI'  # replace with your Gemini API key
# if not api_key:
#     raise ValueError("Gemini API key missing!")

# # ⚙️ LLM Configuration
# config_list_gemini = [{
#     "model": "gemini-1.5-flash",
#     "api_key": api_key,
#     "api_type": "google"
# }]

# # 💬 Streamlit UI
# st.title("💼 Financial Portfolio Manager")
# st.markdown("AI-powered personalized investment report")

# salary = st.text_input("Enter your annual salary (₹)", placeholder="1200000")
# portfolio = st.text_area("Describe your investment portfolio:", placeholder="- ₹5L Fixed Deposit\n- ₹3L Mutual Funds\n- ₹2L Stocks\n- ₹10L Real Estate")

# run = st.button("Generate Report")

# # 🧠 Setup Agents
# portfolio_analyst = AssistantAgent(
#     name="PortfolioAnalyst",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Analyze the user's portfolio and determine investment strategy. 
#     Output ONLY in JSON format: {"strategy": "Growth" or "Value", "reason": "brief explanation"}
#     """
# )

# growth_strategist = AssistantAgent(
#     name="GrowthStrategist",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Suggest high-growth investments: mid-cap mutual funds, global ETFs, tech stocks, or crypto.
#     Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
#     """
# )

# value_strategist = AssistantAgent(
#     name="ValueStrategist",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Suggest stable investments: bonds, blue-chip stocks, or government schemes.
#     Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
#     """
# )

# financial_advisor = AssistantAgent(
#     name="FinancialAdvisor",
#     llm_config={"config_list": config_list_gemini},
#     system_message="""
#     Compile a comprehensive financial report with:
#     1. Portfolio Analysis Summary
#     2. Recommended Strategy
#     3. Specific Investment Recommendations
#     4. Implementation Plan
#     5. Risk Assessment
#     Ensure the report is professional and detailed.
#     """
# )

# user_proxy = UserProxyAgent(
#     name="UserProxy",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=1,
#     is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
#     code_execution_config=False
# )

# # 🧠 Function to extract JSON
# def extract_strategy(content):
#     try:
#         data = json.loads(content.strip())
#         return data.get("strategy", "Growth")
#     except:
#         return "Growth"

# # 🚀 Workflow Execution
# def manage_investment_portfolio(salary, portfolio_text):
#     message = f"""
#     Hi! I want help managing my investments.
#     My salary is ₹{salary}/year.
#     My portfolio includes:
#     {portfolio_text}
#     """

#     # Step 1: Analyze portfolio
#     analysis_result = user_proxy.initiate_chat(
#         portfolio_analyst,
#         message=message,
#         summary_method="last_msg",
#         silent=True
#     )

#     analysis_summary = analysis_result.chat_history[-1]["content"]
#     strategy = extract_strategy(analysis_summary)

#     # Step 2: Recommend based on strategy
#     agent = growth_strategist if strategy == "Growth" else value_strategist
#     recommendations_result = user_proxy.initiate_chat(
#         agent,
#         message=f"User's portfolio: {portfolio_text}\nStrategy: {strategy}",
#         summary_method="last_msg",
#         silent=True
#     )
#     recommendations_summary = recommendations_result.chat_history[-1]["content"]

#     # Step 3: Generate financial report with full context
#     report_prompt = f"""
# You are a financial advisor. Here is the user's background:

# Salary: ₹{salary}/year  
# Portfolio:  
# {portfolio_text}

# Portfolio Analysis Result:
# {analysis_summary}

# Investment Recommendations:
# {recommendations_summary}

# Using this, create a complete report with the following:
# 1. Summary of portfolio
# 2. Recommended Strategy
# 3. Specific Investment Advice
# 4. Risk Analysis
# 5. Action Plan

# The report should be informative, friendly, and detailed.
# """

#     report_result = user_proxy.initiate_chat(
#         financial_advisor,
#         message=report_prompt,
#         summary_method="last_msg",
#         silent=True
#     )

#     return report_result.chat_history[-1]["content"]
# # ⏳ Run the app logic
# if run:
#     if not salary or not portfolio:
#         st.warning("Please enter both salary and portfolio.")
#     else:
#         with st.spinner("Generating personalized financial report..."):
#             result = manage_investment_portfolio(salary, portfolio)
#             st.subheader("📊 Your Personalized Financial Report")
#             st.markdown(result)



import streamlit as st
import json
import autogen
from autogen import AssistantAgent, UserProxyAgent


import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')  
if not api_key:
    raise ValueError("Gemini API key missing!")


config_list_gemini = [{
    "model": "gemini-2.5-flash",
    "api_key": api_key,
    "api_type": "google"
}]


st.title("💼 Financial Portfolio Manager")
st.markdown("AI-powered personalized investment report")

with st.form("financial_form"):
    salary = st.text_input("Annual Salary (₹)", placeholder="1200000")
    age = st.number_input("Your Age", min_value=18, max_value=100, step=1)
    expenses = st.text_input("Annual Expenses (₹)", placeholder="500000")
    goals = st.text_area("Financial Goals", placeholder="Retirement in 20 years, buying a home in 5 years")
    risk = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])

    st.subheader("🪙 Portfolio Details")
    mutual_funds = st.text_area("Mutual Funds (Name + Type + Amount)", placeholder="Axis Bluechip - Equity - ₹2L")
    stocks = st.text_area("Stocks (Name + Qty + Buy Price)", placeholder="Infosys - 10 shares - ₹1500")
    real_estate = st.text_area("Real Estate (Type + Location + Value)", placeholder="Residential Apartment - Mumbai - ₹10L")
    fixed_deposit = st.text_input("Fixed Deposit (Total ₹)", placeholder="500000")

    submit = st.form_submit_button("Generate Report")


portfolio_analyst = AssistantAgent(
    name="PortfolioAnalyst",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    Analyze the user's portfolio and determine investment strategy. 
    Output ONLY in JSON format: {"strategy": "Growth" or "Value", "reason": "brief explanation"}
    """
)

growth_strategist = AssistantAgent(
    name="GrowthStrategist",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    Suggest high-growth investments: mid-cap mutual funds, global ETFs, tech stocks, or crypto.
    Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
    """
)

value_strategist = AssistantAgent(
    name="ValueStrategist",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    Suggest stable investments: bonds, blue-chip stocks, or government schemes.
    Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
    """
)

financial_advisor = AssistantAgent(
    name="FinancialAdvisor",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    Compile a comprehensive financial report with:
    1. Portfolio Analysis Summary
    2. Recommended Strategy
    3. Specific Investment Recommendations
    4. Implementation Plan
    5. Risk Assessment
    Format the report in Markdown. Add "TERMINATE" at the end when done.
    """
)

user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config=False
)

def extract_strategy(content):
    try:
        data = json.loads(content.strip())
        return data.get("strategy", "Growth")
    except:
        return "Growth"


def manage_investment_portfolio():
    message = f"""
User Profile:
- Age: {age}
- Annual Salary: ₹{salary}
- Annual Expenses: ₹{expenses}
- Risk Tolerance: {risk}
- Financial Goals: {goals}

Current Portfolio:
- Mutual Funds: {mutual_funds or 'None'}
- Stocks: {stocks or 'None'}
- Real Estate: {real_estate or 'None'}
- Fixed Deposit: ₹{fixed_deposit or '0'}
"""

    # Step 1: Portfolio Analysis
    analysis_result = user_proxy.initiate_chat(
        portfolio_analyst,
        message=message,
        summary_method="last_msg",
        silent=True
    )
    analysis_summary = analysis_result.chat_history[-1]["content"]
    strategy = extract_strategy(analysis_summary)

    # Step 2: Get Recommendations
    agent = growth_strategist if strategy == "Growth" else value_strategist
    recommendations_result = user_proxy.initiate_chat(
        agent,
        message=f"{message}\nStrategy: {strategy}",
        summary_method="last_msg",
        silent=True
    )
    recommendations_summary = recommendations_result.chat_history[-1]["content"]

    # Step 3: Generate Final Report
    report_result = user_proxy.initiate_chat(
        financial_advisor,
        message=f"""
Generate a comprehensive financial report based on:

User Profile:
{message}

Portfolio Analysis:
{analysis_summary}

Investment Recommendations:
{recommendations_summary}

Include these sections:
1. Portfolio Analysis Summary
2. Recommended Strategy
3. Specific Investment Recommendations
4. Implementation Plan
5. Risk Assessment
""",
        summary_method="last_msg",
        silent=True
    )

    # Extract the actual report content
    report_content = report_result.chat_history[-1]["content"]
    if "TERMINATE" in report_content:
        return report_content.split("TERMINATE")[0].strip()
    return report_content

# ⏳ Generate and Display
if submit:
    with st.spinner("🧠 Analyzing your portfolio... This may take 1-2 minutes"):
        try:
            result = manage_investment_portfolio()
            st.subheader("📊 Your Personalized Financial Report")
            st.markdown(result)
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.info("Please check your inputs and try again. If the problem persists, try reducing the amount of text in your inputs.")