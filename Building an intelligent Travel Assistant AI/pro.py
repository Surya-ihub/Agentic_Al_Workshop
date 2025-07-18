import os
import requests
import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration
WEATHER_API_KEY = "7dc5e1a4e83d480b98341450251507"
GEMINI_API_KEY = "AIzaSyDG-0xIaprzdT70VTf-LnMt62_s-F8SJqA"

# Initialize LLM - Using Gemini Pro
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GEMINI_API_KEY
)

# ----------------------
# Enhanced Custom Tools
# ----------------------

@tool
def get_current_weather(location: str) -> str:
    """Get current weather information for a given location."""
    try:
        # Use WeatherAPI with proper error handling
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}&aqi=no"
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            # Fallback to OpenWeatherMap if WeatherAPI fails
            owm_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid=9b929a9d9c3f5a5e610cbf9163121d1e&units=metric"
            owm_response = requests.get(owm_url)
            owm_data = owm_response.json()
            
            if owm_data.get("cod") != 200:
                return f"Error: {data.get('error', {}).get('message', 'Weather data not available')}"
            
            return (f"Weather in {location}: "
                    f"{owm_data['main']['temp']}°C, "
                    f"{owm_data['weather'][0]['description']}. "
                    f"Feels like {owm_data['main']['feels_like']}°C. "
                    f"Humidity: {owm_data['main']['humidity']}%, "
                    f"Wind: {owm_data['wind']['speed']} m/s")
        
        return (f"Weather in {location}: "
                f"{data['current']['temp_c']}°C, "
                f"{data['current']['condition']['text']}. "
                f"Feels like {data['current']['feelslike_c']}°C. "
                f"Humidity: {data['current']['humidity']}%, "
                f"Wind: {data['current']['wind_kph']} km/h")
    except Exception as e:
        return f"Weather data unavailable: {str(e)}"

@tool
def get_top_attractions(location: str) -> str:
    """Get top tourist attractions for a given location."""
    try:
        # Use DuckDuckGo with enhanced query
        search = DuckDuckGoSearchAPIWrapper()
        query = f"top 10 tourist attractions in {location} with descriptions"
        raw_results = search.run(query)
        
        # Use LLM to summarize the raw search results
        summary_prompt = (
            f"Identify and summarize the top 5 attractions in {location} based on this information:"
            f"\n\n{raw_results}\n\n"
            "For each attraction, provide:"
            "\n- Name"
            "\n- Type (temple, beach, museum, etc.)"
            "\n- Brief description (1 sentence)"
            "\n- Why it's worth visiting"
            "\nFormat as:"
            "\n1. **Name** (Type) - Description. Why visit: ..."
        )
        
        return llm.invoke(summary_prompt).content
    except Exception as e:
        return f"Attractions search failed: {str(e)}"

@tool
def get_accommodation_recommendations(location: str, travel_style: str) -> str:
    """Get accommodation recommendations based on travel style."""
    try:
        # Use DuckDuckGo with enhanced query
        search = DuckDuckGoSearchAPIWrapper()
        query = f"best {travel_style.lower()} hotels and stays in {location}"
        raw_results = search.run(query)
        
        # Use LLM to summarize the results
        summary_prompt = (
            f"Recommend 3-5 accommodation options in {location} for {travel_style} travelers:"
            f"\n\n{raw_results}\n\n"
            "For each option, provide:"
            "\n- Name"
            "\n- Type (hotel, resort, homestay, etc.)"
            "\n- Key features"
            "\n- Why it's good for {travel_style} travel"
            "\nFormat as:"
            "\n1. **Name** (Type) - Features. Why good for {travel_style}: ..."
        )
        
        return llm.invoke(summary_prompt).content
    except Exception as e:
        return f"Accommodation search failed: {str(e)}"

# ----------------------
# Robust Agent Setup
# ----------------------

# Define tools for the agent
tools = [get_current_weather, get_top_attractions, get_accommodation_recommendations]

# Create the agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful travel assistant. Respond to user requests by:"
     "\n1. FIRST providing current weather"
     "\n2. THEN listing top attractions"
     "\n3. THEN giving accommodation recommendations"
     "\n4. FINALLY offering personalized travel tips"
     "\n\nStructure your response clearly with:"
     "\n### Weather Report ☀️🌧️❄️"
     "\n[weather details]"
     "\n\n### Top Attractions 🏰🏞️🎭"
     "\n[attractions list]"
     "\n\n### Where to Stay 🛏️🏨"
     "\n[accommodation options]"
     "\n\n### Travel Tips 💡"
     "\n[personalized tips]"
     "\n\nUse emojis and keep information concise. If any tool fails, skip that section and explain."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    return_intermediate_steps=True
)

# ----------------------
# Enhanced Streamlit UI
# ----------------------

st.set_page_config(page_title="🌍 Travel Assistant AI", page_icon="✈️", layout="wide")
st.title("🌍 Intelligent Travel Assistant")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Travel Preferences")
    location = st.text_input("Destination", "Chennai")
    travel_style = st.selectbox("Travel Style", 
                               ["Adventure", "Relaxation", "Cultural", "Foodie", "Family", "Business", "Spiritual"])
    travel_days = st.slider("Trip Duration (days)", 1, 14, 3)
    budget = st.selectbox("Budget", ["Budget", "Mid-range", "Luxury"])
    
    if st.button("🧹 Clear History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Where would you like to travel today? 🌎"}]

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about travel destinations..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("✈️ Planning your trip..."):
            try:
                # Get travel information
                response = agent_executor.invoke(
                    {"input": f"Provide comprehensive travel information for {location} "
                              f"for a {travel_days}-day {travel_style.lower()} trip with {budget} budget"}
                )
                
                # Display response
                full_response = st.empty()
                full_response.markdown(response["output"])
                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response["output"]}
                )
                
                # Show debug info in expander
                with st.expander("🔍 Debug Details"):
                    st.write("**Intermediate Steps:**")
                    for step in response.get("intermediate_steps", []):
                        st.json(step)
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# Add footer
st.divider()
st.caption("""
    Powered by: 
    <img src="https://www.gstatic.com/lamda/images/gemini_sparkle_resting_v2_darkmode_2d4785dff9491523d32a.svg" width="20"> Google Gemini | 
    <img src="https://langchain.com/img/brand/theme-image.png" width="20"> LangChain | 
    WeatherAPI.com
    """, unsafe_allow_html=True)

# Add some styling
st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stSpinner > div > div {
        border-top-color: #ff4b4b;
    }
    .st-b7 {
        background-color: #f0f2f6;
    }
    .st-c0 {
        background-color: #ffffff;
    }
    .st-emotion-cache-4oy321 {
        background-color: #f0f5ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)