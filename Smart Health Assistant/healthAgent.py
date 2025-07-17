
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import google.generativeai as genai

# === Streamlit UI ===
st.set_page_config(page_title="Smart Health Assistant", layout="wide")
st.title("🤖 Smart Health Assistant")

with st.sidebar:
    st.header("Configuration")
    gemini_api_key = st.text_input("Enter Gemini 1.5 Flash API Key:", type="password")
    st.markdown("[Get Gemini API Key](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.caption("This assistant calculates BMI, provides health recommendations, creates meal plans, and generates workout schedules based on your inputs.")

# === Session State ===
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "final_plan" not in st.session_state:
    st.session_state.final_plan = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Utility: Gemini Config Wrapper ===
def get_gemini_config(api_key: str, model: str = "gemini-1.5-flash"):
    return [{
        "model": model,
        "api_key": api_key,
        "api_type": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta"
    }]

# === BMI Tool ===
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

# === Health Form ===
with st.form("health_form"):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        dietary_preference = st.selectbox("Dietary Preference", ["Veg", "Non-Veg", "Vegan"])
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Generate Health Plan")

# === Agent Initialization ===
def init_agents(api_key):
    genai.configure(api_key=api_key)
    config_list = get_gemini_config(api_key)

    bmi_agent = AssistantAgent(
        name="BMI_Agent",
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message="""You are a BMI specialist. Analyze BMI results and:
        1. Calculate BMI from weight (kg) and height (cm)
        2. Categorize (underweight, normal, overweight, obese)
        3. Provide health recommendations
        Always include the exact BMI value in your response."""
    )

    diet_agent = AssistantAgent(
        name="Diet_Planner",
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message=f"""You are a nutritionist. Create meal plans based on:
        1. BMI analysis from BMI_Agent
        2. Dietary preference ({dietary_preference})
        Include breakfast, lunch, dinner, and snacks with portions."""
    )

    workout_agent = AssistantAgent(
        name="Workout_Scheduler",
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message=f"""You are a fitness trainer. Create weekly workout plans based on:
        1. Age ({age}) and gender ({gender})
        2. BMI recommendations
        3. Meal plan from Diet_Planner
        Include cardio, strength training with duration and intensity."""
    )

    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message="Collects and shares user data with other agents."
    )

    user_proxy.register_function(function_map={"calculate_bmi": calculate_bmi})

    return user_proxy, bmi_agent, diet_agent, workout_agent, config_list

# === Submit Handler ===
if submit_btn and gemini_api_key:
    try:
        user_proxy, bmi_agent, diet_agent, workout_agent, config_list = init_agents(gemini_api_key)

        groupchat = GroupChat(
            agents=[user_proxy, bmi_agent, diet_agent, workout_agent],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config={"config_list": config_list, "cache_seed": None}
        )

        initial_message = f"""
        User Health Profile:
        - Basic Information:
          • Weight: {weight} kg
          • Height: {height} cm
          • Age: {age}
          • Gender: {gender}
        - Preferences:
          • Dietary Preference: {dietary_preference}

        Please proceed with the health assessment in this sequence:
        1. Calculate BMI using the 'calculate_bmi' function with weight={weight} and height={height}
        2. Analyze BMI and provide recommendations
        3. Create a meal plan based on BMI analysis and dietary preference
        4. Develop a workout schedule based on age, gender, and meal plan
        """

        with st.spinner("Generating your personalized health plan..."):
            user_proxy.initiate_chat(
                manager,
                message=initial_message,
                clear_history=True
            )

            st.session_state.conversation = []
            for msg in groupchat.messages:
                if msg['role'] != 'system' and msg['content'].strip():
                    st.session_state.conversation.append((msg['name'], msg['content']))
                    if msg['name'] == "Workout_Scheduler":
                        st.session_state.final_plan = msg['content']

        st.success("Health plan generated successfully!")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.info("Please ensure: 1) Valid API key 2) Stable internet connection 3) Correct input values")

# === Results Display ===
if st.session_state.conversation:
    st.divider()
    st.subheader("Health Plan Generation Process")

    for agent, message in st.session_state.conversation:
        with st.expander(f"{agent} says:"):
            st.markdown(message)

    st.divider()
    st.subheader("🌟 Your Complete Health Plan")

    if st.session_state.final_plan:
        st.markdown(st.session_state.final_plan)
        st.download_button(
            label="Download Health Plan",
            data=st.session_state.final_plan,
            file_name="personalized_health_plan.txt",
            mime="text/plain"
        )
    else:
        st.warning("Workout schedule not generated. Please try again.")

elif not submit_btn:
    st.divider()
    st.info("""
    **Instructions:**
    1. Enter your Gemini API key in the sidebar
    2. Fill in your health details
    3. Click "Generate Health Plan"
    4. View your personalized recommendations
    """)
