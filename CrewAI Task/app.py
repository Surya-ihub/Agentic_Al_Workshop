# from crewai import Agent, Task, Crew
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# def logistics_analyst_tools():
#     return []

# def optimization_strategist_tools():
#     return []

# logistics_analyst = Agent(
#     role="Logistics Analyst",
#     goal="Analyze logistics operations to find inefficiencies in delivery routes and inventory turnover.",
#     backstory="A seasoned analyst with years of experience in identifying bottlenecks in supply chain networks.",
#     verbose=True,
#     llm=llm,
#     tools=logistics_analyst_tools()
# )

# optimization_strategist = Agent(
#     role="Optimization Strategist",
#     goal="Design data-driven strategies to optimize logistics operations and improve performance.",
#     backstory="Known for implementing cost-saving logistics strategies using advanced AI models.",
#     verbose=True,
#     llm=llm,
#     tools=optimization_strategist_tools()
# )

# #  Sample product input
# products = ["TV", "Laptops", "Headphones"]

# #  Define tasks
# task1 = Task(
#     description=f"Analyze logistics data for the following products: {products}. Focus on delivery routes and inventory turnover trends.",
#     expected_output="Summary of current inefficiencies and potential improvement areas in logistics operations.",
#     agent=logistics_analyst
# )

# task2 = Task(
#     description="Based on the logistics analyst's findings, develop an optimization strategy to reduce delivery time and improve inventory management.",
#     expected_output="Detailed optimization strategy with action points to improve logistics efficiency.",
#     agent=optimization_strategist
# )


# crew = Crew(
#     agents=[logistics_analyst, optimization_strategist],
#     tasks=[task1, task2],
#     verbose=True
# )

# result = crew.kickoff()


# print("\n\n🔍 Final Optimization Strategy:\n")
# print(result)

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set page configuration
st.set_page_config(page_title="Logistics Optimizer", layout="wide")

# Streamlit UI
st.title("🚚 Logistics Optimization using CrewAI")
st.write("Enter a list of products to analyze logistics and generate optimization strategies.")

# Input form
with st.form("logistics_form"):
    product_input = st.text_input("Enter product names separated by commas", "TV, Laptops, Headphones")
    submitted = st.form_submit_button("Optimize Logistics")

if submitted:
    with st.spinner("Running CrewAI agents..."):

        # Prepare LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # Define tools (empty for now)
        def logistics_analyst_tools():
            return []

        def optimization_strategist_tools():
            return []

        # Define agents
        logistics_analyst = Agent(
            role="Logistics Analyst",
            goal="Analyze logistics operations to find inefficiencies in delivery routes and inventory turnover.",
            backstory="A seasoned analyst with years of experience in identifying bottlenecks in supply chain networks.",
            verbose=True,
            llm=llm,
            tools=logistics_analyst_tools()
        )

        optimization_strategist = Agent(
            role="Optimization Strategist",
            goal="Design data-driven strategies to optimize logistics operations and improve performance.",
            backstory="Known for implementing cost-saving logistics strategies using advanced AI models.",
            verbose=True,
            llm=llm,
            tools=optimization_strategist_tools()
        )

        # Parse product input
        products = [p.strip() for p in product_input.split(",") if p.strip()]

        # Define tasks
        task1 = Task(
            description=f"Analyze logistics data for the following products: {products}. Focus on delivery routes and inventory turnover trends.",
            expected_output="Summary of current inefficiencies and potential improvement areas in logistics operations.",
            agent=logistics_analyst
        )

        task2 = Task(
            description="Based on the logistics analyst's findings, develop an optimization strategy to reduce delivery time and improve inventory management.",
            expected_output="Detailed optimization strategy with action points to improve logistics efficiency.",
            agent=optimization_strategist
        )

        # Create Crew
        crew = Crew(
            agents=[logistics_analyst, optimization_strategist],
            tasks=[task1, task2],
            verbose=True
        )

        # Execute CrewAI workflow
        result = crew.kickoff()

    # Show result
    st.success("✅ Optimization Complete!")
    st.subheader("🔍 Final Optimization Strategy")
    st.markdown(result)
