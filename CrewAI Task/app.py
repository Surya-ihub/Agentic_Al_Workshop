from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def logistics_analyst_tools():
    return []

def optimization_strategist_tools():
    return []

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

#  Sample product input
products = ["TV", "Laptops", "Headphones"]

#  Define tasks
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


crew = Crew(
    agents=[logistics_analyst, optimization_strategist],
    tasks=[task1, task2],
    verbose=True
)

result = crew.kickoff()


print("\n\n🔍 Final Optimization Strategy:\n")
print(result)
