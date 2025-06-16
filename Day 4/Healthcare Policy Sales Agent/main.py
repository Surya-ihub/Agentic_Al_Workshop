import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load policy data
with open("policy_data.json") as f:
    policy_data = json.load(f)

# Define FastAPI app
app = FastAPI()

# Define request model
class UserInput(BaseModel):
    age: int
    family_type: str
    dependents: int
    special_requirements: str

# Define the recommend endpoint
@app.post("/recommend")
async def recommend_policy(user_input: UserInput):
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are a healthcare insurance advisor. Suggest the best insurance policy from the list below based on user details.

User Details:
- Age: {user_input.age}
- Family Type: {user_input.family_type}
- Dependents: {user_input.dependents}
- Special Requirements: {user_input.special_requirements}

Available Policies:
{json.dumps(policy_data, indent=2)}

Provide a recommendation with bullet points explaining the match.
    """

    response = model.generate_content(prompt)
    return {"recommendation": response.text.strip()}
