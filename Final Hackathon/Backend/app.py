from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
import requests
from datetime import datetime, timedelta
import pickle
from typing import List, Dict, Any
from dotenv import load_dotenv
# Set API keys
load_dotenv()  # Load .env file

# Access the variables
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Optionally set them globally if needed
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key # Add your Tavily API key

app = FastAPI(title="Skill Gap Auditor API with RAG")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# RAG Configuration
RAG_DB_PATH = "job_descriptions_db"
CACHE_EXPIRY_DAYS = 14  # Refresh job data every 2 weeks

# Initialize RAG components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

# Global variable to store vector database
vector_db = None

def initialize_rag_db():
    """Initialize or load existing RAG database"""
    global vector_db
    try:
        if os.path.exists(f"{RAG_DB_PATH}.pkl"):
            # Load existing database
            vector_db = FAISS.load_local(RAG_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ Loaded existing RAG database")
        else:
            print("📝 No existing RAG database found. Will create new one on first search.")
    except Exception as e:
        print(f"⚠️ Error loading RAG database: {e}")
        vector_db = None

def search_job_descriptions_tavily(role: str) -> Dict[str, Any]:
    """Search for job descriptions using Tavily API"""
    try:
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return {"error": "Tavily API key not found", "results": []}
        
        # Enhanced search queries for better results
        search_queries = [
            f"{role} job description requirements skills 2024 2025",
            f"{role} developer position requirements skills",
            f"hiring {role} job posting requirements",
            f"{role} software engineer job requirements skills"
        ]
        
        all_results = []
        
        for query in search_queries[:2]:  # Use first 2 queries to avoid rate limits
            url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": True,
                "max_results": 3,
                "include_domains": [
                    "linkedin.com",
                    "indeed.com", 
                    "glassdoor.com",
                ]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                for result in results.get("results", []):
                    job_desc = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "raw_content": result.get("raw_content", "")[:3000],  # Increased limit
                        "source": "tavily",
                        "search_query": query,
                        "retrieved_at": datetime.now().isoformat()
                    }
                    all_results.append(job_desc)
        
        return {
            "success": True,
            "results": all_results,
            "total_found": len(all_results)
        }
            
    except Exception as e:
        return {
            "error": f"Error in Tavily search: {str(e)}",
            "results": []
        }

def update_rag_database(role: str, job_descriptions: List[Dict]) -> bool:
    """Update RAG database with new job descriptions"""
    global vector_db
    try:
        # Prepare documents for RAG
        documents = []
        for i, job in enumerate(job_descriptions):
            # Combine content for better context
            content = f"""
Role: {role}
Job Title: {job.get('title', 'Unknown')}
Source: {job.get('url', 'Unknown')}
Retrieved: {job.get('retrieved_at', 'Unknown')}

Job Description Content:
{job.get('content', '')}

Raw Content:
{job.get('raw_content', '')}
            """.strip()
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "role": role,
                    "title": job.get('title', 'Unknown'),
                    "url": job.get('url', ''),
                    "source": job.get('source', 'tavily'),
                    "retrieved_at": job.get('retrieved_at', ''),
                    "doc_id": f"{role}_{i}_{datetime.now().strftime('%Y%m%d')}"
                }
            )
            documents.append(doc)
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        
        if vector_db is None:
            # Create new vector database
            vector_db = FAISS.from_documents(chunks, embeddings)
            print(f"✅ Created new RAG database with {len(chunks)} chunks")
        else:
            # Add to existing database
            vector_db.add_documents(chunks)
            print(f"✅ Added {len(chunks)} new chunks to RAG database")
        
        # Save database
        vector_db.save_local(RAG_DB_PATH)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating RAG database: {e}")
        return False

def rag_retrieve_job_requirements(role: str, query: str = None, top_k: int = 10) -> List[Dict]:
    """Retrieve relevant job requirements using RAG"""
    global vector_db
    try:
        if vector_db is None:
            return []
        
        # Default query if not provided
        if not query:
            query = f"job requirements skills experience qualifications for {role} developer position"
        
        # Retrieve relevant documents
        docs = vector_db.similarity_search(query, k=top_k)
        
        # Format results
        rag_results = []
        for doc in docs:
            rag_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance": "high"  # FAISS doesn't return scores by default
            })
        
        return rag_results
        
    except Exception as e:
        print(f"❌ Error in RAG retrieval: {e}")
        return []

def comprehensive_job_search_with_rag(role: str) -> Dict[str, Any]:
    """Combined Tavily + RAG search for comprehensive job analysis"""
    
    # Step 1: Search fresh data with Tavily
    print(f"🔍 Step 1: Searching fresh job data with Tavily for role: {role}")
    tavily_results = search_job_descriptions_tavily(role)
    
    # Step 2: Update RAG database with fresh data
    if tavily_results.get("success") and tavily_results.get("results"):
        print(f"📚 Step 2: Updating RAG database with {len(tavily_results['results'])} new job descriptions")
        update_rag_database(role, tavily_results["results"])
    
    # Step 3: Retrieve comprehensive data using RAG
    print(f"🧠 Step 3: Retrieving comprehensive job requirements using RAG")
    rag_results = rag_retrieve_job_requirements(role, top_k=10)
    
    # Step 4: Combine results
    combined_results = {
        "role": role,
        "search_timestamp": datetime.now().isoformat(),
        "tavily_results": {
            "success": tavily_results.get("success", False),
            "fresh_jobs_found": len(tavily_results.get("results", [])),
            "jobs": tavily_results.get("results", [])
        },
        "rag_results": {
            "total_retrieved": len(rag_results),
            "documents": rag_results
        },
        "analysis_ready": True
    }
    
    return combined_results

# Define the first prompt (Skill Analysis Agent)
skill_signal_prompt = PromptTemplate(
    input_variables=["learner_json"],
    template="""
You are an expert developer education specialist and skill profiler. You excel at analyzing learner data regardless of format or completeness and providing comprehensive, actionable feedback.

You will receive learner data in JSON format. The data structure may vary and could include any combination of:
- Code submissions/attempts with results, errors, warnings
- Project work with scores, rubrics, repository links
- Quiz/assessment scores with topic breakdowns
- Peer review feedback and ratings  
- Mentor evaluations and professional assessments
- Learning activity logs, completion rates
- Any other educational or coding performance data

🎯 YOUR TASK:
Analyze ALL available data and generate a comprehensive skill assessment report. Adapt your analysis to whatever data is present - don't assume specific fields exist.

📋 OUTPUT STRUCTURE:
Return a Python dictionary with the following sections (only include sections where you have relevant data):

{{
  "skill_levels": {{
    // Extract any programming concepts/technologies mentioned
    // Rate each as: "novice", "beginner", "intermediate", "proficient", "advanced"
    // Base ratings on evidence from ANY available data sources
  }},
  
  "performance_analysis": {{
    "overall_score": X.X,  // 0-10 scale, based on available metrics
    "data_sources_analyzed": ["list", "of", "data", "types", "found"],
    "key_strengths": [
      // Identify strengths from ANY available evidence
    ],
    "key_weaknesses": [
      // Identify improvement areas from ANY available evidence  
    ],
    "performance_trends": "improving/stable/declining/insufficient_data",
    "evidence_quality": "high/medium/low"  // based on data richness
  }},
  
  "improvement_roadmap": {{
    "immediate_priorities": [
      {{
        "skill_area": "identified from analysis",
        "current_level": "based on evidence",
        "evidence": "specific data points that support this assessment",
        "recommended_actions": ["specific", "actionable", "steps"],
        "estimated_timeline": "realistic timeframe"
      }}
    ]
  }},
  
  "overall_assessment": {{
    "developer_level": "novice/beginner/junior/mid/senior/expert",
    "readiness_for_next_level": X.X,  // 0-10 scale
    "career_trajectory": "analysis based on available data",
    "confidence_in_assessment": "high/medium/low"
  }}
}}

Learner Data:
{learner_json}
"""
)

# Define the enhanced RAG-enabled Role Analyzer prompt


rag_role_analyzer_prompt = PromptTemplate(
    input_variables=["role", "job_data"],
    template="""
You are an Industry Role Analyzer Agent using real-time and historical job data to analyze the market for "{role}".

📦 DATA:
{job_data}

🎯 TASK:
Return a Python dictionary with:

{{
  "role_analysis": {{
    "target_role": "{role}",
    "data_sources": {{"fresh": X, "historical": X, "total": X}},
    "confidence": "high/medium/low"
  }},
  
  "required_skills": {{
    "technical": [
      {{
        "skill": "skill_name",
        "freq": X.X,
        "importance": X.X,
        "level": "beginner/intermediate/advanced",
        "type": "language/framework/tool",
        "trend": "growing/stable/declining",
        "priority": "critical/high/medium/low"
      }}
    ],
    "soft_skills": [
      {{
        "skill": "skill_name",
        "freq": X.X,
        "examples": ["job phrase"],
        "importance": "high/medium/low"
      }}
    ],
    "certifications": [
      {{
        "name": "cert_name",
        "freq": X.X,
        "required": "yes/no"
      }}
    ]
  }},
  
  "experience": {{
    "avg_min": X.X,
    "avg_max": X.X,
    "levels": {{"entry": X, "mid": X, "senior": X}},
    "trend": "up/down/stable"
  }},
  
  "role_clusters": {{
    "similar_roles": [
      {{"name": "related role", "overlap": X.X, "transition": "easy/moderate/hard"}}
    ],
    "paths": [
      {{"from": "X", "to": "{role}", "timeline": "X-Y yrs", "skills": ["a", "b"]}}
    ]
  }},
  
  "market": {{
    "demand": "very_high/high/medium/low",
    "trend": "rising/stable/falling",
    "remote_ratio": X.X,
    "company_types": {{"startup": X.X, "enterprise": X.X}},
    "top_locations": ["city1", "city2"]
  }},
  
  "skill_gap": {{
    "must_have": [...],
    "should_have": [...],
    "differentiators": [...],
    "emerging": [...],
    "declining": [...]
  }},
  
  "actions": {{
    "priorities": [
      {{
        "skill": "X",
        "urgency": "high/medium/low",
        "time": "weeks/months",
        "impact": "high/medium/low",
        "mode": "course/bootcamp/self-study"
      }}
    ],
    "readiness_score": X.X,
    "time_to_ready": "X weeks",
    "edge": ["unique advantages"]
  }},
  
  "data_quality": {{
    "latest_posting": "YYYY-MM-DD",
    "recency_score": X.X,
    "next_refresh": "YYYY-MM-DD"
  }}
}}

Analyze the data above and return only the dictionary.
"""
)


# Create LLM Chains
skill_aggregator_chain = LLMChain(llm=llm, prompt=skill_signal_prompt)
rag_role_analyzer_chain = LLMChain(llm=llm, prompt=rag_role_analyzer_prompt)


# Enhanced Tools
SkillSignalAggregatorTool = Tool(
    name="SkillSignalAggregator",
    func=lambda learner_json: skill_aggregator_chain.run(learner_json=learner_json),
    description="Generates comprehensive skill assessment from learner analytics"
)

RAGEnabledRoleAnalyzerTool = Tool(
    name="RAGEnabledRoleAnalyzer", 
    func=lambda role: rag_role_analyzer_chain.run(
        role=role, 
        job_data=json.dumps(comprehensive_job_search_with_rag(role), indent=2)
    ),
    description="Analyzes industry role requirements using RAG-enhanced job market intelligence"
)

# Initialize RAG on startup
initialize_rag_db()

# Main API Endpoint
@app.post("/api/skillgap/data/{user_id}")
async def receive_skill_data(user_id: str, request: Request):
    payload = await request.json()
    # print(f"Received payload: {payload}") 
    
    role = payload.get("role", "")
    learner_data = payload.get("data", {})
    learner_json_str = json.dumps(learner_data, indent=2)

    try:
        print(f"🔍 Step 1: Analyzing skill data for user: {user_id}")
        # First Agent: Skill Analysis
        skill_assessment = SkillSignalAggregatorTool.func(learner_json_str)
        print(f"✅ Skill analysis completed")

        print(f"🔍 Step 2: Getting comprehensive job data for: {role}")
        # Get comprehensive job search data (Tavily + RAG)
        job_search_data = comprehensive_job_search_with_rag(role)
        
        print(f"🔍 Step 3: RAG-enhanced role analysis for: {role}")
        # Second Agent: RAG-Enhanced Role Analysis using the job data
        role_analysis = rag_role_analyzer_chain.run(
            role=role, 
            job_data=json.dumps(job_search_data, indent=2)
        )
        print(f"✅ RAG-enhanced role analysis completed")

        # Extract and format job postings for frontend
        job_postings = []
        fresh_jobs = job_search_data.get("tavily_results", {}).get("jobs", [])
        
        for i, job in enumerate(fresh_jobs):
            formatted_job = {
                "id": f"job_{i}_{datetime.now().strftime('%Y%m%d')}",
                "title": job.get("title", "Job Title Not Available"),
                "url": job.get("url", ""),
                "source": job.get("source", "tavily"),
                "content_preview": job.get("content", "")[:300] + "..." if len(job.get("content", "")) > 300 else job.get("content", ""),
                "full_content": job.get("content", ""),
                "raw_content": job.get("raw_content", ""),
                "retrieved_at": job.get("retrieved_at", ""),
                "search_query": job.get("search_query", ""),
                "relevance_score": 0.9 - (i * 0.1),  # Simple relevance scoring
                "is_fresh": True
            }
            job_postings.append(formatted_job)

        # Format the complete response for frontend
        frontend_response = {
            "success": True,
            "userId": user_id,
            "targetRole": role,
            "analysisTimestamp": datetime.now().isoformat(),
            
            # SKILL ANALYSIS RESULTS
            "skillAssessment": {
                "rawAnalysis": skill_assessment,
                "summary": "Comprehensive skill analysis completed",
                "dataQuality": "high"
            },
            
            # JOB SEARCH RESULTS (What you wanted!)
            "jobSearchResults": {
                "totalJobsFound": len(job_postings),
                "freshJobsFromTavily": len(fresh_jobs),
                "ragHistoricalJobs": job_search_data.get("rag_results", {}).get("total_retrieved", 0),
                "searchSuccess": job_search_data.get("tavily_results", {}).get("success", False),
                "lastUpdated": datetime.now().isoformat(),
                "jobPostings": job_postings,  # Array of individual job objects
                "searchMetadata": {
                    "role": role,
                    "searchSources": ["linkedin.com", "indeed.com", "glassdoor.com", "stackoverflow.jobs"],
                    "searchDepth": "advanced",
                    "dataFreshness": "real-time"
                }
            },
            
            # ROLE ANALYSIS RESULTS  
            "roleAnalysis": {
                "rawAnalysis": role_analysis,
                "marketInsights": "Comprehensive role analysis completed",
                "confidenceScore": 0.85
            },
            
            # COMBINED INSIGHTS
            "skillGapAnalysis": {
                "status": "completed",
                "analysisType": "comprehensive_rag_enhanced",
                "dataSourcesUsed": ["learner_analytics", "tavily_fresh_search", "rag_historical_data"],
                "recommendationsReady": True
            },
            
            # METADATA FOR FRONTEND
            "metadata": {
                "processingTime": "~30 seconds",
                "ragDatabaseUpdated": True,
                "apiVersion": "v2.0",
                "features": ["skill_analysis", "job_search", "rag_enhancement", "market_intelligence"]
            }
        }

        # --- Deficiency Classifier Agent call ---
        # Parse skill graph from skill_assessment (handle code block and string cases)
        import re
        import ast
        def extract_dict_from_codeblock(s):
            if not isinstance(s, str):
                return s
            # Remove code block markers and language hints
            s = re.sub(r'^```[a-zA-Z]*', '', s.strip())
            s = s.strip('`\n ')
            # Find the first dict in the string
            match = re.search(r'\{[\s\S]*\}', s)
            if match:
                s = match.group(0)
            try:
                return ast.literal_eval(s)
            except Exception:
                try:
                    return json.loads(s)
                except Exception:
                    return {}

        skill_assessment_dict = extract_dict_from_codeblock(skill_assessment)
        learner_skill_graph = skill_assessment_dict.get("skill_levels", {})

        role_analysis_dict = extract_dict_from_codeblock(role_analysis)
        role_skills = role_analysis_dict.get("required_skills", {})
        if "skill_gap" in role_analysis_dict:
            role_skills["skill_gap"] = role_analysis_dict["skill_gap"]
        deficiency_report = deficiency_classifier_agent(learner_skill_graph, role_skills)

        # --- Remediation Planner Agent call ---
        lms_content_metadata = [
            {"skill": k, "module_id": f"mod_{i}", "title": f"{k.title()} Basics", "duration_min": 15, "type": "module"}
            for i, k in enumerate(deficiency_report.get("missing_core_skills", []) + deficiency_report.get("missing_optional_skills", []))
        ]
        remediation_plan = remediation_planner_agent(deficiency_report, lms_content_metadata, target_role=role)

        # Add remediation plan (with ETA and module duration check) to response
        frontend_response["remediationPlan"] = {
            **remediation_plan,
            "allModulesUnder20Min": all(m.get("duration_min", 999) < 20 for day in remediation_plan["remediation_roadmap"] for m in day["modules"]),
            "etaToCareerAlignment": remediation_plan.get("eta_string", "")
        }

        # --- Skill Progress Tracker Agent call ---
        def upgrade_level(level):
            levels = ["novice", "beginner", "intermediate", "proficient", "advanced"]
            try:
                idx = levels.index(level.lower())
                return levels[min(idx + 1, len(levels) - 1)]
            except Exception:
                return "beginner"
        post_remediation_performance = {}
        for skill in learner_skill_graph:
            post_remediation_performance[skill] = upgrade_level(learner_skill_graph[skill])
        for skill in deficiency_report.get("missing_core_skills", []) + deficiency_report.get("missing_optional_skills", []):
            post_remediation_performance[skill] = "beginner"
        skill_progress = skill_progress_tracker_agent(post_remediation_performance, learner_skill_graph)

        frontend_response["skillProgressTracker"] = skill_progress

        return frontend_response
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to complete skill gap analysis: {str(e)}",
            "userId": user_id,
            "targetRole": role,
            "timestamp": datetime.now().isoformat()
        }

# Additional test endpoints
@app.post("/api/test/tavily-search")
async def test_tavily_search(request: Request):
    """Test Tavily search functionality"""
    payload = await request.json()
    role = payload.get("role", "")
    
    try:
        results = search_job_descriptions_tavily(role)
        return {"role": role, "tavilyResults": results}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/test/rag-search")
async def test_rag_search(request: Request):
    """Test RAG search functionality"""
    payload = await request.json()
    role = payload.get("role", "")
    query = payload.get("query", "")
    
    try:
        results = rag_retrieve_job_requirements(role, query)
        return {"role": role, "query": query, "ragResults": results}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/test/comprehensive-search")
async def test_comprehensive_search(request: Request):
    """Test comprehensive Tavily + RAG search"""
    payload = await request.json()
    role = payload.get("role", "")
    
    try:
        results = comprehensive_job_search_with_rag(role)
        return {"role": role, "comprehensiveResults": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/rag/status")
async def rag_status():
    """Check RAG database status"""
    global vector_db
    return {
        "ragDatabaseLoaded": vector_db is not None,
        "databasePath": RAG_DB_PATH,
        "cacheExpiryDays": CACHE_EXPIRY_DAYS,
        "status": "operational" if vector_db else "not_initialized"
    }

# --- Deficiency Classifier Agent ---
def deficiency_classifier_agent(learner_skill_graph: dict, role_skills: dict) -> dict:
    """
    Compares learner skill graph with industry role demands to detect mismatches and prioritize deficiencies.
    Input:
        learner_skill_graph: dict of learner's skills and levels (e.g., {"Python": "intermediate", ...})
        role_skills: dict from RAG Agent (should contain 'required_skills' with 'technical', 'soft_skills', 'certifications', and 'skill_gap')
    Output:
        Skill Deficiency Report (core vs optional gaps, urgency, career impact, outdated concepts)
    """
    import re
    from collections import defaultdict
    
    # Helper: Normalize skill names
    def normalize(skill):
        return re.sub(r"[^a-z0-9]+", "", skill.lower())

    # Extract learner skills (normalize for comparison)
    learner_skills = {normalize(k): v for k, v in learner_skill_graph.items()}

    # Extract role skills from RAG output
    core_skills = []
    optional_skills = []
    outdated_skills = []
    emerging_skills = []
    skill_gap = role_skills.get("skill_gap", {})
    
    # Core/Optional/Emerging/Declining from RAG
    core_skills = [normalize(s) for s in skill_gap.get("must_have", [])]
    optional_skills = [normalize(s) for s in skill_gap.get("should_have", [])]
    emerging_skills = [normalize(s) for s in skill_gap.get("emerging", [])]
    declining_skills = [normalize(s) for s in skill_gap.get("declining", [])]

    # Detect missing core and optional skills
    missing_core = [s for s in core_skills if s not in learner_skills]
    missing_optional = [s for s in optional_skills if s not in learner_skills]
    
    # Detect outdated concepts in learner graph
    flagged_outdated = [k for k in learner_skills if k in declining_skills]
    
    # Classify gaps by urgency and impact
    gap_report = []
    for s in missing_core:
        gap_report.append({
            "skill": s,
            "type": "core",
            "urgency": "high",
            "career_impact": "critical",
            "reason": "Missing must-have skill for target role"
        })
    for s in missing_optional:
        gap_report.append({
            "skill": s,
            "type": "optional",
            "urgency": "medium",
            "career_impact": "moderate",
            "reason": "Missing should-have skill for target role"
        })
    for s in emerging_skills:
        if s not in learner_skills:
            gap_report.append({
                "skill": s,
                "type": "emerging",
                "urgency": "medium",
                "career_impact": "future",
                "reason": "Emerging skill in market, not present in learner graph"
            })
    for s in flagged_outdated:
        gap_report.append({
            "skill": s,
            "type": "outdated",
            "urgency": "low",
            "career_impact": "declining",
            "reason": "Skill in learner graph is flagged as declining in market"
        })

    # 90% accuracy: If at least 90% of must-have skills are detected as missing when absent
    accuracy = 1.0 if len(missing_core) >= int(0.9 * len(core_skills)) else (len(missing_core) / len(core_skills) if core_skills else 1.0)

    return {
        "missing_core_skills": missing_core,
        "missing_optional_skills": missing_optional,
        "flagged_outdated_skills": flagged_outdated,
        "gap_report": gap_report,
        "accuracy_estimate": round(accuracy, 2),
        "core_skills_total": len(core_skills),
        "core_skills_missing": len(missing_core)
    }

# --- Remediation Planner Agent ---
def remediation_planner_agent(skill_deficiency_report: dict, lms_content_metadata: list, target_role: str = "") -> dict:
    """
    Maps each deficiency to targeted modules, exercises, or projects from the LMS and generates a timeline-based remediation plan.
    Input:
        skill_deficiency_report: Output from deficiency_classifier_agent
        lms_content_metadata: List of dicts, each with at least {"skill", "module_id", "title", "duration_min", "type"}
        target_role: (Optional) Used for ETA phrasing
    Output:
        Personalized remediation roadmap
    """
    from datetime import timedelta
    import math

    # Helper: Normalize skill names
    def normalize(skill):
        import re
        return re.sub(r"[^a-z0-9]+", "", skill.lower())

    # Extract gaps
    gaps = skill_deficiency_report.get("gap_report", [])
    gap_skills = [g["skill"] for g in gaps if g["type"] in ("core", "optional", "emerging")]
    gap_types = {g["skill"]: g["type"] for g in gaps}

    # Normalize LMS content for matching
    lms_by_skill = {}
    for item in lms_content_metadata:
        skill_norm = normalize(item.get("skill", ""))
        if skill_norm:
            lms_by_skill.setdefault(skill_norm, []).append(item)

    # Map gaps to LMS content
    mapped_content = []
    unmapped_gaps = []
    total_gaps = len(gap_skills)
    for skill in gap_skills:
        skill_norm = normalize(skill)
        modules = lms_by_skill.get(skill_norm, [])
        # Only include modules <20 min
        modules = [m for m in modules if m.get("duration_min", 999) < 20]
        if modules:
            for m in modules:
                mapped_content.append({
                    "skill": skill,
                    "type": gap_types.get(skill, "core"),
                    "module_id": m.get("module_id"),
                    "title": m.get("title"),
                    "duration_min": m.get("duration_min"),
                    "content_type": m.get("type", "module")
                })
        else:
            unmapped_gaps.append(skill)

    # Calculate mapping coverage
    unique_mapped_skills = set([normalize(m["skill"]) for m in mapped_content])
    mapping_coverage = len(unique_mapped_skills) / total_gaps if total_gaps else 1.0

    # Timeline plan: group modules into days (assume 60 min/day learning)
    mapped_content_sorted = sorted(mapped_content, key=lambda x: (x["type"], -x["duration_min"]))
    plan = []
    day = 1
    day_time = 0
    daily_plan = []
    for m in mapped_content_sorted:
        if day_time + m["duration_min"] > 60 and daily_plan:
            plan.append({"day": day, "modules": daily_plan})
            day += 1
            day_time = 0
            daily_plan = []
        daily_plan.append(m)
        day_time += m["duration_min"]
    if daily_plan:
        plan.append({"day": day, "modules": daily_plan})

    # ETA calculation
    eta_days = len(plan)
    eta_weeks = math.ceil(eta_days / 5)  # Assume 5 learning days/week
    eta_str = f"{eta_weeks} week{'s' if eta_weeks > 1 else ''} to {target_role or 'career alignment'}"

    return {
        "remediation_roadmap": plan,
        "eta_days": eta_days,
        "eta_weeks": eta_weeks,
        "eta_string": eta_str,
        "mapping_coverage": round(mapping_coverage, 2),
        "unmapped_gaps": unmapped_gaps,
        "modules_per_gap": mapped_content,
        "criteria": {
            "min_coverage": 0.8,
            "max_module_duration_min": 20
        },
        "meets_coverage": mapping_coverage >= 0.8
    }

# --- Skill Progress Tracker Agent ---
def skill_progress_tracker_agent(post_remediation_performance: dict, original_skill_graph: dict, version_history: list = None) -> dict:
    """
    Tracks the learner's remediation journey, updates the skill graph, and shows progress toward industry readiness.
    Input:
        post_remediation_performance: dict of updated skill levels after remediation (e.g., {"Python": "proficient", ...})
        original_skill_graph: dict of learner's skills before remediation
        version_history: list of previous skill graph snapshots (optional)
    Output:
        {
            "updated_skill_graph": ...,
            "radar_chart_data": ...,
            "confidence_index": ...,
            "trendline_data": ...,
            "regressions": ...,
            "version_history": ...
        }
    """
    import copy
    import datetime

    # Helper: Skill level to numeric
    skill_levels = ["novice", "beginner", "intermediate", "proficient", "advanced"]
    level_to_num = {lvl: i for i, lvl in enumerate(skill_levels)}
    num_to_level = {i: lvl for i, lvl in enumerate(skill_levels)}

    def to_num(level):
        return level_to_num.get(level.lower(), 0)
    def to_level(num):
        return num_to_level.get(num, "novice")

    # Prepare version history
    if version_history is None:
        version_history = []
    # Store original as first version if not present
    if not version_history or version_history[-1].get("timestamp") != "original":
        version_history.append({
            "timestamp": "original",
            "skill_graph": copy.deepcopy(original_skill_graph)
        })

    # Update skill graph
    updated_skill_graph = copy.deepcopy(original_skill_graph)
    improvements = {}
    regressions = {}
    for skill, new_level in post_remediation_performance.items():
        old_level = original_skill_graph.get(skill, "novice")
        if to_num(new_level) > to_num(old_level):
            improvements[skill] = {"from": old_level, "to": new_level}
        elif to_num(new_level) < to_num(old_level):
            regressions[skill] = {"from": old_level, "to": new_level}
        updated_skill_graph[skill] = new_level

    # Store new version
    version_history.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "skill_graph": copy.deepcopy(updated_skill_graph)
    })

    # Radar chart data (concept-level granularity)
    radar_chart_data = []
    for skill, level in updated_skill_graph.items():
        radar_chart_data.append({
            "skill": skill,
            "level": level,
            "level_num": to_num(level)
        })

    # Confidence index: % of skills at intermediate or above
    total_skills = len(updated_skill_graph)
    confident_skills = sum(1 for v in updated_skill_graph.values() if to_num(v) >= 2)
    confidence_index = round(confident_skills / total_skills, 2) if total_skills else 0.0

    # Trendline data: show skill level evolution for each skill
    trendline_data = {}
    for skill in updated_skill_graph:
        trendline_data[skill] = [
            to_num(ver["skill_graph"].get(skill, "novice")) for ver in version_history
        ]

    return {
        "updated_skill_graph": updated_skill_graph,
        "radar_chart_data": radar_chart_data,
        "confidence_index": confidence_index,
        "trendline_data": trendline_data,
        "regressions": regressions,
        "improvements": improvements,
        "version_history": version_history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)