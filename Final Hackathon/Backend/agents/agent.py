import json
from typing import Dict, List, Any, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
import os

# Set up Google AI API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA5DKab1Qb-rKh0fgbswrMkMb3HN05fz9c"

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=2048
)

# Define the state structure for LangGraph
class SkillAggregatorState(TypedDict):
    learner_data: Dict[str, Any]
    code_analysis: Dict[str, Any]
    project_analysis: Dict[str, Any]
    quiz_analysis: Dict[str, Any]
    peer_review_analysis: Dict[str, Any]
    mentor_analysis: Dict[str, Any]
    aggregated_skills: Dict[str, Any]
    final_skill_graph: Dict[str, Any]
    processing_status: str

# Sample learner data - this simulates what would come from your LMS
sample_learner_data = {
    "learner_id": "learner_001",
    "name": "John Doe",
    "course": "Full Stack Web Development",
    "data_sources": {
        "code_submissions": [
            {
                "assignment_id": "html_basics_001",
                "code": """
<!DOCTYPE html>
<html>
<head>
    <title>My Portfolio</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        .header { background: #333; color: white; padding: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Welcome to My Portfolio</h1>
    </div>
    <p>This is my first webpage!</p>
</body>
</html>
                """,
                "submission_date": "2024-01-15",
                "concept_tags": ["HTML structure", "CSS styling", "semantic markup"],
                "instructor_feedback": "Good basic structure, but missing semantic HTML5 elements"
            },
            {
                "assignment_id": "js_functions_002",
                "code": """
function calculateSum(a, b) {
    return a + b;
}

function findMax(arr) {
    let max = arr[0];
    for(let i = 1; i < arr.length; i++) {
        if(arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Usage
console.log(calculateSum(5, 3));
console.log(findMax([1, 5, 3, 9, 2]));
                """,
                "submission_date": "2024-01-20",
                "concept_tags": ["JavaScript functions", "loops", "arrays", "variable declaration"],
                "instructor_feedback": "Good logic, but could use more modern ES6 features like arrow functions and Math.max"
            },
            {
                "assignment_id": "react_component_003",
                "code": """
import React, { useState } from 'react';

function TodoApp() {
    const [todos, setTodos] = useState([]);
    const [input, setInput] = useState('');

    const addTodo = () => {
        if(input.trim()) {
            setTodos([...todos, { id: Date.now(), text: input, completed: false }]);
            setInput('');
        }
    };

    return (
        <div>
            <h2>My Todo App</h2>
            <input 
                value={input} 
                onChange={(e) => setInput(e.target.value)}
                placeholder="Add a todo..."
            />
            <button onClick={addTodo}>Add</button>
            <ul>
                {todos.map(todo => (
                    <li key={todo.id}>{todo.text}</li>
                ))}
            </ul>
        </div>
    );
}

export default TodoApp;
                """,
                "submission_date": "2024-02-01",
                "concept_tags": ["React hooks", "state management", "event handling", "component structure"],
                "instructor_feedback": "Great use of hooks! Missing delete functionality and proper key handling"
            }
        ],
        "project_evaluations": [
            {
                "project_name": "Personal Portfolio Website",
                "score": 85,
                "max_score": 100,
                "rubric_scores": {
                    "html_structure": 90,
                    "css_styling": 80,
                    "responsiveness": 75,
                    "javascript_interactivity": 85,
                    "code_quality": 90
                },
                "feedback": "Well-structured project with clean code. Needs improvement in responsive design.",
                "completion_date": "2024-01-25"
            },
            {
                "project_name": "Task Management App",
                "score": 78,
                "max_score": 100,
                "rubric_scores": {
                    "react_components": 85,
                    "state_management": 75,
                    "user_interface": 80,
                    "functionality": 70,
                    "code_organization": 85
                },
                "feedback": "Good React implementation but missing CRUD operations and proper error handling.",
                "completion_date": "2024-02-05"
            }
        ],
        "quiz_scores": [
            {"topic": "HTML Fundamentals", "score": 92, "max_score": 100, "date": "2024-01-10"},
            {"topic": "CSS Basics", "score": 88, "max_score": 100, "date": "2024-01-12"},
            {"topic": "JavaScript Fundamentals", "score": 85, "max_score": 100, "date": "2024-01-18"},
            {"topic": "React Basics", "score": 82, "max_score": 100, "date": "2024-01-28"},
            {"topic": "Node.js Basics", "score": 75, "max_score": 100, "date": "2024-02-03"}
        ],
        "peer_reviews": [
            {
                "reviewer": "peer_001",
                "project_reviewed": "Personal Portfolio Website",
                "feedback": "Great design and clean code structure. Could add more interactive elements.",
                "rating": 4.2,
                "max_rating": 5.0,
                "date": "2024-01-26"
            },
            {
                "reviewer": "peer_002", 
                "project_reviewed": "Task Management App",
                "feedback": "Good use of React hooks but the UI could be more polished. Missing some functionality.",
                "rating": 3.8,
                "max_rating": 5.0,
                "date": "2024-02-06"
            }
        ],
        "mentor_evaluations": [
            {
                "mentor": "mentor_001",
                "evaluation_type": "mid_course_review",
                "strengths": ["Strong problem-solving skills", "Good coding practices", "Quick learner"],
                "improvement_areas": ["Modern JavaScript features", "Responsive design", "Error handling"],
                "overall_rating": 4.0,
                "max_rating": 5.0,
                "comments": "John shows good potential but needs to focus on modern development practices and edge cases.",
                "date": "2024-01-30"
            }
        ]
    }
}

# Define LangGraph workflow nodes

# Node 1: Code Analysis
def analyze_code_submissions(state: SkillAggregatorState) -> SkillAggregatorState:
    """Analyze code submissions to extract programming skills and patterns"""
    print("🔍 Analyzing Code Submissions...")
    
    code_analysis_prompt = PromptTemplate(
        input_variables=["code_data"],
        template="""
        Analyze the following code submissions and extract technical skills, coding patterns, and proficiency indicators.
        
        CODE SUBMISSIONS:
        {code_data}
        
        For each submission, identify:
        1. Technical concepts demonstrated
        2. Code quality indicators
        3. Modern best practices usage
        4. Areas for improvement
        5. Proficiency level estimate (0-100)
        
        Return analysis in JSON format:
        {{
            "skills_identified": {{"skill_name": "proficiency_score"}},
            "code_quality_metrics": {{"metric": "score"}},
            "patterns_observed": ["list of patterns"],
            "improvement_suggestions": ["list of suggestions"]
        }}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=code_analysis_prompt)
    code_data = json.dumps(state["learner_data"]["data_sources"]["code_submissions"], indent=2)
    result = chain.run(code_data=code_data)
    
    try:
        # Clean and parse result
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        code_analysis = json.loads(result)
    except:
        code_analysis = {"error": "Failed to parse code analysis", "raw_result": result}
    
    state["code_analysis"] = code_analysis
    state["processing_status"] = "code_analysis_complete"
    return state

# Node 2: Project Analysis
def analyze_project_evaluations(state: SkillAggregatorState) -> SkillAggregatorState:
    """Analyze project evaluations and rubric scores"""
    print("📊 Analyzing Project Evaluations...")
    
    project_analysis_prompt = PromptTemplate(
        input_variables=["project_data"],
        template="""
        Analyze the following project evaluations to understand practical application skills.
        
        PROJECT EVALUATIONS:
        {project_data}
        
        Extract:
        1. Practical skills demonstrated
        2. Rubric-based competency levels
        3. Project complexity handling
        4. Consistency across projects
        5. Overall project delivery capability
        
        Return JSON:
        {{
            "practical_skills": {{"skill_name": "proficiency_score"}},
            "rubric_analysis": {{"category": "average_score"}},
            "project_complexity": "beginner/intermediate/advanced",
            "consistency_score": 85
        }}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=project_analysis_prompt)
    project_data = json.dumps(state["learner_data"]["data_sources"]["project_evaluations"], indent=2)
    result = chain.run(project_data=project_data)
    
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        project_analysis = json.loads(result)
    except:
        project_analysis = {"error": "Failed to parse project analysis", "raw_result": result}
    
    state["project_analysis"] = project_analysis
    state["processing_status"] = "project_analysis_complete"
    return state

# Node 3: Quiz Analysis
def analyze_quiz_scores(state: SkillAggregatorState) -> SkillAggregatorState:
    """Analyze quiz performance to understand theoretical knowledge"""
    print("📝 Analyzing Quiz Scores...")
    
    quiz_analysis_prompt = PromptTemplate(
        input_variables=["quiz_data"],
        template="""
        Analyze quiz performance data to assess theoretical knowledge and learning progression.
        
        QUIZ SCORES:
        {quiz_data}
        
        Determine:
        1. Knowledge areas and strength levels
        2. Learning progression trends
        3. Theoretical vs practical gaps
        4. Consistency in performance
        
        Return JSON:
        {{
            "knowledge_areas": {{"topic": "score"}},
            "learning_trend": "improving/stable/declining",
            "theoretical_strength": 85,
            "consistency_rating": "high/medium/low"
        }}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=quiz_analysis_prompt)
    quiz_data = json.dumps(state["learner_data"]["data_sources"]["quiz_scores"], indent=2)
    result = chain.run(quiz_data=quiz_data)
    
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        quiz_analysis = json.loads(result)
    except:
        quiz_analysis = {"error": "Failed to parse quiz analysis", "raw_result": result}
    
    state["quiz_analysis"] = quiz_analysis
    state["processing_status"] = "quiz_analysis_complete"
    return state

# Node 4: Peer Review Analysis
def analyze_peer_reviews(state: SkillAggregatorState) -> SkillAggregatorState:
    """Analyze peer feedback for collaboration and soft skills"""
    print("👥 Analyzing Peer Reviews...")
    
    peer_analysis_prompt = PromptTemplate(
        input_variables=["peer_data"],
        template="""
        Analyze peer review feedback to understand collaboration skills and peer perception.
        
        PEER REVIEWS:
        {peer_data}
        
        Extract:
        1. Collaboration effectiveness
        2. Code review skills
        3. Communication patterns
        4. Peer-perceived strengths
        
        Return JSON:
        {{
            "collaboration_skills": {{"skill": "rating"}},
            "peer_feedback_themes": ["theme1", "theme2"],
            "average_peer_rating": 4.0,
            "collaboration_effectiveness": "high/medium/low"
        }}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=peer_analysis_prompt)
    peer_data = json.dumps(state["learner_data"]["data_sources"]["peer_reviews"], indent=2)
    result = chain.run(peer_data=peer_data)
    
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        peer_analysis = json.loads(result)
    except:
        peer_analysis = {"error": "Failed to parse peer analysis", "raw_result": result}
    
    state["peer_review_analysis"] = peer_analysis
    state["processing_status"] = "peer_analysis_complete"
    return state

# Node 5: Mentor Analysis
def analyze_mentor_evaluations(state: SkillAggregatorState) -> SkillAggregatorState:
    """Analyze mentor evaluations for expert assessment"""
    print("👨‍🏫 Analyzing Mentor Evaluations...")
    
    mentor_analysis_prompt = PromptTemplate(
        input_variables=["mentor_data"],
        template="""
        Analyze mentor evaluations to understand expert assessment of learner capabilities.
        
        MENTOR EVALUATIONS:
        {mentor_data}
        
        Extract:
        1. Expert-identified strengths
        2. Critical improvement areas
        3. Professional readiness indicators
        4. Mentorship insights
        
        Return JSON:
        {{
            "expert_strengths": ["strength1", "strength2"],
            "critical_gaps": ["gap1", "gap2"],
            "professional_readiness": 75,
            "mentor_confidence": "high/medium/low"
        }}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=mentor_analysis_prompt)
    mentor_data = json.dumps(state["learner_data"]["data_sources"]["mentor_evaluations"], indent=2)
    result = chain.run(mentor_data=mentor_data)
    
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        mentor_analysis = json.loads(result)
    except:
        mentor_analysis = {"error": "Failed to parse mentor analysis", "raw_result": result}
    
    state["mentor_analysis"] = mentor_analysis
    state["processing_status"] = "mentor_analysis_complete"
    return state

# Node 6: Skill Aggregation
def aggregate_skill_signals(state: SkillAggregatorState) -> SkillAggregatorState:
    """Combine all analysis results into unified skill signals"""
    print("🔄 Aggregating Skill Signals...")
    
    aggregation_prompt = PromptTemplate(
        input_variables=["all_analyses"],
        template="""
        Combine the following analysis results into a unified skill assessment:
        
        ANALYSIS RESULTS:
        {all_analyses}
        
        Create a comprehensive skill aggregation that:
        1. Weights different data sources appropriately
        2. Resolves conflicts between assessments
        3. Identifies consistent patterns
        4. Calculates consolidated proficiency scores
        
        Return JSON:
        {{
            "consolidated_skills": {{"skill_name": {{"score": 85, "confidence": "high", "sources": ["code", "projects"]}}}},
            "data_source_weights": {{"source": "weight_percentage"}},
            "conflict_resolutions": ["resolution1", "resolution2"],
            "confidence_metrics": {{"overall_confidence": 85}}
        }}
        """
    )
    
    all_analyses = {
        "code_analysis": state.get("code_analysis", {}),
        "project_analysis": state.get("project_analysis", {}),
        "quiz_analysis": state.get("quiz_analysis", {}),
        "peer_analysis": state.get("peer_review_analysis", {}),
        "mentor_analysis": state.get("mentor_analysis", {})
    }
    
    chain = LLMChain(llm=llm, prompt=aggregation_prompt)
    result = chain.run(all_analyses=json.dumps(all_analyses, indent=2))
    
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        aggregated_skills = json.loads(result)
    except:
        aggregated_skills = {"error": "Failed to parse aggregation", "raw_result": result}
    
    state["aggregated_skills"] = aggregated_skills
    state["processing_status"] = "aggregation_complete"
    return state

# Node 7: Final Skill Graph Generation
def generate_final_skill_graph(state: SkillAggregatorState) -> SkillAggregatorState:
    """Generate the final structured skill graph output"""
    print("📈 Generating Final Skill Graph...")
    
    final_graph_prompt = PromptTemplate(
        input_variables=["aggregated_data", "learner_info"],
        template="""
        Create the final skill graph for the learner based on aggregated analysis:
        
        AGGREGATED DATA:
        {aggregated_data}
        
        LEARNER INFO:
        {learner_info}
        
        Generate a comprehensive skill graph with:
        1. Technical skills with proficiency levels
        2. Soft skills assessment
        3. Learning patterns and trends
        4. Strengths and improvement areas
        5. Overall readiness metrics
        
        Use this exact JSON structure:
        {{
            "learner_id": "learner_001",
            "assessment_date": "2024-02-10",
            "skill_graph": {{
                "technical_skills": {{
                    "HTML/CSS": {{
                        "proficiency_level": "INTERMEDIATE",
                        "proficiency_score": 85,
                        "evidence_sources": ["code_submissions", "projects"],
                        "strengths": ["semantic structure", "basic styling"],
                        "improvement_areas": ["responsive design", "advanced CSS"],
                        "trend": "improving"
                    }}
                }},
                "soft_skills": {{
                    "collaboration": {{
                        "proficiency_level": "INTERMEDIATE",
                        "proficiency_score": 75,
                        "evidence_sources": ["peer_reviews"],
                        "observations": ["good communication", "needs leadership development"]
                    }}
                }}
            }},
            "overall_assessment": {{
                "primary_strengths": ["problem solving", "clean code", "quick learner"],
                "critical_gaps": ["modern frameworks", "testing", "deployment"],
                "learning_velocity": "fast",
                "engagement_level": "high"
            }}
        }}
        """
    )
    
    learner_info = {
        "learner_id": state["learner_data"]["learner_id"],
        "name": state["learner_data"]["name"],
        "course": state["learner_data"]["course"]
    }
    
    chain = LLMChain(llm=llm, prompt=final_graph_prompt)
    result = chain.run(
        aggregated_data=json.dumps(state.get("aggregated_skills", {}), indent=2),
        learner_info=json.dumps(learner_info, indent=2)
    )
    
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        final_skill_graph = json.loads(result)
    except:
        final_skill_graph = {"error": "Failed to parse final graph", "raw_result": result}
    
    state["final_skill_graph"] = final_skill_graph
    state["processing_status"] = "complete"
    return state

# Create the LangGraph workflow
workflow = StateGraph(SkillAggregatorState)

# Add nodes
workflow.add_node("analyze_code", analyze_code_submissions)
workflow.add_node("analyze_projects", analyze_project_evaluations)
workflow.add_node("analyze_quizzes", analyze_quiz_scores)
workflow.add_node("analyze_peer_reviews", analyze_peer_reviews)
workflow.add_node("analyze_mentor_feedback", analyze_mentor_evaluations)
workflow.add_node("aggregate_signals", aggregate_skill_signals)
workflow.add_node("generate_skill_graph", generate_final_skill_graph)

# Define the workflow edges
workflow.add_edge(START, "analyze_code")
workflow.add_edge("analyze_code", "analyze_projects")
workflow.add_edge("analyze_projects", "analyze_quizzes")
workflow.add_edge("analyze_quizzes", "analyze_peer_reviews")
workflow.add_edge("analyze_peer_reviews", "analyze_mentor_feedback")
workflow.add_edge("analyze_mentor_feedback", "aggregate_signals")
workflow.add_edge("aggregate_signals", "generate_skill_graph")
workflow.add_edge("generate_skill_graph", END)

# Compile the graph
skill_aggregator_graph = workflow.compile()

# Initialize the state
initial_state = SkillAggregatorState(
    learner_data=sample_learner_data,
    code_analysis={},
    project_analysis={},
    quiz_analysis={},
    peer_review_analysis={},
    mentor_analysis={},
    aggregated_skills={},
    final_skill_graph={},
    processing_status="initialized"
)

# Execute the workflow
print("🚀 Starting Skill Signal Aggregator Agent (LangGraph)")
print("=" * 70)

try:
    # Run the workflow
    final_state = skill_aggregator_graph.invoke(initial_state)
    
    print("\n✅ Workflow Complete!")
    print("=" * 70)
    
    # Display results
    print("\n📊 Final Skill Graph:")
    print("=" * 70)
    if "final_skill_graph" in final_state and "error" not in final_state["final_skill_graph"]:
        print(json.dumps(final_state["final_skill_graph"], indent=2))
        
        # Extract visualization data
        skill_graph = final_state["final_skill_graph"]
        technical_skills = skill_graph.get("skill_graph", {}).get("technical_skills", {})
        
        print("\n🎯 Technical Skills Summary:")
        print("=" * 70)
        for skill, data in technical_skills.items():
            level = data.get("proficiency_level", "N/A")
            score = data.get("proficiency_score", 0)
            trend = data.get("trend", "N/A")
            print(f"• {skill}: {level} ({score}%) - {trend}")
        
        # Overall metrics
        overall = skill_graph.get("overall_assessment", {})
        print(f"\n🏆 Learning Velocity: {overall.get('learning_velocity', 'N/A')}")
        print(f"📈 Engagement Level: {overall.get('engagement_level', 'N/A')}")
        print(f"💪 Top Strengths: {', '.join(overall.get('primary_strengths', []))}")
        print(f"🎯 Focus Areas: {', '.join(overall.get('critical_gaps', []))}")
        
    else:
        print("❌ Error in final skill graph generation")
        print(final_state.get("final_skill_graph", {}))
    
    print("\n📋 Processing Summary:")
    print("=" * 70)
    print(f"Status: {final_state.get('processing_status', 'Unknown')}")
    print("Nodes executed: Code → Projects → Quizzes → Peer Reviews → Mentor → Aggregation → Final Graph")
    
except Exception as e:
    print(f"❌ Error executing workflow: {e}")
    import traceback
    traceback.print_exc()

print("\n🔗 Ready for Next Agent!")
print("=" * 70)
print("Output format ready for Agent 2 (Industry Role Analyzer)")