import streamlit as st
import google.generativeai as genai
from tavily import TavilyClient
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import concurrent.futures
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    content: str
    url: str
    relevance_score: float = 0.0

@dataclass
class ResearchData:
    topic: str
    questions: List[str]
    search_results: Dict[str, List[SearchResult]]
    report: str
    metadata: Dict
    
class ResearchAgent:
    def __init__(self, gemini_api_key: str, tavily_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.tavily_api_key = tavily_api_key
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients with error handling"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
        
        try:
            self.tavily = TavilyClient(api_key=self.tavily_api_key)
            logger.info("Tavily client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily: {e}")
            raise
    
    def generate_research_questions(self, topic: str, num_questions: int = 6) -> List[str]:
        """Generate research questions with improved prompting"""
        prompt = f"""
        You are an expert research assistant. Generate {num_questions} comprehensive research questions about: "{topic}"
        
        Guidelines:
        - Cover different aspects: definition, history, current state, impacts, solutions, future trends
        - Make questions specific enough to yield actionable insights
        - Ensure questions are web-searchable
        - Avoid overly broad or vague questions
        
        Format: Return ONLY a numbered list of questions, one per line.
        
        Example format:
        1. What are the main causes of [topic]?
        2. How has [topic] evolved over the past decade?
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            questions = []
            
            for line in response.text.split('\n'):
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    # Extract question after number and period
                    question = re.sub(r'^\d+\.\s*', '', line)
                    if question:
                        questions.append(question)
            
            return questions[:num_questions]  # Ensure we don't exceed requested number
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return [f"What is {topic}?", f"What are the main aspects of {topic}?"]
    
    def search_web_parallel(self, questions: List[str], max_results: int = 3) -> Dict[str, List[SearchResult]]:
        """Perform parallel web searches for efficiency"""
        def search_single_question(question: str) -> tuple:
            try:
                search_result = self.tavily.search(
                    query=question, 
                    max_results=max_results,
                    search_depth="advanced"
                )
                
                results = []
                for result in search_result.get('results', []):
                    results.append(SearchResult(
                        title=result.get('title', 'No title'),
                        content=result.get('content', 'No content'),
                        url=result.get('url', ''),
                        relevance_score=result.get('score', 0.0)
                    ))
                
                return question, results
            except Exception as e:
                logger.error(f"Error searching for '{question}': {e}")
                return question, []
        
        # Use ThreadPoolExecutor for parallel searches
        search_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_question = {
                executor.submit(search_single_question, q): q for q in questions
            }
            
            for future in concurrent.futures.as_completed(future_to_question):
                question, results = future.result()
                search_results[question] = results
        
        return search_results
    
    def generate_research_report(self, research_data: ResearchData) -> str:
        """Generate comprehensive research report with improved structure"""
        
        # Count total sources
        total_sources = sum(len(results) for results in research_data.search_results.values())
        
        # Format research context
        research_context = f"# Research Topic: {research_data.topic}\n\n"
        research_context += f"Total Sources Analyzed: {total_sources}\n\n"
        
        for i, question in enumerate(research_data.questions):
            research_context += f"## Research Question {i+1}: {question}\n"
            results = research_data.search_results.get(question, [])
            
            for j, result in enumerate(results):
                research_context += f"### Source {j+1}: {result.title}\n"
                research_context += f"URL: {result.url}\n"
                research_context += f"Content: {result.content[:800]}...\n"  # Limit content length
                research_context += f"Relevance Score: {result.relevance_score:.2f}\n\n"
        
        prompt = f"""
        You are a professional research analyst. Create a comprehensive research report using the following data:
        
        {research_context}
        
        Report Requirements:
        1. **Executive Summary**: 2-3 sentence overview of key findings
        2. **Introduction**: Context and scope of research
        3. **Methodology**: Brief description of research approach
        4. **Key Findings**: Organize by research questions with:
           - Clear headings for each question
           - Synthesis of information from multiple sources
           - Specific data points and evidence
           - Conflicting viewpoints if any
        5. **Analysis**: Critical evaluation of findings
        6. **Conclusions**: Summary of insights and implications
        7. **Recommendations**: Actionable next steps (if applicable)
        8. **Sources**: Referenced throughout with proper attribution
        
        Use professional markdown formatting with:
        - Clear hierarchy of headings (##, ###, ####)
        - Bullet points for key information
        - **Bold** for emphasis
        - > Blockquotes for important insights
        
        Keep the tone professional but accessible.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"

# Streamlit Configuration
st.set_page_config(
    page_title="ReAct Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .research-question {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'research_agent' not in st.session_state:
        st.session_state.research_agent = None
    if 'research_data' not in st.session_state:
        st.session_state.research_data = None
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []

init_session_state()

# Sidebar for API Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Keys
    gemini_key = st.text_input(
        "Gemini API Key",
        value="AIzaSyBUgD1d0p8SNP73gbGTMb5mGCZajuHQ4Mo",
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    tavily_key = st.text_input(
        "Tavily API Key",
        value="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",
        type="password",
        help="Get your API key from Tavily"
    )
    
    # Research Parameters
    st.subheader("Research Parameters")
    num_questions = st.slider("Number of Research Questions", 3, 10, 6)
    max_sources = st.slider("Sources per Question", 2, 5, 3)
    
    # Initialize Agent
    if st.button("Initialize Agent"):
        try:
            st.session_state.research_agent = ResearchAgent(gemini_key, tavily_key)
            st.success("✅ Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize: {str(e)}")
    
    # Research History
    if st.session_state.research_history:
        st.subheader("📚 Research History")
        for i, research in enumerate(st.session_state.research_history):
            if st.button(f"📄 {research['topic'][:30]}...", key=f"history_{i}"):
                st.session_state.research_data = research

# Main Content
st.markdown('<div class="main-header"><h1>🧠 ReAct Research Agent</h1><p>Intelligent Web Research with AI-Powered Analysis</p></div>', unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "🔍 Enter your research topic:",
        value="Climate change impacts on biodiversity",
        help="Be specific for better results"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    research_button = st.button("🚀 Start Research", type="primary")

# Advanced Options
with st.expander("⚙️ Advanced Options"):
    col1, col2, col3 = st.columns(3)
    with col1:
        include_analysis = st.checkbox("Include Critical Analysis", value=True)
    with col2:
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    with col3:
        export_format = st.selectbox("Export Format", ["Markdown", "JSON", "PDF"])

# Research Process
if research_button and topic:
    if not st.session_state.research_agent:
        st.error("❌ Please initialize the agent first using the sidebar.")
    else:
        agent = st.session_state.research_agent
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_container = st.container()
        
        try:
            with status_container:
                st.info("🔍 **Step 1/3**: Generating research questions...")
                progress_bar.progress(10)
                
                questions = agent.generate_research_questions(topic, num_questions)
                
                if not questions:
                    st.error("Failed to generate research questions. Please try again.")
                    st.stop()
                
                st.success(f"✅ Generated {len(questions)} research questions")
                progress_bar.progress(30)
                
                # Display questions
                st.subheader("📋 Research Questions")
                for i, question in enumerate(questions):
                    st.markdown(f'<div class="research-question"><strong>Q{i+1}:</strong> {question}</div>', unsafe_allow_html=True)
                
                st.info("🌐 **Step 2/3**: Searching the web...")
                progress_bar.progress(50)
                
                search_results = agent.search_web_parallel(questions, max_sources)
                
                total_sources = sum(len(results) for results in search_results.values())
                st.success(f"✅ Found {total_sources} sources across all questions")
                progress_bar.progress(70)
                
                st.info("📝 **Step 3/3**: Generating research report...")
                progress_bar.progress(85)
                
                # Create research data object
                research_data = ResearchData(
                    topic=topic,
                    questions=questions,
                    search_results=search_results,
                    report="",
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "total_sources": total_sources,
                        "num_questions": len(questions)
                    }
                )
                
                report = agent.generate_research_report(research_data)
                research_data.report = report
                
                st.session_state.research_data = research_data
                st.session_state.research_history.append({
                    "topic": topic,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "questions": questions,
                    "search_results": search_results,
                    "report": report
                })
                
                progress_bar.progress(100)
                st.success("✅ **Research Complete!**")
                
        except Exception as e:
            st.error(f"❌ Research failed: {str(e)}")
            logger.error(f"Research failed for topic '{topic}': {e}")

# Display Results
if st.session_state.research_data:
    research_data = st.session_state.research_data
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Report", "🔍 Questions", "📚 Sources", "📈 Analytics"])
    
    with tab1:
        st.subheader("Research Report")
        
        # Export options
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("📥 Download Report"):
                st.download_button(
                    label="Download as Markdown",
                    data=research_data.report,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            if st.button("📋 Copy to Clipboard"):
                st.code(research_data.report, language="markdown")
        
        # Display report
        st.markdown(research_data.report, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Research Questions")
        for i, question in enumerate(research_data.questions):
            with st.expander(f"Question {i+1}: {question}"):
                results = research_data.search_results.get(question, [])
                st.write(f"**Sources found:** {len(results)}")
                for j, result in enumerate(results):
                    st.write(f"**{j+1}.** [{result.title}]({result.url})")
                    st.write(f"*Relevance: {result.relevance_score:.2f}*")
    
    with tab3:
        st.subheader("Source Analysis")
        all_sources = []
        for results in research_data.search_results.values():
            all_sources.extend(results)
        
        if all_sources:
            # Sort by relevance
            all_sources.sort(key=lambda x: x.relevance_score, reverse=True)
            
            st.write(f"**Total Sources:** {len(all_sources)}")
            st.write(f"**Average Relevance Score:** {sum(s.relevance_score for s in all_sources) / len(all_sources):.2f}")
            
            for i, source in enumerate(all_sources[:10]):  # Show top 10
                with st.expander(f"#{i+1} {source.title} (Score: {source.relevance_score:.2f})"):
                    st.write(f"**URL:** {source.url}")
                    st.write(f"**Content Preview:** {source.content[:300]}...")
    
    with tab4:
        st.subheader("Research Analytics")
        
        if research_data.metadata:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Questions", research_data.metadata.get('num_questions', 0))
            
            with col2:
                st.metric("Total Sources", research_data.metadata.get('total_sources', 0))
            
            with col3:
                st.metric("Avg Sources/Question", 
                         round(research_data.metadata.get('total_sources', 0) / 
                               max(research_data.metadata.get('num_questions', 1), 1), 1))
        
        # Word cloud or other analytics could be added here
        st.info("📈 Advanced analytics features coming soon!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Gemini AI, and Tavily | Enhanced ReAct Research Agent")