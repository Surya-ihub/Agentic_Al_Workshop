# Clothing Store Competitor Intelligence 👔

An AI-powered multi-agent system for analyzing clothing store competitors using Streamlit and Google's Gemini AI. This application employs multiple specialized AI agents to provide comprehensive market analysis for retail businesses.

## Features

- **Multi-Agent Analysis**: Uses specialized AI agents for research, strategy, and reporting
- **Location-Based Intelligence**: Analyzes competitors in specific geographic areas
- **Customizable Analysis**: Adjustable detail levels and competitor count
- **Professional Reports**: Generates business-ready markdown reports
- **Interactive UI**: User-friendly Streamlit interface
- **Export Functionality**: Download reports as markdown files

## Architecture

The system uses three specialized agents:

1. **Research Analyst**: Identifies competitors, analyzes market positioning, and studies foot traffic patterns
2. **Strategy Consultant**: Develops competitive strategies and identifies market opportunities
3. **Report Compiler**: Creates professional, formatted reports with actionable insights

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Internet connection for API access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd clothing-store-competitor-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Get your Gemini API key:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in the application

## Usage

1. Run the Streamlit application:
```bash
streamlit run competitor_analysis_agentic.py
```

2. Configure the analysis in the sidebar:
   - Enter your Gemini API key
   - Specify the location (e.g., "Koramangala, Bangalore")
   - Set the number of competitors to analyze (3-10)
   - Choose detail level (Summary, Detailed, Comprehensive)

3. Click "Generate Report" to start the analysis

4. View the generated report and download it as needed

## Configuration Options

### Detail Levels
- **Summary**: Basic competitor overview with key insights
- **Detailed**: Comprehensive analysis with market positioning
- **Comprehensive**: In-depth analysis with strategic recommendations

### Location Examples
- "Koramangala, Bangalore"
- "Connaught Place, Delhi"
- "Bandra, Mumbai"
- "Park Street, Kolkata"

## Sample Output

The system generates reports with the following sections:

1. **Competitor Overview**: Table format with store names, positioning, and key details
2. **Market Analysis**: Foot traffic patterns, peak hours, and market gaps
3. **Strategic Recommendations**: Actionable insights for operations and marketing
4. **Executive Summary**: Key findings and next steps

## Technical Details

### Agent Communication
- Uses AutoGen framework for multi-agent orchestration
- Round-robin speaker selection for systematic analysis
- Maximum 6 rounds of agent interaction

### API Integration
- Google Gemini 1.5 Flash model for AI responses
- LangChain integration for message handling
- Error handling for API failures

### Data Processing
- Converts AutoGen messages to LangChain format
- Handles deepcopy issues with custom reply functions
- Manages conversation state across agents

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Gemini API key is valid and has sufficient quota
2. **Location Not Found**: Use specific, well-known location names
3. **Slow Response**: Large competitor counts may take longer to process
4. **Report Not Generated**: Check the conversation log for agent errors

### Error Messages
- "Please enter your Gemini API key": Add a valid API key in the sidebar
- "Analysis error": Check internet connection and API key validity
- "Final report not found": The system will show the full conversation instead

## Limitations

- Requires active internet connection for API calls
- Analysis quality depends on Gemini's knowledge of the specified location
- Real-time data may not be available for all locations
- API rate limits may affect large analyses

## Security Notes

- API keys are handled securely (password field in UI)
- No data is stored locally
- All analysis is performed via Google's secure APIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error messages in the Streamlit interface
3. Ensure all dependencies are correctly installed
4. Verify your Gemini API key is active

## Roadmap

- [ ] Add support for additional AI models (OpenAI, Claude)
- [ ] Implement real-time data integration
- [ ] Add visualization charts and graphs
- [ ] Include social media sentiment analysis
- [ ] Add competitor pricing analysis
- [ ] Implement scheduled report generation

## Changelog

### v1.0.0
- Initial release with multi-agent competitor analysis
- Streamlit UI with configurable parameters
- Gemini 1.5 Flash integration
- Markdown report generation and download