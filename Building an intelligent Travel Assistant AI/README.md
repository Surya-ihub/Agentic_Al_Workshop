# 🌍 Travel Assistant AI

An intelligent travel planning assistant powered by Google Gemini, LangChain, and Streamlit that provides comprehensive travel information including weather, attractions, accommodations, and personalized recommendations.

## ✨ Features

- **Real-time Weather Information**: Get current weather conditions for any destination
- **Top Attractions Discovery**: Find the best tourist spots with detailed descriptions
- **Accommodation Recommendations**: Get personalized hotel and stay suggestions based on travel style
- **Interactive Chat Interface**: Natural conversation flow with chat history
- **Customizable Travel Preferences**: Set travel style, duration, and budget
- **Multi-API Integration**: Robust fallback systems for reliable data retrieval

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Google Gemini (gemini-2.5-flash)
- **Agent Framework**: LangChain
- **Search**: DuckDuckGo Search API
- **Weather APIs**: WeatherAPI.com (primary), OpenWeatherMap (fallback)
- **Language**: Python 3.8+

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Google Gemini API
  - WeatherAPI.com
  - OpenWeatherMap (optional, for fallback)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd travel-assistant-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API keys**
   - Open the main Python file and replace the API keys:
   ```python
   WEATHER_API_KEY = "your_weather_api_key"
   GEMINI_API_KEY = "your_gemini_api_key"
   ```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🔧 Configuration

### API Keys Setup

1. **Google Gemini API**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Replace `GEMINI_API_KEY` in the code

2. **WeatherAPI.com**:
   - Sign up at [WeatherAPI.com](https://www.weatherapi.com/)
   - Get your free API key
   - Replace `WEATHER_API_KEY` in the code

3. **OpenWeatherMap** (Optional):
   - Sign up at [OpenWeatherMap](https://openweathermap.org/api)
   - Get your API key
   - Replace the key in the fallback weather function

### Environment Variables (Recommended)

For better security, use environment variables:

```bash
export WEATHER_API_KEY="your_weather_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

Then update the code:
```python
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

## 🎯 Usage

1. **Set Travel Preferences**: Use the sidebar to configure:
   - Destination location
   - Travel style (Adventure, Relaxation, Cultural, etc.)
   - Trip duration
   - Budget level

2. **Ask Questions**: Type natural language queries like:
   - "Plan a 5-day trip to Tokyo"
   - "What's the weather like in Paris?"
   - "Find luxury hotels in Bali"
   - "What are the top attractions in New York?"

3. **Get Comprehensive Information**: The assistant provides:
   - Current weather conditions
   - Top-rated attractions with descriptions
   - Accommodation recommendations
   - Personalized travel tips

## 📁 Project Structure

```
travel-assistant-ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file (recommended)
```

## 🔍 Features Deep Dive

### Weather Integration
- Primary: WeatherAPI.com for detailed weather data
- Fallback: OpenWeatherMap for reliability
- Provides temperature, conditions, humidity, and wind information

### Smart Search
- Uses DuckDuckGo search for finding attractions and accommodations
- AI-powered result summarization for better readability
- Contextual search based on travel preferences

### Intelligent Agent
- LangChain-based agent with tool-calling capabilities
- Structured response format with clear sections
- Error handling and graceful degradation

### User Experience
- Clean, intuitive chat interface
- Persistent chat history
- Customizable travel preferences
- Debug information for transparency

## 🛡️ Error Handling

The application includes robust error handling:
- API failures gracefully handled with fallback options
- Clear error messages for users
- Detailed debug information available
- Timeout protection for API calls

## 🔮 Future Enhancements

- **Flight Information**: Integration with flight APIs
- **Currency Conversion**: Real-time exchange rates
- **Language Translation**: Multi-language support
- **Image Generation**: Visual travel content
- **Itinerary Planning**: Day-by-day trip schedules
- **User Profiles**: Save preferences and trip history
- **Offline Mode**: Cached data for limited connectivity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues:

1. Check the [Issues](https://github.com/your-username/travel-assistant-ai/issues) section
2. Ensure all API keys are correctly configured
3. Verify internet connectivity for API calls
4. Check Python version compatibility (3.8+)

## 🙏 Acknowledgments

- Google Gemini for powerful AI capabilities
- LangChain for agent framework
- Streamlit for the beautiful UI framework
- WeatherAPI.com for weather data
- DuckDuckGo for search capabilities

---

**Built with ❤️ for travelers worldwide** 🌎✈️