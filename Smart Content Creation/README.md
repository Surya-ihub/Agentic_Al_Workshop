# 🤖 Agentic AI Content Refinement

A Streamlit application that simulates a conversation between two AI agents - a Content Creator and a Content Critic - to iteratively improve content about Generative AI topics.

## Features

- **Content Creator Agent**: Drafts and revises technical content in markdown format
- **Content Critic Agent**: Provides constructive feedback on content quality and accuracy
- **Interactive UI**: Easy-to-use Streamlit interface with customizable parameters
- **Conversation History**: View the complete dialogue between agents

## Setup

### Prerequisites

- Python 3.8 or higher
- Google AI API key (Gemini)

### Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update the API key in the code:
   - Open the main Python file
   - Replace `"AIzaSyDG-0xIaprzdT70VTf-LnMt62_s-F8SJqA"` with your actual Google AI API key

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Configure the simulation:
   - Enter a discussion topic (default: "Agentic AI")
   - Set the number of conversation turns (3-5)
   - Click "Start Simulation"

4. Watch as the agents collaborate to create and refine content

## How It Works

1. **Turn 1**: Content Creator generates initial content about the topic
2. **Turn 2**: Content Critic evaluates and provides feedback
3. **Turn 3**: Content Creator revises based on feedback
4. **Additional turns**: Continue the refinement process

## API Key Setup

To get your Google AI API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Replace the placeholder key in the code

## File Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Customization

- Modify system messages to change agent behavior
- Adjust conversation turns for longer/shorter sessions
- Update prompts for different content types
- Change the UI layout in Streamlit sections

## Troubleshooting

- **API Errors**: Ensure your Google AI API key is valid and has quota
- **Rate Limits**: The app includes built-in delays to avoid rate limiting
- **Model Issues**: Check that the Gemini model is accessible in your region

## License

This project is open source and available under the MIT License.