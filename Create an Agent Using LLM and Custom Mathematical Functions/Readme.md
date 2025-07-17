# Math-Q&A Agent with LangGraph

## Overview

A conversational AI agent that combines general knowledge responses with mathematical calculation capabilities. The agent uses Google's Gemini model through LangChain and LangGraph for state management and tool calling.

## Features

- **Natural Language Processing**: Understands and responds to general knowledge questions
- **Mathematical Calculations**: Performs addition, subtraction, multiplication, and division
- **Conversation Memory**: Maintains context across multiple interactions
- **Error Handling**: Gracefully manages calculation errors (like division by zero)

## Prerequisites

- Python 3.8+
- Google API key (for Gemini access)
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/math-qa-agent.git
   cd math-qa-agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Run the interactive chat interface:

```bash
python math_agent.py
```

Example interactions:

```
You: What's the capital of France?
Agent: The capital of France is Paris.

You: What is 15 plus 27?
Agent: 15 plus 27 equals 42.

You: Divide 10 by 0
Agent: Error: Cannot divide by zero
```

Type 'exit' or 'quit' to end the session.

## Architecture

```
User Input → LangGraph Workflow → Agent → Tools (if needed) → Response
```

Key components:

- **State Management**: Handled by LangGraph's StateGraph
- **Agent**: Uses create_tool_calling_agent from LangChain
- **Tools**: Math operations (+, -, \*, /)
- **LLM**: Google's Gemini 1.5 Flash model

## Configuration

Modify these parameters in `math_agent.py` as needed:

- `model="gemini-1.5-flash"` - Change the Gemini model version
- `temperature=0` - Adjust creativity (0-1)
- System prompt in `system_prompt` variable

## Adding New Tools

To add additional capabilities:

1. Create a new tool function with `@tool` decorator
2. Add it to the `tools` list
3. Update the system prompt if needed

Example:

```python
@tool
def power(a: float, b: float) -> float:
    """Raise a to the power of b."""
    return a ** b

tools = [..., power]  # Add to existing tools
```

## Limitations

- Currently supports only basic math operations
- Dependent on Google's Gemini API availability
- No persistent chat history between sessions

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
