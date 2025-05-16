# Product Research Assistant

A LangChain-based intelligent agent system that helps users compare technology products, analyze reviews, and make informed purchasing decisions through natural language interactions.

## Table of Contents
- [Overview](#overview)
- [Implementation Architecture](#implementation-architecture)
- [Core Components](#core-components)
- [Text Processing and Chunking Strategy](#text-processing-and-chunking-strategy)
- [Error Handling Approach](#error-handling-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Technical Challenges](#technical-challenges)
- [Potential Improvements](#potential-improvements)
- [Contributing](#contributing)

## Overview

The Product Research Assistant is an AI-powered agent that leverages Large Language Models (LLMs) to help users research and compare technology products. It combines natural language understanding with structured data processing to provide comprehensive product comparisons, price analyses, sentiment evaluations, and personalized recommendations.

Key capabilities include:
- Comparing specifications across multiple products
- Analyzing price differences and calculating payment plans
- Processing and summarizing product reviews
- Performing unit conversions and calculations
- Maintaining conversation context for follow-up questions

## Implementation Architecture

The Product Research Assistant is built on a modular architecture following the agent-based paradigm:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Interface │────▶│  Agent Executor │────▶│   LLM (GPT-4o)  │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  Memory System  │     │  ReAct Agent    │
                        │                 │     │                 │
                        └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │                 │
                                                │  Tool Suite     │
                                                │                 │
                                                └─────────────────┘
```

### Flow of Operation

1. User submits a natural language query
2. The agent executor processes the query through the LLM
3. The ReAct agent determines which tools to use
4. Tools execute operations and return results
5. Results are processed and formatted into a comprehensive response
6. Memory system stores the interaction for context in future queries

## Core Components

### 1. ReAct Agent Framework

The system implements LangChain's ReAct (Reasoning + Acting) agent framework, which combines reasoning and acting in an iterative process:

```python
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
    early_stopping_method="generate"
)
```

The ReAct pattern enables the agent to:
- Reason about the user's request
- Plan a sequence of actions
- Execute tools based on the plan
- Observe results and adjust subsequent actions
- Formulate a comprehensive response

### 2. LLM Integration

The system uses OpenAI's GPT-4o-mini model through the ChatOpenAI interface:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0.2
)
```

Key configuration choices:
- **Model Selection**: GPT-4o-mini provides a balance of performance and cost-effectiveness
- **Temperature**: Set to 0.2 to ensure consistent, factual responses with minimal hallucination
- **API Integration**: Managed through environment variables for security

### 3. Specialized Tools

The agent is equipped with four custom tools:

#### a. Product Search Tool
```python
@tool
def search_products(query: str) -> str:
    """Search for product information in our database."""
    logger.info(f"Searching for products: {query}")
    query = query.lower()
    results = {}
    
    for product, info in PRODUCT_DATABASE.items():
        if any(term in product for term in query.split()):
            results[product] = info
    
    return str(results)
```

#### b. Review Analysis Tool
```python
@tool
def analyze_reviews(review_text: str) -> Dict[str, Any]:
    """Analyze product review text and extract sentiment and key points."""
    # Implementation details in Text Processing section
```

#### c. Calculation Tool
```python
@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        result = safe_eval(expression)
        return f"{result:.2f}"
    except Exception as e:
        return f"Error calculating: {str(e)}"
```

#### d. Unit Conversion Tool
```python
@tool
def convert_units(from_unit: str, to_unit: str, value: float) -> str:
    """Convert a value from one unit to another."""
    conversions = {
        ("inches", "cm"): lambda x: x * 2.54,
        ("cm", "inches"): lambda x: x / 2.54,
        # Additional conversions...
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{result:.2f} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} is not supported."
```

### 4. Memory Management

The system implements ConversationBufferMemory to maintain context across interactions:

```python
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history,
    return_messages=True
)
```

Memory considerations:
- Stores previous interactions as message objects
- Provides context for follow-up questions
- Enables the agent to reference previous findings

### 5. Prompt Engineering

The system uses a structured prompt template that guides the agent to format responses consistently:

```python
system_prompt = """
You are a Product Research Assistant that helps users compare technology products.
Available tools: {tool_names}
Tool descriptions: {tools}

When comparing products, always structure your response in this format:
1. Feature Comparison (in a table when possible)
2. Price Analysis
3. User Sentiment
4. Recommendation

Use the tools to gather information and analyze products.

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""
```

This prompt template:
- Defines the agent's role and capabilities
- Specifies the expected response format
- Provides clear instructions for the ReAct process
- Ensures consistent output structure

## Text Processing and Chunking Strategy

### The 200-Character Limitation Challenge

A significant technical challenge in the implementation was handling long review texts. The `analyze_reviews` tool initially faced a 200-character limitation, which caused issues when processing comprehensive product reviews that often exceed this limit.

This limitation manifested in several ways:
- Truncated reviews lost critical context and sentiment
- Key points from later sections were completely missed
- Overall sentiment analysis was skewed toward the beginning of reviews
- Product-specific features mentioned later were ignored

### Chunking Solution Implementation

To address this limitation, we implemented a sophisticated chunking strategy:

#### 1. Text Chunking Function
```python
def chunk_text(text: str, chunk_size: int = 150, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks to preserve context."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end == text_length and end - start < chunk_size // 2 and chunks:
            chunks[-1] = text[start - chunk_size + overlap:end]
            break
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

#### 2. Individual Chunk Processing
Each chunk is processed separately to extract:
- Positive sentiment indicators
- Negative sentiment indicators
- Key points and insights
- Product-specific features

#### 3. Result Combination Algorithm
```python
def combine_chunk_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine results from multiple review chunks."""
    combined_sentiment = "neutral"
    positive_counts = sum(r["positive_count"] for r in results)
    negative_counts = sum(r["negative_count"] for r in results)
    
    if positive_counts > negative_counts:
        combined_sentiment = "positive"
    elif negative_counts > positive_counts:
        combined_sentiment = "negative"
    
    # Prioritize key points about sound quality
    sound_quality_keywords = ["sound", "audio", "bass", "treble", "mids", "highs", 
                             "clarity", "balance", "noise cancellation", "detail"]
    
    key_points = []
    seen = set()
    
    # First add points that mention sound quality
    for result in results:
        for point in result["key_points"]:
            if point not in seen and any(keyword in point.lower() for keyword in sound_quality_keywords):
                seen.add(point)
                key_points.append(point)
    
    # Then add other points until we reach 5 total points
    if len(key_points) < 5:
        for result in results:
            for point in result["key_points"]:
                if point not in seen and len(key_points) < 5:
                    seen.add(point)
                    key_points.append(point)
    
    return {
        "sentiment": combined_sentiment,
        "key_points": key_points,
        "positive_count": positive_counts,
        "negative_count": negative_counts
    }
```

### Technical Considerations for Chunking

The chunking strategy was carefully designed with these considerations:
- **Chunk Size**: 150 characters balances processing efficiency with context preservation
- **Overlap Size**: 50 characters ensures sentences split across chunks are properly captured
- **Deduplication**: The seen set prevents duplicate points from being included
- **Prioritization**: Domain-specific keywords help prioritize the most relevant points
- **Sentiment Aggregation**: Positive and negative counts are summed across all chunks for balanced sentiment analysis

This approach allows the system to process reviews of arbitrary length while maintaining context and extracting meaningful insights.

## Error Handling Approach

The implementation includes a comprehensive error handling strategy:

### 1. Safe Evaluation for Calculations

To prevent code injection and ensure security, mathematical expressions are evaluated in a sandboxed environment:

```python
def safe_eval(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"Invalid characters in expression: {expression}")
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
```

### 2. Graceful Degradation

When tools encounter issues, they return reasonable defaults rather than failing completely:

```python
if not review_text.strip():
    logger.warning("Empty review text provided")
    return {"sentiment": "neutral", "key_points": [], "positive_count": 0, "negative_count": 0}
```

### 3. Exception Logging

Comprehensive logging captures errors at various levels:

```python
try:
    chunk_result = process_review_chunk(chunk)
    results.append(chunk_result)
except Exception as e:
    logger.error(f"Error processing chunk {i+1}: {str(e)}")
    continue
```

### 4. User Feedback

Clear error messages are provided to maintain transparency:

```python
except Exception as e:
    logger.error(f"Error processing input: {str(e)}")
    print(f"Error: {str(e)}")
```

### 5. Iteration Management

The agent executor includes parameters to prevent infinite loops:

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
    early_stopping_method="generate"
)
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd product-research-assistant
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv langchain_env
   source langchain_env/bin/activate  # On Windows: langchain_env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Running the Assistant

```
python ChatModels/chat.py
```

### Example Queries

1. **Product Comparison**:
   ```
   Compare the battery life of the latest iPhone and Samsung Galaxy phones
   ```

2. **Price Analysis and Payment Calculation**:
   ```
   What is the price difference between MacBook Air and Dell XPS 13? Calculate the monthly payment if I buy the cheaper one on a 12-month installment plan
   ```

3. **Review Analysis**:
   ```
   Summarize user opinions about the sound quality of Sony WH-1000XM5 headphones
   ```

4. **Unit Conversion**:
   ```
   Convert 65 inches to centimeters
   ```

### Response Format

The assistant provides structured responses with:
1. Feature Comparison (often in table format)
2. Price Analysis
3. User Sentiment
4. Recommendation

## Testing

### Running Tests

```
python test_agent.py
```

The test script verifies:
- Product comparison functionality
- Price calculation accuracy
- Review analysis capabilities
- Edge case handling

### Test Coverage

The testing framework covers:
- Basic functionality tests
- Edge cases (empty reviews, invalid calculations)
- Response format verification
- Tool execution accuracy

## Technical Challenges

Beyond the 200-character limitation discussed earlier, several other technical challenges were addressed:

### 1. Agent Reasoning Loop

Initial implementations encountered issues with the agent getting stuck in reasoning loops. This was addressed by:
- Implementing a maximum iteration limit
- Adding early stopping mechanisms
- Refining the prompt to encourage decisive actions

### 2. Memory Management

Token limits posed challenges for long conversations. Solutions included:
- Implementing conversation summarization
- Pruning older messages when necessary
- Focusing on relevant context retention

### 3. Error Recovery

When tools failed, the agent would sometimes get confused. Improvements included:
- Better error messages that the agent could understand
- Fallback strategies for failed tool calls
- Explicit guidance in the system prompt for handling errors

## Potential Improvements

### 1. Dynamic Product Database
Replace the static product database with a real-time API that:
- Fetches current product information
- Accesses real user reviews
- Updates pricing information automatically
- Expands to new product categories

### 2. Advanced Sentiment Analysis
Implement more sophisticated sentiment analysis:
- Use embeddings or fine-tuned models instead of keyword counting
- Detect nuanced emotions (excitement, disappointment, satisfaction)
- Identify comparative sentiments between products
- Weight reviews by recency and relevance

### 3. Multi-modal Capabilities
Add support for image processing:
- Analyze product photos for visual features
- Process screenshots of specifications
- Compare product designs visually
- Generate comparison charts and graphs

### 4. Personalization
Incorporate user preferences:
- Build user profiles based on past interactions
- Adjust recommendations based on stated preferences
- Remember previously discussed products
- Adapt to user's technical knowledge level

### 5. Performance Optimization
Implement caching and optimization:
- Cache frequently requested product information
- Pre-process common review analyses
- Implement parallel processing for multiple tool calls
- Optimize chunking strategies based on content type

### 6. Expanded Tool Set
Add more specialized tools:
- Price tracking over time
- Feature comparison across product categories
- Competitor analysis
- Technical compatibility checking
- Environmental impact assessment

### 7. Enhanced Memory Management
Implement more sophisticated memory strategies:
- Hierarchical memory with short and long-term components
- Importance-based retention policies
- Summarization of previous interactions
- Contextual retrieval based on current query

### 8. Comprehensive Testing Framework
Develop a robust testing framework:
- Automated regression testing
- Performance benchmarking
- User satisfaction metrics
- A/B testing of different prompt strategies

## Contributing

Contributions to the Product Research Assistant are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

Please ensure your code follows the project's style guidelines and passes all tests.


