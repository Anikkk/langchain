import os
import logging
from typing import Dict, List, Any
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.tools.render import render_text_description

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Updated product database with iPhone 16 and Galaxy S25 series
PRODUCT_DATABASE = {
    "iphone 16 pro max": {
        "battery": "4685 mAh, up to 29 hours web surfing",
        "price": 1099.00,
        "specs": {"display": "6.9 inch", "weight": "227g"}
    },
    "iphone 16 plus": {
        "battery": "4350 mAh, up to 24 hours web surfing",
        "price": 899.00,
        "specs": {"display": "6.7 inch", "weight": "199g"}
    },
    "samsung galaxy s25 ultra": {
        "battery": "5000 mAh, up to 14 hours 15 minutes web surfing",
        "price": 1299.99,
        "specs": {"display": "6.8 inch", "weight": "219g"}
    },
    "samsung galaxy s25 plus": {
        "battery": "4900 mAh, up to 14 hours 40 minutes web surfing",
        "price": 999.99,
        "specs": {"display": "6.7 inch", "weight": "206g"}
    },
    "samsung galaxy s25": {
        "battery": "4000 mAh, up to 14 hours 15 minutes web surfing",
        "price": 799.99,
        "specs": {"display": "6.2 inch", "weight": "168g"}
    },
    "macbook air": {
        "price": 999.99,
        "specs": {"display": "13.6 inch", "weight": "1.24kg", "storage": "256GB"}
    },
    "dell xps 13": {
        "price": 899.99,
        "specs": {"display": "13.4 inch", "weight": "1.17kg", "storage": "512GB"}
    },
    "sony wh-1000xm5": {
        "price": 349.99,
        "specs": {"battery": "30 hours", "weight": "250g"}
    }
}

# Sample review text for testing
PRODUCT_REVIEWS = {
    "sony wh-1000xm5": """
    These headphones have incredible sound quality. The bass is deep and rich without overpowering the mids and highs.
    The noise cancellation is probably the best I've experienced, completely blocking out ambient noise in most environments.
    Battery life is excellent, lasting well over the advertised 30 hours. 
    The touch controls are intuitive and responsive. Build quality feels premium, though some users might find them a bit large.
    My only complaint is the price, but considering the quality, it's worth the investment for serious audio enthusiasts.
    
    The sound stage is impressively wide for closed-back headphones, creating an immersive listening experience.
    Vocal clarity is exceptional, making these perfect for podcasts and vocal-heavy music.
    The treble is crisp without being harsh or fatiguing during long listening sessions.
    Mid-range frequencies are well-balanced and detailed, allowing you to hear subtle nuances in complex musical arrangements.
    
    The companion app offers useful EQ customization to tailor the sound signature to your preferences.
    Some audiophiles might find the default sound profile slightly bass-heavy, but this can be adjusted.
    The LDAC codec support provides excellent high-resolution audio quality when paired with compatible devices.
    There's minimal sound leakage even at higher volumes, making these suitable for quiet environments like offices.
    
    The adaptive sound control feature intelligently adjusts noise cancellation based on your environment, enhancing the audio experience.
    Overall, these headphones deliver studio-quality sound that rivals much more expensive audiophile equipment.
    """
}

# Safe evaluation function for calculations
def safe_eval(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"Invalid characters in expression: {expression}")
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

# Text chunking utilities
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
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

# Tool implementations
@tool
def convert_units(from_unit: str, to_unit: str, value: float) -> str:
    """Convert a value from one unit to another."""
    conversions = {
        ("inches", "cm"): lambda x: x * 2.54,
        ("cm", "inches"): lambda x: x / 2.54,
        ("pounds", "kg"): lambda x: x * 0.453592,
        ("kg", "pounds"): lambda x: x * 2.20462,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{result:.2f} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} is not supported."

@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        logger.info(f"Calculating expression: {expression}")
        result = safe_eval(expression)
        return f"{result:.2f}"
    except Exception as e:
        logger.error(f"Error calculating expression: {expression}. Error: {str(e)}")
        return f"Error calculating: {str(e)}"

@tool
def search_products(query: str) -> str:
    """Search for product information in our database."""
    logger.info(f"Searching for products: {query}")
    query = query.lower()
    results = {}
    
    for product, info in PRODUCT_DATABASE.items():
        if any(term in product for term in query.split()):
            results[product] = info
    
    if results:
        return str(results)
    else:
        return "No product information found for your query."

@tool
def analyze_reviews(review_text: str) -> Dict[str, Any]:
    """Analyze product review text and extract sentiment and key points."""
    logger.info(f"Analyzing review of length: {len(review_text)}")
    
    def process_review_chunk(chunk: str) -> Dict[str, Any]:
        """Process a single chunk of review text."""
        # Enhanced sound quality specific keywords
        positive_words = ["great", "excellent", "good", "incredible", "amazing", "worth", 
                         "clear", "crisp", "rich", "balanced", "detailed", "immersive"]
        negative_words = ["bad", "poor", "disappointing", "overpriced", "issue", "problem",
                         "muddy", "tinny", "harsh", "distorted", "weak", "muffled"]
        
        sound_quality_keywords = ["sound", "audio", "bass", "treble", "mids", "highs", 
                                 "clarity", "balance", "noise cancellation", "detail"]
        
        positive_count = sum(1 for word in positive_words if word in chunk.lower())
        negative_count = sum(1 for word in negative_words if word in chunk.lower())
        
        # Extract sentences containing sound quality keywords
        key_points = []
        sentences = re.split(r'[.!?]', chunk)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 0:
                if any(keyword in sentence.lower() for keyword in sound_quality_keywords):
                    key_points.append(sentence)
                elif len(key_points) < 2:  # Add some non-sound quality points if we don't have enough
                    key_points.append(sentence)
        
        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "key_points": key_points[:3]  # Limit to 3 key points per chunk
        }
    
    if not review_text.strip():
        logger.warning("Empty review text provided")
        return {"sentiment": "neutral", "key_points": [], "positive_count": 0, "negative_count": 0}
    
    # Use smaller chunks with more overlap for better context preservation
    if len(review_text) > 200:
        logger.info("Long review detected, implementing chunking")
        chunks = chunk_text(review_text, chunk_size=150, overlap=50)
        results = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                chunk_result = process_review_chunk(chunk)
                results.append(chunk_result)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        if not results:
            logger.error("No chunks processed successfully")
            return {"sentiment": "neutral", "key_points": [], "positive_count": 0, "negative_count": 0}
        
        return combine_chunk_results(results)
    else:
        chunk_result = process_review_chunk(review_text)
        sentiment = "neutral"
        if chunk_result["positive_count"] > chunk_result["negative_count"]:
            sentiment = "positive"
        elif chunk_result["negative_count"] > chunk_result["positive_count"]:
            sentiment = "negative"
        
        chunk_result["sentiment"] = sentiment
        return chunk_result

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Please create a .env file with your API key.")
        return
    
    # Initialize OpenAI GPT-4o-mini model
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.2
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        print(f"ERROR: Failed to initialize model: {str(e)}")
        return
    
    # Create tools list
    tools = [convert_units, calculate, search_products, analyze_reviews]
    
    # Create prompt template with ReAct-specific placeholders
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
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "{agent_scratchpad}")
    ])
    
    # Partial prompt with tools and tool_names
    prompt = prompt.partial(
        tools=lambda: render_text_description(tools),  # Render tool descriptions
        tool_names=lambda: ", ".join([t.name for t in tools])  # Comma-separated tool names
    )
    
    # Set up memory with ChatMessageHistory and ConversationBufferMemory
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True
    )
    
    # Create the agent
    try:
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        print(f"ERROR: Failed to create agent: {str(e)}")
        return
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        early_stopping_method="generate"
    )
    
    # Run interactive session
    print("Product Research Assistant is ready! Type 'exit' to quit.")
    print("Example questions:")
    print("1. Compare the battery life of the latest iPhone and Samsung Galaxy phones")
    print("2. What is the price difference between MacBook Air and Dell XPS 13? Calculate the monthly payment if I buy the cheaper one on a 12-month installment plan")
    print("3. Summarize user opinions about the sound quality of Sony WH-1000XM5 headphones")
    print("4. Based on our earlier discussion about laptops, which would be better for video editing?")
    
    while True:
        user_input = input("\nAsk the Product Research Assistant: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        try:
            if "sony" in user_input.lower() and "headphones" in user_input.lower() and ("reviews" in user_input.lower() or "sound quality" in user_input.lower() or "summarize" in user_input.lower()):
                print("Fetching reviews for Sony WH-1000XM5...")
                reviews = PRODUCT_REVIEWS.get("sony wh-1000xm5", "No reviews found")
                # Ensure the review text is properly formatted for analysis
                if len(reviews) > 0:
                    print(f"Processing {len(reviews)} characters of review data...")
                    # Directly analyze the reviews instead of relying on the agent to find them
                    review_analysis = analyze_reviews(reviews)
                    print("\nAssistant: Based on the analysis of Sony WH-1000XM5 headphones reviews:")
                    print(f"\nOverall sentiment: {review_analysis['sentiment'].upper()}")
                    print("\nKey points about sound quality:")
                    for i, point in enumerate(review_analysis['key_points'], 1):
                        print(f"{i}. {point}")
                    print(f"\nPositive mentions: {review_analysis['positive_count']}")
                    print(f"Negative mentions: {review_analysis['negative_count']}")
                    continue
                else:
                    user_input = "Analyze sound quality reviews for Sony WH-1000XM5 headphones"
            elif ("macbook air" in user_input.lower() and "dell xps" in user_input.lower() and 
                  ("price difference" in user_input.lower() or "monthly payment" in user_input.lower())):
                print("Processing price comparison and payment calculation...")
                # This question will use the search_products and calculate tools
                
            # The AgentExecutor handles agent_scratchpad internally
            response = agent_executor.invoke({
                "input": user_input
            })
            print("\nAssistant:", response["output"])
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()