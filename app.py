import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import logging
import datetime
import json
from pathlib import Path
from get_started import show_get_started
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"api_tester_{datetime.datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add handlers for all log levels
logger.setLevel(logging.DEBUG)

def compare_texts(client, texts, model="text-embedding-3-small"):
    """Compare multiple texts using embeddings.
    
    Args:
        client: The Azure OpenAI client
        texts: List of texts to compare
        model: The embedding model to use (default: text-embedding-3-small)
    
    Returns:
        DataFrame containing pairwise similarities
    """
    try:
        # Input validation
        if not texts or len(texts) < 2:
            raise ValueError("At least two texts are required for comparison")
        
        # Get embeddings for all texts
        logger.info(f"Getting embeddings for {len(texts)} texts using {model}")
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        embeddings = [data.embedding for data in response.data]
        
        # Calculate similarities between all pairs
        n = len(texts)
        similarities = []
        for i in range(n):
            row = []
            for j in range(n):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                # Format similarity score with color indicators
                row.append(f"{similarity:.4f}")
            similarities.append(row)
        
        # Create DataFrame with formatted headers
        df = pd.DataFrame(
            similarities,
            columns=[f"Text {i+1}" for i in range(n)],
            index=[f"Text {i+1}" for i in range(n)]
        )
        
        logger.info("Text comparison completed successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error in text comparison: {str(e)}")
        if "rate limit" in str(e).lower():
            raise Exception("Rate limit reached. Please wait and try again.") from e
        elif "authentication" in str(e).lower():
            raise Exception("Authentication failed. Please check your API key.") from e
        elif "quota" in str(e).lower():
            raise Exception("Weekly quota exceeded.") from e
        else:
            raise Exception(f"Error comparing texts: {str(e)}") from e

# Load environment variables
load_dotenv()

# Security Settings
MAX_LOG_SIZE = 1024 * 1024  # 1MB
MAX_PROMPT_LENGTH = 4000
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Constants - matching the notebook exactly
AZURE_ENDPOINT = "https://cuhk-apip.azure-api.net"
MODELS = {
    "gpt-4o-mini": {
        "type": "chat",
        "description": "Fastest and most cost-effective model for general text generation and conversations"
    },
    "gpt-4o": {
        "type": "chat",
        "description": "More powerful model for complex reasoning and detailed responses"
    },
    "o1-mini": {
        "type": "chat",
        "description": "Latest model optimized for efficiency and improved responses"
    },
    "dall-e-3": {
        "type": "image",
        "description": "Create high-quality images from text descriptions",
        "api_version": "2024-02-01"  # DALL-E 3 requires this specific API version
    },
    "text-embedding-ada-002": {
        "type": "embedding",
        "description": "Legacy embedding model (1536 dimensions)"
    },
    "text-embedding-3-small": {
        "type": "embedding",
        "description": "Latest embedding model with improved performance (1536 dimensions)"
    }
}
API_VERSION = "2023-05-15"  # Default API version for most models

# Initialize session state
if 'show_code' not in st.session_state:
    st.session_state.show_code = False
    logger.debug("Code examples visibility initialized")
if 'api_calls_count' not in st.session_state:
    st.session_state.api_calls_count = 0
    logger.info("API calls counter initialized")
if 'page' not in st.session_state:
    st.session_state.page = 'API Tester'
    logger.info("Default page set to API Tester")
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = None
    logger.warning("API call timestamp not found, initializing to None")

# Security functions
def validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    if not api_key:
        return False
    # Check if key matches expected format (32 hex characters)
    return len(api_key) == 32 and all(c in '0123456789abcdef' for c in api_key.lower())

def sanitize_prompt(prompt: str) -> str:
    """Sanitize user input prompt."""
    if not prompt:
        return ""
    # Remove any potential harmful characters
    return prompt.strip()[:MAX_PROMPT_LENGTH]

def check_rate_limit() -> bool:
    """Check if we're within rate limits."""
    if not st.session_state.last_api_call:
        return True
    
    time_since_last_call = datetime.datetime.now() - st.session_state.last_api_call
    return time_since_last_call.total_seconds() >= 12  # Ensure minimum 12 seconds between calls

def rotate_logs():
    """Rotate logs if they get too large."""
    if log_file.exists() and log_file.stat().st_size > MAX_LOG_SIZE:
        backup = log_file.with_suffix('.log.1')
        if log_file.exists():
            log_file.rename(backup)

# Page config with security headers
st.set_page_config(
    page_title="CUHK Azure OpenAI API Tester",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "CUHK Azure OpenAI API Tester - For Academic Use Only"
    }
)

# Sidebar navigation
with st.sidebar:
    st.title("📚 Navigation")
    
    # Navigation buttons
    if st.button("🔑 Get Started Guide", use_container_width=True):
        st.session_state.page = 'Get Started'
    if st.button("🤖 API Tester", use_container_width=True):
        st.session_state.page = 'API Tester'
    
    st.divider()
    
    # Rest of the sidebar content...
    st.title("📊 API Usage")
    
    # Log viewer expander
    with st.expander("View Logs", expanded=False):
        if log_file.exists():
            # Create two columns for the buttons
            col1, col2 = st.columns(2)
            
            # Refresh button in first column
            if col1.button("🔄 Refresh Logs"):
                st.rerun()
            
            # Clear button in second column
            if col2.button("🗑️ Clear Logs"):
                try:
                    with open(log_file, 'w') as f:
                        f.write('')
                    st.success("Logs cleared!")
                    logger.info("Logs cleared by user")
                except Exception as e:
                    logger.error(f"Error clearing logs: {str(e)}")
                    st.error(f"Error clearing logs: {str(e)}")
            
            # Add some space before showing logs
            st.markdown("---")
            
            # Display logs
            with open(log_file, 'r') as f:
                logs = f.readlines()
                logs.reverse()  # Show most recent logs first
                # Filter and colorize logs based on level
                log_display = []
                for log in logs[:10]:  # Only show last 10 logs
                    if "WARNING" in log:
                        log_display.append(f"🟡 {log}")
                    elif "ERROR" in log:
                        log_display.append(f"🔴 {log}")
                    elif "DEBUG" in log:
                        log_display.append(f"🔵 {log}")
                    else:  # INFO
                        log_display.append(f"⚪ {log}")
                st.text_area("Recent Logs (Last 10)", value=''.join(log_display), height=200)
        else:
            st.info("No logs available yet")
        
        # Add log stats
        st.metric("API Calls Count", st.session_state.api_calls_count)

    # Code examples section
    st.title("💻 Code Examples")
    if st.button("Toggle Examples"):
        st.session_state.show_code = not st.session_state.show_code
    
    if st.session_state.show_code:
        with st.expander("1. Basic Setup", expanded=False):
            st.code('''
from openai import AzureOpenAI

# Initialize the client
client = AzureOpenAI(
    azure_endpoint="https://cuhk-apip.azure-api.net",
    api_version="2024-02-01",  # Use appropriate version for your model
    api_key="your_api_key_here"
)''', language='python')
            
        with st.expander("2. Chat Models (gpt-4o-mini/gpt-4o/o1-mini)", expanded=False):
            st.code('''
# Chat with gpt-4o-mini or gpt-4o
response = client.chat.completions.create(
    model="gpt-4o-mini",  # or "gpt-4o"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,  # Control response creativity (0-1)
    max_tokens=150    # Limit response length
)
print(response.choices[0].message.content)

# Chat with o1-mini (note: doesn't support system messages)
response = client.chat.completions.create(
    model="o1-mini",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ]
)
print(response.choices[0].message.content)''', language='python')

        with st.expander("3. Image Generation (dall-e-3)", expanded=False):
            st.code('''
# Generate an image with DALL-E 3
response = client.images.generate(
    model="dall-e-3",
    prompt="A beautiful sunset view of CUHK campus",
    n=1,
    size="1024x1024",  # Options: 1024x1024, 1024x1792, 1792x1024
    quality="standard",  # "standard" or "hd"
    style="natural"     # "natural" or "vivid"
)

# Get the image URL
image_url = response.data[0].url

# If you want to save the image:
import requests
from PIL import Image
from io import BytesIO

response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image.save("generated_image.png")''', language='python')
            
        with st.expander("4. Text Embeddings", expanded=False):
            st.code('''
# Get embedding using latest model
response = client.embeddings.create(
    model="text-embedding-3-small",  # or "text-embedding-ada-002" for legacy
    input="The quick brown fox jumps over the lazy dog"
)
embedding = response.data[0].embedding  # 1536-dimensional vector

# Compare text similarity
from scipy.spatial.distance import cosine
import numpy as np

def get_similarity(text1, text2, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=[text1, text2]
    )
    emb1 = response.data[0].embedding
    emb2 = response.data[1].embedding
    return 1 - cosine(emb1, emb2)  # Cosine similarity''', language='python')
            
        with st.expander("5. Error Handling", expanded=False):
            st.code('''
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except Exception as e:
    if "rate limit" in str(e).lower():
        print("Rate limit reached (5 calls/minute, 100 calls/week)")
    elif "authentication" in str(e).lower():
        print("Check your API key")
    elif "quota" in str(e).lower():
        print("Weekly quota reached")
    else:
        print(f"Error: {str(e)}")''', language='python')

        with st.expander("Important Notes", expanded=False):
            st.markdown("""
            ### Rate Limits
            - 5 calls per minute
            - 100 calls per week maximum
            - Email notification at 75% quota
            
            ### Available Models
            
            #### Chat Models
            - `gpt-4o-mini`: Fast responses for general text generation and conversations
            - `gpt-4o`: More powerful model for complex reasoning and detailed responses
            - `o1-mini`: Latest model optimized for efficiency and improved responses

            #### Image Generation
            - `dall-e-3`: Create high-quality images from text descriptions
              - Sizes: 1024x1024, 1024x1792, 1792x1024
              - Quality: standard, hd
              - Style: natural, vivid

            #### Embedding Models
            - `text-embedding-ada-002`: Legacy embedding model (1536 dimensions)
            - `text-embedding-3-small`: Latest embedding model with improved performance (1536 dimensions)
            
            ### Security Best Practices
            - Never share your API key
            - Use environment variables
            - Run locally only
            - Monitor your usage
            """)

# Display the selected page
if st.session_state.page == 'Get Started':
    show_get_started()
else:
    # Main API Tester content
    st.title("🎓 CUHK Azure OpenAI API Tester")
    
    # Rate limit warning
    st.warning("""
    ⚠️ **Usage Limits**:
    - 5 calls per minute
    - 100 calls per week maximum
    - Email notification at 75% of quota
    - Need an API key? Click "🔑 Get Started Guide" in the sidebar!
    """)
    
    st.markdown("""
    This app allows you to test the CUHK Azure OpenAI API. Please enter your API key and select a test case to try.
    """)
    
    # API Key input
    api_key = st.text_input("Enter your API Key", type="password", value=os.getenv("AZURE_API_KEY", ""))

    # Validate API key
    if api_key and not validate_api_key(api_key):
        st.error("Invalid API key format. Please check your key.")
        st.stop()

    # Test cases for chat models
    chat_test_cases = {
        "Course Assistant": {
            "messages": [
                {"role": "system", "content": "You are a helpful teaching assistant for CUHK students."},
                {"role": "user", "content": "Can you explain what is machine learning and give a simple example that CUHK students can relate to?"}
            ]
        },
        "Code Helper": {
            "messages": [
                {"role": "system", "content": "You are a Python programming expert."},
                {"role": "user", "content": """Help me understand this code and suggest improvements:

def process_student_grades(grades):
    sum = 0
    for i in range(len(grades)):
        sum = sum + grades[i]
    return sum/len(grades)"""}
            ]
        },
        "Research Assistant": {
            "messages": [
                {"role": "system", "content": "You are a research assistant helping with academic writing."},
                {"role": "user", "content": "I'm writing a paper about artificial intelligence in education. Can you help me structure an introduction paragraph that highlights the importance of this topic?"}
            ]
        }
    }

    # Test cases for image generation
    image_test_cases = {
        "Campus Scene": "A photorealistic image of CUHK campus with the Goddess of Democracy statue in the foreground and modern academic buildings in the background, during a sunny day.",
        "Technical Diagram": "A clear technical diagram explaining how neural networks process information, with labeled nodes and connections, suitable for an academic presentation.",
        "Educational Visual": "An informative illustration showing the concept of machine learning, with a computer analyzing different types of data represented by colorful icons and arrows."
    }

    # Test cases for embedding model
    embedding_test_cases = {
        "Single Text": "Artificial Intelligence and Machine Learning are transforming education at CUHK, enabling personalized learning experiences and innovative teaching methods.",
        "Compare Texts": [
            "CUHK offers courses in artificial intelligence and machine learning",
            "The Chinese University provides AI and ML education programs",
            "HKUST has computer science and engineering departments"
        ]
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        # Model selection
        selected_model = st.selectbox(
            "Select a model",
            list(MODELS.keys()),
            format_func=lambda x: f"{x} - {MODELS[x]['description']}"
        )

        # Initialize OpenAI client
        client = None
        if api_key:
            try:
                logger.info("Attempting to initialize Azure OpenAI client")
                # Use model-specific API version if available
                api_version = MODELS[selected_model].get("api_version", API_VERSION)
                client = AzureOpenAI(
                    azure_endpoint=AZURE_ENDPOINT,
                    api_version=api_version,
                    api_key=api_key
                )
                logger.info("Azure OpenAI client initialized successfully")
                st.success("API key accepted! You can now test the API.")
            except Exception as e:
                logger.error(f"Error initializing client: {str(e)}")
                st.error(f"Error initializing client: {str(e)}")
                st.info("""
                Please check:
                1. Your API key is correct (it should be in the format like '867bad37dee74bd69076e99dfcbe0596')
                2. You have access to the CUHK Azure OpenAI service
                3. The API endpoint is accessible from your network
                """)

        model_type = MODELS[selected_model]["type"]
        
        if model_type == "chat":
            # Test case selection for chat models
            selected_test = st.selectbox("Select a test case", list(chat_test_cases.keys()))
            
            # Custom prompt input
            st.subheader("Or enter your own prompt")
            custom_prompt = st.text_area("Enter your prompt", height=100)
            if custom_prompt:
                custom_prompt = sanitize_prompt(custom_prompt)
        
        elif model_type == "image":
            # Test case selection for image generation
            selected_test = st.selectbox("Select a test case", list(image_test_cases.keys()))
            
            # Custom prompt input
            st.subheader("Or enter your own image description")
            custom_prompt = st.text_area(
                "Enter a detailed description of the image you want to create",
                height=100,
                help="Be specific about what you want in the image. Include details about style, composition, and content."
            )
            if custom_prompt:
                custom_prompt = sanitize_prompt(custom_prompt)
            
            # Image size selection
            size = st.selectbox(
                "Select image size",
                ["1024x1024", "1024x1792", "1792x1024"],
                help="Choose the dimensions of your generated image"
            )
            
            # Quality selection
            quality = st.selectbox(
                "Select image quality",
                ["standard", "hd"],
                help="HD quality provides more detail but uses more credits"
            )
            
            # Style selection
            style = st.selectbox(
                "Select image style",
                ["natural", "vivid"],
                help="Natural for photorealistic images, vivid for more artistic ones"
            )
        
        else:  # embedding models
            # Test case selection for embedding model
            selected_test = st.selectbox("Select a test case", list(embedding_test_cases.keys()))
            
            if selected_test == "Single Text":
                custom_prompt = st.text_area(
                    "Enter text for embedding", 
                    value=embedding_test_cases[selected_test],
                    height=100
                )
            else:
                st.subheader("Compare multiple texts")
                texts = []
                for i in range(3):
                    text = st.text_area(
                        f"Text {i+1}", 
                        value=embedding_test_cases[selected_test][i],
                        height=50
                    )
                    texts.append(text)
                custom_prompt = texts

    with col2:
        st.info("""💡 **Tips**: 
        
        **Chat Models**:
        - GPT-4o-mini: Quick responses
        - GPT-4o: Complex tasks
        - o1-mini: Latest & efficient
        
        **Image Generation**:
        - DALL-E 3: Create detailed images
        - Use clear descriptions
        - Specify style and content
        
        **Embeddings**:
        - text-embedding-ada-002: Legacy
        - text-embedding-3-small: Latest
        
        Check the sidebar for examples!
        """)

    # Helper function to convert usage to dict
    def usage_to_dict(usage):
        if usage:
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                "total_tokens": usage.total_tokens
            }
        return {}

    # Submit button
    if st.button("Submit", type="primary"):
        if not client:
            logger.warning("Attempted to submit without valid client")
            st.error("Please enter your API key first.")
        elif not check_rate_limit():
            logger.warning("Rate limit check failed")
            st.error("Please wait a few seconds between API calls.")
        else:
            try:
                with st.spinner("Getting response..."):
                    logger.info(f"Making API call to {selected_model}...")
                    start_time = datetime.datetime.now()
                    
                    # Update last API call time
                    st.session_state.last_api_call = datetime.datetime.now()
                    st.session_state.api_calls_count += 1

                    # Rotate logs if needed
                    rotate_logs()

                    if model_type == "chat":
                        if custom_prompt:
                            # For o1-mini, don't include system message
                            if selected_model == "o1-mini":
                                messages = [
                                    {"role": "user", "content": custom_prompt}
                                ]
                            else:
                                messages = [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": custom_prompt}
                                ]
                            logger.info(f"Custom prompt submitted: {custom_prompt[:50]}...")
                        else:
                            # For o1-mini, use only the user messages from test cases
                            if selected_model == "o1-mini":
                                messages = [msg for msg in chat_test_cases[selected_test]["messages"] if msg["role"] == "user"]
                            else:
                                messages = chat_test_cases[selected_test]["messages"]
                            logger.info(f"Test case submitted: {selected_test}")

                        response = client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            temperature=1  # Matching the notebook default
                        )
                        
                        end_time = datetime.datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        # Convert usage to dictionary for logging
                        usage_dict = usage_to_dict(response.usage)
                        logger.info(f"Usage stats: {json.dumps(usage_dict)}")
                        
                        # Display response in a clean container
                        with st.container():
                            st.subheader("Response")
                            st.write(response.choices[0].message.content)
                            
                            # Display usage statistics
                            with st.expander("Usage Statistics"):
                                st.json(usage_dict)
                                st.metric("Response Time", f"{duration:.2f}s")
                    
                    elif model_type == "image":
                        prompt = custom_prompt if custom_prompt else image_test_cases[selected_test]
                        
                        # Get the correct model configuration
                        model_config = MODELS[selected_model]
                        deployment_name = model_config.get("deployment_name", selected_model)
                        
                        response = client.images.generate(
                            model=deployment_name,  # Use the deployment name for the API call
                            prompt=prompt,
                            n=1,
                            size=size,
                            quality=quality,
                            style=style
                        )
                        
                        end_time = datetime.datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        # Display the generated image
                        with st.container():
                            st.subheader("Generated Image")
                            st.image(response.data[0].url, caption=prompt)
                            
                            # Display generation details
                            with st.expander("Generation Details"):
                                st.json({
                                    "size": size,
                                    "quality": quality,
                                    "style": style,
                                    "duration": f"{duration:.2f}s"
                                })
                    
                    else:  # embedding models
                        if isinstance(custom_prompt, list):
                            # Multiple texts for comparison
                            similarities = compare_texts(client, custom_prompt, selected_model)
                            
                            end_time = datetime.datetime.now()
                            duration = (end_time - start_time).total_seconds()
                            
                            # Display similarities with enhanced visualization
                            with st.container():
                                st.subheader("Text Similarities")
                                st.dataframe(
                                    similarities,
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # Add explanation
                                st.caption("""
                                Similarity scores range from 0 to 1:
                                - 1.0000: Identical texts
                                - >0.8000: Very similar content
                                - 0.5000-0.8000: Moderately similar
                                - <0.5000: Different content
                                """)
                        
                        else:
                            # Single text embedding
                            response = client.embeddings.create(
                                model=selected_model,
                                input=custom_prompt
                            )
                            
                            end_time = datetime.datetime.now()
                            duration = (end_time - start_time).total_seconds()
                            
                            embedding = response.data[0].embedding
                            
                            # Display the embedding
                            with st.container():
                                st.subheader("Embedding Vector")
                                st.write(f"Dimensions: {len(embedding)}")
                                
                                # Show first few dimensions
                                preview = embedding[:10]
                                st.write("First 10 dimensions:")
                                st.json(preview)
                                
                                # Option to download full embedding
                                full_embedding = np.array(embedding)
                                st.download_button(
                                    "Download Full Embedding (NPY)",
                                    data=full_embedding.tobytes(),
                                    file_name="embedding.npy",
                                    mime="application/octet-stream"
                                )
                        
                        # Display timing
                        st.metric("Response Time", f"{duration:.2f}s")
                    
                    logger.info(f"API call successful. Duration: {duration:.2f}s")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"API call failed: {error_msg}")
                st.error(f"Error: {error_msg}")
                
                # Check for specific error types
                if "rate limit" in error_msg.lower():
                    logger.warning("Rate limit detected!")
                    st.warning("You've hit the rate limit. Please wait a few minutes before trying again.")
                elif "authentication" in error_msg.lower():
                    logger.warning("Authentication error detected!")
                    st.warning("Authentication failed. Please check your API key.")
                elif "JSON" in error_msg:
                    logger.error("JSON serialization error")
                    st.error("There was an error processing the response. Please try again.")
                
                st.info("""
                If you're seeing an error, please check:
                1. Your API key is correct
                2. You have access to the CUHK Azure OpenAI service
                3. The API endpoint is accessible from your network
                4. The model name is correct
                """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ for CUHK Students</p>
        <p>For more information, visit the <a href='https://cuhk-apip.developer.azure-api.net/'>CUHK Azure OpenAI API Documentation</a></p>
    </div>
    """, unsafe_allow_html=True) 