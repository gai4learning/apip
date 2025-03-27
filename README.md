# CUHK Azure OpenAI API Tester

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple Streamlit web application for testing the CUHK Azure OpenAI API. This tool helps students learn and experiment with the API while monitoring their usage.

⚠️ **IMPORTANT: Local Development Only**
- This repository is for **learning purposes only**
- Run the application **locally** on your machine
- Do NOT deploy to cloud platforms
- Keep your API keys secure and never expose them online
- For learning use only

## 🛡️ Security Notice

This application is designed for local development and learning purposes only. Deploying it to the cloud or public servers poses several risks:

1. **API Key Exposure**: Your CUHK Azure OpenAI API key could be exposed, leading to:
   - Unauthorized usage of your quota
   - Potential misuse of CUHK's resources
   - Risk of exceeding rate limits

2. **Access Control**: The app doesn't implement:
   - User authentication
   - Rate limiting
   - IP restrictions
   - Other security measures needed for production

3. **Compliance**: Public deployment may violate:
   - CUHK's API usage policies
   - Data protection requirements
   - Service terms and conditions

**Always run this application locally and never share your API keys!**

## 📚 Academic Honesty Notice

CUHK places very high importance on academic honesty. Use of these APIs must comply with [CUHK's Policy on Academic Honesty](https://www.cuhk.edu.hk/policy/academichonesty/):

1. **Appropriate Use**:
   - Use the API as a learning tool to understand AI capabilities
   - Practice responsible AI development
   - Enhance your understanding of AI concepts

2. **Prohibited Uses**:
   - Do NOT use the API to generate assignments or exam answers
   - Do NOT submit AI-generated content as your own work
   - Do NOT use the API to assist in any form of academic dishonesty

3. **Consequences**:
   - CUHK maintains a zero-tolerance policy on academic dishonesty
   - Violations may lead to disciplinary action
   - Serious cases may result in termination of studies

Always consult with your professors about appropriate use of AI tools in your coursework. When in doubt, ask first!

## 🔒 Security Best Practices

1. **API Key Protection**:
   - NEVER share your API key with anyone
   - NEVER commit your `.env` file to version control
   - NEVER expose your key in public repositories or forums
   - NEVER include your key in screenshots or shared code
   - Immediately report if you suspect your key has been compromised

2. **Local Development Only**:
   - This app is designed for local use only
   - DO NOT deploy to public servers or cloud platforms
   - DO NOT expose the app outside your local network
   - Always run on localhost (127.0.0.1)

3. **Data Privacy**:
   - Be mindful of what data you send to the API
   - DO NOT submit sensitive or personal information
   - DO NOT submit confidential university data
   - Consider data privacy regulations when using the service



## ⚠️ AI Content Disclaimer

Please be aware of the following regarding AI-generated content:

1. **Accuracy**: While AI models strive for accuracy, they may occasionally:
   - Provide incorrect or outdated information
   - Make mistakes in calculations or reasoning
   - Generate plausible-sounding but inaccurate responses

2. **Limitations**:
   - AI responses should be verified, especially for critical applications
   - Models may exhibit biases present in their training data
   - Complex or nuanced topics may require human expertise and verification

3. **Best Practices**:
   - Always verify important information from authoritative sources
   - Use AI-generated content as a starting point, not the final answer
   - Exercise critical thinking and professional judgment
   - When in doubt, consult with professors or domain experts

## 💡 Features

- Test API calls with predefined cases
- Monitor API usage and rate limits
- View code examples
- Track API response times
- Export usage statistics

## ⚠️ Important Notes

1. This is a learning tool - use it responsibly
2. Keep your API key secure
3. Monitor your usage to avoid hitting limits
4. Plan ahead if you need expanded access 

## 🚀 Getting Started

### Prerequisites
- CUHK account
- Python 3.8 or higher
- Git (for cloning the repository)

1. **Prerequisites**:
   ```bash
   # Make sure you have Python 3.8+ installed
   python --version
   
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Installation**:
   ```bash
   # Clone this repository
   git clone https://github.com/gai4learning/apip.git
   cd apip
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your API key
   ```

3. **Running the App**:
   ```bash
   streamlit run app.py
   ```

4. **First Time Setup**:
   - Get your API key (see Getting Your API Key below)
   - Enter your key in the app
   - Test with a simple prompt first
   - Check the logs to ensure everything works

5. **Troubleshooting**:
   - Check the logs in the sidebar
   - Verify your API key is correct
   - Ensure you're within rate limits
   - Read error messages carefully

### Getting Your API Key

1. **Access the Portal**
   - Go to [CUHK API Developer Portal](https://cuhk-apip.developer.azure-api.net/)
   - Sign up / in with your CUHK account (top right corner)

2. **Subscribe to Starter Plan**
   - Navigate to "Products" in the top menu
   - Click on "Starter"
   - Under "Your subscription", enter "Starter" as "Your new product subscription name"
   - Click "Subscribe"
   - You should be automatically re-directed to "Profile" to review the Subscription information

3. **Get Your API Keys**
   - After approval, you'll see both Primary and Secondary keys
   - Use either key in your `.env` file
   - Regenerate your keys as needed or regularly

4. **Monitor Your usage**
   - Regularly review your usage of the APIs under "Reports"

### Available Models

1. **Chat Models**:
   - `gpt-4o-mini`: Quick responses, good for testing and simple tasks
   - `gpt-4o`: Complex reasoning, detailed explanations, and longer contexts
   - `o1-mini`: Latest model with improved efficiency and quality

2. **Image Generation**:
   - `dall-e-3`: Create detailed images from text descriptions
     - Photorealistic images
     - Artistic illustrations
     - Technical diagrams
     - Educational visuals

3. **Embedding Models**:
   - `text-embedding-ada-002`: Legacy model for:
     - Basic semantic search
     - Text similarity
     - Content clustering
   - `text-embedding-3-small`: Latest model for:
     - Improved semantic understanding
     - Better multilingual support
     - More accurate similarity comparisons

## 📊 Usage Limits

### Starter Subscription Limits
- 5 calls per minute
- 100 calls per week maximum
- Email notification at 75% of quota
- Weekly quota renewal

## ⚠️ Usage Monitoring

Your API usage is monitored by CUHK. Please note:

1. **Usage Tracking**:
   - All API calls are logged and monitored
   - Usage patterns are analyzed for abnormal behavior
   - Rate limits and weekly quotas are strictly enforced

2. **Warning**:
   Excessive or abnormal usage patterns and use of the APIs for purposes other than learning will result in:
   - Immediate suspension of your API access
   - Investigation of usage patterns
   - Reporting to relevant academic units
   - Possible disciplinary actions

3. **Prohibited Activities**:
   - Attempting to circumvent rate limits
   - Sharing API keys with others
   - Using the service for non-academic purposes
   - Automated mass requests
   - Any form of commercial use

Use the API responsibly and only for authorized academic purposes. If you need increased limits for legitimate academic work, please follow the proper channels to request expanded access.

## 📈 Need More Access?

If you need higher rate limits or access to additional features, you'll need to request expanded access.

### When to Request More Access
- Hitting rate limits frequently
- Need higher throughput
- Require access to more models
- Working on research projects

### How to Request

1. **Prepare Information**:
   - Project name and description
   - Course code (if applicable)
   - Expected duration
   - Usage requirements
   - Technical justification

2. **Submit Request**:
   - Access the ITSC Service Desk System:
     - [ServiceDesk Portal](http://servicedesk.itsc.cuhk.edu.hk)
   - Create a new General Enquiry under Category E-Learning Service
   - Category: "Azure OpenAI API - Access Extension Request"
   - Include your supervisor's information if applicable

3. **Request Template**:
```
Subject: Azure OpenAI API - Access Extension Request

Project Details:
- Name: [Your Project Name]
- Course: [Course Code, if applicable]
- Duration: [Expected timeline]

Current Limitations:
- [Describe what limits you're hitting]

Requested Access:
- [Specify needed rate limits]
- [List required models]

Justification:
- [Brief explanation]

```

## 📚 Additional Resources

- [CUHK API Portal for Teaching and Learning](https://cuhk-apip.developer.azure-api.net/)
- [Azure OpenAI Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
- [Azure OpenAI API Specs Version](https://github.com/Azure/azure-rest-api-specs/tree/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference)
- [Azure OpenAI Service Quotas and Limits](https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits)
- [OpenAI API Key Safety Best Practices](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
- [OpenAI Best Practices for API Usage](https://platform.openai.com/docs/guides/rate-limits/best-practices)

