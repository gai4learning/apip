import streamlit as st

def show_get_started():
    st.title("🔑 Get Started with CUHK Azure OpenAI API")
    
    st.markdown("""
    Welcome to the CUHK Azure OpenAI API Tester! This tool helps you learn and experiment with the API.
    
    For detailed setup instructions and documentation, please check our GitHub repository's README.
    """)
    
    # Quick Reference
    st.header("📝 Quick Reference")
    
    # Rate Limits
    st.subheader("Rate Limits")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Calls per Minute", "5")
    with col2:
        st.metric("Calls per Week", "100")
    
    st.caption("You'll receive an email notification when you reach 75% of your weekly quota.")
    
    # Security Warning
    st.warning("""
    ⚠️ Important: Keep your API key secure!
    
    - Never share your key
    - Don't commit it to version control
    - Use environment variables
    """)
    
    # Need More?
    st.info("""
    💡 Need higher limits?
    
    Check the README for instructions on requesting expanded access for your project.
    """)

if __name__ == "__main__":
    show_get_started() 