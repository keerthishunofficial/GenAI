import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Support Bot Lab",
    page_icon="🤖",
    layout="wide"
)

# --- PROMPT TEMPLATES ---

zero_shot_template = """You are a polite and professional customer support chatbot.
Answer the customer's query underneath.

Customer Query: {user_input}
Support Response:"""

few_shot_template = """You are a helpful and professional customer support chatbot.
Here are some examples of how to respond:

Customer: Where is my order? 
Support: I'd be happy to help you track your order! Could you please provide your 8-digit order number?

Customer: Can I get a refund for this broken item?
Support: I am so sorry to hear that your item arrived broken. We can certainly process a refund or replacement for you. Please share your order number and a photo of the damaged item.

Customer: The app is extremely slow and terrible!
Support: Thanks for reaching out, and I apologize for the frustration! Our engineering team is currently looking into performance issues. Could you let me know what device you are using?

Now, respond to the following customer query:
Customer: {user_input}
Support Response:"""

cot_template = """You are a helpful and professional customer support chatbot.
Before answering the customer, please think step-by-step.

1. Identify the core intent of the customer's query (refund, status, complaint, etc.).
2. Determine if any specific information (like an order number) is needed.
3. Formulate a polite and empathetic opening.
4. Construct the final response based on steps 1-3.

Answer the user thoughtfully using the exact following format:
<thinking>
Step 1: [intent]
Step 2: [info needed]
Step 3: [opening]
Step 4: [response formulation]
</thinking>
Final Support Response: [the actual polite response to the customer]

Customer Query: {user_input}"""


# --- BACKEND LOGIC ---

def get_chatbot_response(api_key, strategy, user_input):
    """
    Calls the Groq API using the selected prompt engineering strategy.
    """
    client = Groq(api_key=api_key)
    
    # Decide which prompt to use based on user selection
    if strategy == "Zero-Shot Prompting":
        prompt = zero_shot_template.format(user_input=user_input)
    elif strategy == "Few-Shot Prompting":
        prompt = few_shot_template.format(user_input=user_input)
    elif strategy == "Chain-of-Thought Prompting":
        prompt = cot_template.format(user_input=user_input)
    else:
        prompt = user_input
        
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.5,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

# --- STREAMLIT UI ---

st.title("🤖 Customer Support Chatbot Lab")
st.markdown("Experiment with how **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought** prompting change model behavior.")

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
default_key = os.getenv("GROQ_API_KEY", "")
api_key = st.sidebar.text_input("Enter Groq API Key:", value=default_key, type="password")

# Sanitize the key in case of accidental quotes or whitespace paste
api_key = api_key.strip().strip("\"'")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Example Queries to Try:")
st.sidebar.code("My headphones broke after 2 days!")
st.sidebar.code("Where is order #48291?")
st.sidebar.code("I'm furious, you billed me twice!")

# Main chat interface
tabs = st.tabs(["💬 Single Strategy Chat", "📊 Compare All Strategies"])

with tabs[0]:
    strategy = st.radio(
        "Select Prompt Strategy:",
        ("Zero-Shot Prompting", "Few-Shot Prompting", "Chain-of-Thought Prompting"),
        horizontal=True
    )

    st.markdown("---")

    user_input = st.text_area("Customer Query:", height=100, placeholder="Type a customer support request here...", key="single")

    if st.button("Submit to Chatbot", type="primary"):
        if not api_key:
            st.error("Please enter a Groq API Key in the sidebar.")
        elif not user_input.strip():
            st.warning("Please type a message first.")
        else:
            with st.spinner(f"Generating response using {strategy}..."):
                output = get_chatbot_response(api_key, strategy, user_input)
            
            st.markdown("### 💬 Chatbot Response:")
            st.info(output)

with tabs[1]:
    st.markdown("### 📊 Prompt Strategy Comparison")
    st.markdown("See exactly how the three strategies perform side-by-side on the same query to observe quality differences.")
    
    compare_input = st.text_area("Customer Query:", height=100, placeholder="Type a request to see how all three respond...", key="compare")
    
    if st.button("Run Comparison", type="primary"):
        if not api_key:
            st.error("Please enter a Groq API Key in the sidebar.")
        elif not compare_input.strip():
            st.warning("Please type a message first.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Zero-Shot")
                with st.spinner("Generating..."):
                    out1 = get_chatbot_response(api_key, "Zero-Shot Prompting", compare_input)
                st.info(out1)
                
            with col2:
                st.subheader("Few-Shot")
                with st.spinner("Generating..."):
                    out2 = get_chatbot_response(api_key, "Few-Shot Prompting", compare_input)
                st.success(out2)
                
            with col3:
                st.subheader("Chain-of-Thought")
                with st.spinner("Generating..."):
                    out3 = get_chatbot_response(api_key, "Chain-of-Thought Prompting", compare_input)
                st.warning(out3)

