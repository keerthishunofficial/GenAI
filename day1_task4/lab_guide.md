# Prompt Engineering Lab: Customer Support Chatbot

This guide demonstrates how to build a simple, responsive customer support chatbot using Python, Streamlit, and a free-tier LLM (Groq API). We will explore three prompt engineering strategies to observe how LLMs alter their behavior and response quality based on our instructions.

## 1. Architecture Overview

- **Frontend (UI)**: **Streamlit** - A lightweight Python framework used to build our chat interface where users can select strategies and input queries.
- **Backend Logic**: **Python** - Handles the API requests, injecting user queries into our structured prompt templates.
- **LLM Provider**: **Groq API** - A free-tier, high-speed LLM inference endpoint. We will use the `llama-3.3-70b-versatile` model.
- **Prompt Strategies**:
  1. **Zero-Shot**: Pure instructions with no given examples.
  2. **Few-Shot**: Includes prior examples to align the model's answering pattern and tone.
  3. **Chain-of-Thought (CoT)**: Forces the model to explicitly reason through the query before answering.

## 2. Environment Setup

### Prerequisites
You need Python 3.8+ installed on your machine. You will also need a free API key from [Groq Console](https://console.groq.com/keys).

### Installation
1. Open your terminal in the project directory.
2. Install the necessary dependencies:
```bash
pip install streamlit groq jupyter python-dotenv
```

3. Set your API Key in an `.env` file (or provide it in the Streamlit UI):
```properties
GROQ_API_KEY="your_api_key_here"
```

## 3. Step-by-Step Implementation

### Step 3.1: Designing Prompt Templates

Let's define the three strategies specifically for our customer support use-cases (order tracking, refunds, product info, complaints).

**1. Zero-Shot Prompt**
Basic context and instruction. The LLM must figure out how to respond purely based on conversational norms.
```python
zero_shot_template = """You are a polite and professional customer support chatbot.
Answer the customer's query underneath.

Customer Query: {user_input}
Support Response:"""
```

**2. Few-Shot Prompt**
Provide a few exact examples of how we want the chatbot to sound. This drastically improves consistency.
```python
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
```

**3. Chain-of-Thought (CoT) Prompt**
Forces the model to rationalize its approach before outputting the final answer. This is excellent for complex logic or ensuring it doesn't forget constraints.
```python
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

Customer Query: {user_input}
"""
```

### Step 3.2: Chatbot Backend Logic
We will create a function that takes the strategy, user query, and API key, and calls the Groq API.

```python
from groq import Groq

def get_chatbot_response(api_key, strategy, user_input):
    client = Groq(api_key=api_key)
    
    # Select Template
    if strategy == "Zero-Shot":
        prompt = zero_shot_template.format(user_input=user_input)
    elif strategy == "Few-Shot":
        prompt = few_shot_template.format(user_input=user_input)
    elif strategy == "Chain-of-Thought":
        prompt = cot_template.format(user_input=user_input)
        
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
    )
    return response.choices[0].message.content
```

### Step 3.3: Connecting the Streamlit Interface
We tie the logic and prompts together into an interactive web interface using `streamlit`.

```python
import streamlit as st

st.set_page_config(page_title="Support Bot GenAI Lab", page_icon="🤖")

st.title("🤖 Customer Support Chatbot Lab")
st.markdown("Evaluate **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought** prompting strategies.")

# ... sidebar inputs and event listeners ... (see app.py)
```

## 4. Observations & Self-Check
To verify the differences, try running the query: *"My coffee mug arrived shattered."*

- **Zero-Shot**: Will likely be generic. Example: *"I'm sorry to hear that. Contact our team at support@email.com for help."*
- **Few-Shot**: Will explicitly follow the structure laid out in the examples. Example: *"I am so sorry to hear that your item arrived broken. We can certainly process a refund or replacement for you. Please share your order number and a photo of the damaged item."*
- **Chain-of-Thought**: Will show its internal logic formatting.
  *Step 1: intent is refund/replace for broken item.*
  *Step 2: need photo and order number.*
  *Step 3: empathetic apology.* 
  And finally, output a highly structured, accurate response.

---
*Note: The complete runnable code is automatically generated in your workspace files! run `streamlit run app.py` to start it.*
