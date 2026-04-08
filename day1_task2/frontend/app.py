import streamlit as st
import requests
import json

# Backend API URL (adjust if running on different port)
API_URL = "http://localhost:8000"

st.title("Math Chain-of-Thought Assistant")
st.markdown("Enter a multi-step math problem and get step-by-step reasoning with the final answer.")

# Input section
problem = st.text_area(
    "Math Problem:",
    placeholder="e.g., If a train travels 120 km in 2 hours, what is its average speed?",
    height=100
)

if st.button("Solve Problem", type="primary"):
    if not problem.strip():
        st.error("Please enter a math problem.")
    else:
        with st.spinner("Solving..."):
            try:
                # Send request to backend
                response = requests.post(
                    f"{API_URL}/solve",
                    json={"problem": problem},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display steps
                    st.subheader("Step-by-Step Reasoning:")
                    for step in result["steps"]:
                        st.markdown(f"**Step {step['step_number']}:** {step['reasoning']}")
                        if step.get("computation"):
                            st.code(f"Computation: {step['computation']}")
                        if step.get("result"):
                            st.code(f"Result: {step['result']}")
                    
                    # Display final answer
                    st.success(f"**Final Answer:** {result['final_answer']}")
                    
                else:
                    st.error(f"Error: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {str(e)}")
                st.info("Make sure the FastAPI backend is running on http://localhost:8000")

# Footer
st.markdown("---")
st.markdown("*Powered by Groq API and Chain-of-Thought reasoning*")