import streamlit as st
import pandas as pd
import time
import os
import uuid
import plotly.express as px
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Load environment variables from .env if present
load_dotenv()

# Define the State for LangGraph
class PipelineState(TypedDict):
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    report: dict

# -----------------------------------------
# LangGraph Nodes
# -----------------------------------------
def extract_node(state: PipelineState) -> PipelineState:
    """Extract node is generally for loading data from source. 
    Here it just initializes the process since Streamlit loads the file buffer."""
    start_time = time.time()
    raw_df = state['raw_df']
    
    state['report'] = {
        'initial_rows': len(raw_df),
        'start_time': start_time,
        'audit_log': []
    }
    
    state['report']['audit_log'].append({
        'node': 'Extract',
        'event': 'Loaded Raw Data',
        'rows_extracted': len(raw_df),
        'duration_seconds': round(time.time() - start_time, 4)
    })
    
    return state

def transform_node(state: PipelineState) -> PipelineState:
    """Cleans the dataframe by removing duplicates, fixing dates, and handling NAs."""
    start_time = time.time()
    df = state['raw_df'].copy()
    
    # 1. Drop duplicates
    rows_before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_dropped = rows_before_dedup - len(df)
    
    # 2. Normalize and fix dates
    # Coerce errors to NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    dates_dropped = df['date'].isna().sum()
    df.dropna(subset=['date'], inplace=True)
    
    # 3. Handle missing prices
    prices_dropped = df['price'].isna().sum()
    df.dropna(subset=['price'], inplace=True)
    
    # 4. Handle invalid quantity
    invalid_qty_dropped = (df['quantity'] <= 0).sum()
    df = df[df['quantity'] > 0]
    
    state['clean_df'] = df
    state['report'].update({
        'duplicates_dropped': int(duplicates_dropped),
        'invalid_dates_dropped': int(dates_dropped),
        'missing_prices_dropped': int(prices_dropped),
        'invalid_quantity_dropped': int(invalid_qty_dropped),
    })
    
    state['report']['audit_log'].append({
        'node': 'Transform',
        'event': 'Cleaned Data',
        'anomalies_removed': int(duplicates_dropped + dates_dropped + prices_dropped + invalid_qty_dropped),
        'duration_seconds': round(time.time() - start_time, 4)
    })
    
    return state

def load_node(state: PipelineState) -> PipelineState:
    """Finalizes the pipeline and records execution time."""
    start_time = time.time()
    report = state['report']
    clean_df = state['clean_df']
    
    report['final_rows'] = len(clean_df)
    report['end_time'] = time.time()
    
    report['audit_log'].append({
        'node': 'Load',
        'event': 'Finalized Clean Data',
        'final_rows_verified': len(clean_df),
        'duration_seconds': round(time.time() - start_time, 4)
    })
    
    report['execution_time_seconds'] = round(report['end_time'] - report['start_time'], 4)
    
    state['report'] = report
    return state

# -----------------------------------------
# Graph Construction
# -----------------------------------------
def build_graph():
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("Extract", extract_node)
    workflow.add_node("Transform", transform_node)
    workflow.add_node("Load", load_node)
    
    workflow.set_entry_point("Extract")
    workflow.add_edge("Extract", "Transform")
    workflow.add_edge("Transform", "Load")
    workflow.add_edge("Load", END)
    
    return workflow.compile()

# -----------------------------------------
# Streamlit Interface
# -----------------------------------------
st.set_page_config(page_title="Data Cleaning ETL", layout="wide")

with st.sidebar:
    st.header("Observability Settings")
    st.markdown("Integrate with **LangSmith** to trace your ETL workflow sequentially.")
    langchain_api_key = st.text_input(
        "LangChain API Key", 
        type="password",
        value=os.environ.get("LANGCHAIN_API_KEY", "")
    )
    
    if langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = "E-Commerce-ETL"
        st.success("✅ LangSmith Tracing Enabled")
        st.markdown("[Go to LangSmith Dashboard](https://smith.langchain.com/)")
    else:
        # Disable tracing if key is removed
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        st.info("Provide your API key to monitor execution traces.")

st.title("🛒 E-Commerce Data Cleaning Pipeline")
st.markdown("Upload your raw sales data (CSV) and run it through our automated LangGraph ETL pipeline.")

uploaded_file = st.file_uploader("Upload Raw CSV", type=['csv'])

if uploaded_file is not None:
    # We read the raw CSV here to get the initial dataframe
    raw_df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(5))
    
    if st.button("Run ETL Pipeline"):
        with st.spinner("Executing Pipeline via LangGraph..."):
            pipeline = build_graph()
            
            # Initialize State
            initial_state = {
                'raw_df': raw_df,
                'clean_df': pd.DataFrame(),
                'report': {}
            }
            
            # Setup LangSmith run_id explicitly for easy cross-referencing
            run_id = uuid.uuid4()
            config = RunnableConfig(run_id=run_id)
            
            # Execute Pipeline
            # Note: invoke returns the final state dict
            final_state = pipeline.invoke(initial_state, config=config)
            
            clean_df = final_state['clean_df']
            report = final_state['report']
            
        st.success("Pipeline executed successfully!")
        
        if langchain_api_key:
            st.info(f"🔎 **Observability Trace logged to LangSmith!** Run ID: `{run_id}`")
        
        # --- UI Tabs ---
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Observability Report", "🕸️ Workflow Architecture"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Cleaned Data Preview")
                st.dataframe(clean_df.head(10))
                
                st.markdown("### Cleaned Data Insights")
                if 'product_id' in clean_df.columns and 'quantity' in clean_df.columns:
                    fig_prod = px.histogram(
                        clean_df, 
                        x='product_id', 
                        y='quantity', 
                        color='product_id', 
                        title="Total Cleaned Quantity by Product",
                        text_auto=True
                    )
                    st.plotly_chart(fig_prod, use_container_width=True)
                
                # Download Button
                csv_data = clean_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv_data,
                    file_name="cleaned_sales_data.csv",
                    mime="text/csv"
                )
                
            with col2:
                st.subheader("📊 Execution Report")
                col2_1, col2_2 = st.columns(2)
                col2_1.metric("Initial Records", report['initial_rows'])
                col2_2.metric("Final Cleaned", report['final_rows'])
                
                st.markdown("### Dropped Records Breakdown")
                
                labels = ["Duplicates", "Invalid Dates", "Missing Prices", "Invalid Qty"]
                values = [
                    report.get('duplicates_dropped', 0),
                    report.get('invalid_dates_dropped', 0),
                    report.get('missing_prices_dropped', 0),
                    report.get('invalid_quantity_dropped', 0)
                ]
                
                # Interactive Plotly Donut Chart
                fig_errors = px.pie(
                    names=labels, 
                    values=values, 
                    hole=0.4, 
                    title="Distribution of Data Errors",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_errors.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_errors, use_container_width=True)
                
                st.markdown("---")
                st.write(f"⏱️ **Pipeline Execution Time:** {report['execution_time_seconds']} seconds")
                
        with tab2:
            st.subheader("📋 Pipeline Observability Report")
            st.markdown("Detailed breakdown of node execution timings and intermediate anomaly metrics.")
            
            st.json(report)
            
            import json
            report_json = json.dumps(report, indent=4)
            st.download_button(
                label="📥 Download JSON Report",
                data=report_json,
                file_name="observability_report.json",
                mime="application/json",
                key="download-json"
            )
            
        with tab3:
            st.subheader("LangGraph Workflow Visualization")
            st.markdown("Visual architecture of the LangGraph ETL pipeline logic:")
            try:
                graph_image = pipeline.get_graph().draw_mermaid_png()
                st.image(graph_image)
            except Exception as e:
                st.error(f"Could not render the workflow image. Error: {e}")
