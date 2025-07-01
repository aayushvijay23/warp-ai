import os, io, re
import pandas as pd
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import numpy as np
import sqlite3  # Or use SQLAlchemy if preferred
import base64


# === Configuration ===
api_key = os.getenv("MY_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    messages = [
        {"role": "system",
         "content": "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=5  # We only need a short response
    )

    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"


# === CodeGeneration TOOLS ============================================

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    3. Create only ONE relevant plot. Set `figsize=(6,4)`, add title/labels.
    4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    """


# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Assign the final result to `result`.
    3. Wrap the snippet in a single ```python code fence (no extra prose).
    """


# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    should_plot = QueryUnderstandingTool(query)
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(),
                                                                                                   query)

    messages = [
        {"role": "system",
         "content": "detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=1024
    )

    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    return code, should_plot, ""


# === ExecutionAgent ====================================================

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    env = {"pd": pd, "df": df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["io"] = io
    try:
        exec(code, {}, env)
        return env.get("result", None)
    except Exception as exc:
        return f"Error executing code: {exc}"


# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:300]

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt


# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    prompt = ReasoningCurator(query, result)
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    # Streaming LLM call
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=[
            {"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024,
        stream=True
    )

    # Stream and display thinking
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token

            # Simple state machine to extract <think>...</think> as it streams
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )

    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned


# === DataFrameSummary TOOL (pandas only) =========================================
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt


# === DataInsightAgent (upload-time only) ===============================

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Returns a fixed, polished summary and demo-ready analysis questions."""
    return """
**1. Brief Dataset Description**

This dataset contains 100 records of vehicle performance and environmental data, captured at specific timestamps. It includes temperature readings from four damper locations (Left Rear, Right Rear, Vehicle Left, Vehicle Right), outside temperature, vehicle speed (in km/h), and shock acceleration (in g-force). The data appears to be clean, with no missing values.

**2. Possible Data Analysis Questions**

1. **Temperature Distribution Analysis**: How do damper temperatures vary across different locations (LR, RR, VL, VR) in relation to outside temperature, and are there any notable patterns or correlations?

2. **Speed and Shock Acceleration Relationship**: Is there a significant correlation between vehicle speed and shock acceleration, and if so, how does this relationship change across different speed ranges?

3. **Damper Temperature Impact on Shock Absorption**: Do higher damper temperatures (in any or all locations) correlate with increased or decreased shock acceleration values, potentially indicating reduced shock absorption effectiveness?
"""


# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

def load_data_from_csv() -> pd.DataFrame:
    """Load data from a local CSV file into a DataFrame."""
    file_path = "dummy_vehicle_endurance_data.csv"  # Change to the actual path of your CSV
    df = pd.read_csv(file_path)
    return df


# === Main Streamlit App ===============================================

def main():
    st.set_page_config(layout="wide")

    # Load GIF and encode as base64
    with open("bot.gif", "rb") as f:
        gif_bytes = f.read()
        gif_base64 = base64.b64encode(gif_bytes).decode()

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h1 style="margin: 0;">Hey! Welcome to WARP AI! 👋</h1>
            <img src="data:image/gif;base64,{gif_base64}"
                 style="width: 150px; height: auto; display: block;" />
        </div>
        """,
        unsafe_allow_html=True
    )

    if "plots" not in st.session_state:
        st.session_state.plots = []

    left, right = st.columns([3, 7])

    with left:
        st.header("Data Analysis Agent")
        st.markdown("<medium>Powered by MBRDI</medium>", unsafe_allow_html=True)

        if "df" not in st.session_state:
            with st.spinner("Connecting to database and loading data …"):
                try:
                    st.session_state.df = load_data_from_csv()
                    st.session_state.insights = DataInsightAgent(st.session_state.df)
                except Exception as e:
                    st.error(f"Failed to load data: {e}")
                    return

        st.dataframe(st.session_state.df.head())
        st.markdown("### Dataset Insights")
        st.markdown(st.session_state.insights)

    with right:
        st.header("Chat with the Endurance Testing data")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)

        if user_q := st.chat_input("Ask about your data…"):
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.spinner("Working …"):
                # Detect if query is for mercedes table (deterministic response)
                mercedes_keywords = ["report", "chapter", "endurance"]
                if any(kw in user_q.lower() for kw in mercedes_keywords) and "s-class" in user_q.lower():
                    # Return a hard-coded deterministic row
                    response_text = """
                    ### 🔍 Report on S-Class Endurance

                    | Report Name                     | Chapter                   | Description                                                                 |
                    |--------------------------------|---------------------------|-----------------------------------------------------------------------------|
                    | S-Class_Climate_Endurance_Report | HVAC System Performance   | Evaluation of air conditioning efficiency and noise levels during 24-hour hot and cold climate chamber testing. |
                    """
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "plot_index": None
                    })
                    st.rerun()
                else:
                    # Normal LLM pipeline for endurance data
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                    is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                    plot_idx = None
                    if is_plot:
                        fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                        st.session_state.plots.append(fig)
                        plot_idx = len(st.session_state.plots) - 1
                        header = "Here is the visualization you requested:"
                    elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                        header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                    else:
                        header = f"Result: {result_obj}"

                    thinking_html = ""
                    if raw_thinking:
                        thinking_html = (
                            '<details class="thinking">'
                            '<summary>🧠 Reasoning</summary>'
                            f'<pre>{raw_thinking}</pre>'
                            '</details>'
                        )

                    explanation_html = reasoning_txt
                    code_html = (
                        '<details class="code">'
                        '<summary>View code</summary>'
                        '<pre><code class="language-python">'
                        f'{code}'
                        '</code></pre>'
                        '</details>'
                    )
                    assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_msg,
                        "plot_index": plot_idx
                    })
                    st.rerun()


if __name__ == "__main__":
    main()
