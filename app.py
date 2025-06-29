import os
import json
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in your environment or a .env file.")
    st.stop()

# Instantiate the OpenAI client (uses env var by default)
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Agent definitions
# -----------------------------------------------------------------------------
AGENT_SYSTEM_PROMPTS = {
    "scope": (
        "You are the SCOPE AGENT. Your task is to check whether a proposed Bachelor\'s thesis topic in "
        "Software Engineering is appropriate in scope. Specifically, verify:\n"
        "1. The topic demonstrates the student\'s ability to work scientifically.\n"
        "2. The project is feasible within a standard 3-month timeframe.\n"
        "3. It does not require groundbreaking or original research beyond Bachelor level.\n"
        "Follow the ReAct (Reasoning + Acting) framework to think step-by-step, list any concerns, and conclude with a succinct recommendation (\"ACCEPT\", \"REFINE\", or \"REJECT\")."
    ),
    "critic": (
        "You are the CRITIC AGENT. Analyse the topic for vagueness, over-complexity, or unclear goals. "
        "Suggest concrete refinements to improve clarity and focus. Use the ReAct framework and finish with"
        " your final recommendation (\"OK\", \"NEEDS_REFINEMENT\")."
    ),
    "literature": (
        "You are the LITERATURE AGENT. Estimate the availability of high-quality academic literature that could support the proposed thesis topic. "
        "Name example seminal works or venues if possible. Apply ReAct thinking and end with a rating (\"AMPLE\", \"LIMITED\", \"SCARCE\")."
    ),
    "feasibility": (
        "You are the FEASIBILITY AGENT. Evaluate whether the project can be implemented using standard software engineering tools by a student with an intermediate skill level within 3 months. "
        "Apply the ReAct framework and end with a verdict (\"FEASIBLE\", \"CHALLENGING\", \"NOT_FEASIBLE\")."
    ),
    "judge": (
        "You are the JUDGE AGENT. You will receive the individual assessments from other agents (scope, critic, literature, feasibility). "
        "Synthesise their feedback and provide a final overall decision. The decision must be one of: \"APPROVE\", \"RECOMMEND_REFINEMENT\", or \"REJECT\". "
        "Justify briefly (≤120 words) referencing key points from prior agents."
    ),
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def call_openai(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Wrapper around OpenAI chat completions"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def run_agent(agent_key: str, topic: str) -> Dict[str, str]:
    """Execute an agent with a given topic and return its parsed result."""
    system_prompt = AGENT_SYSTEM_PROMPTS[agent_key]
    user_prompt = (
        "Evaluate the following thesis topic:\n" + topic + "\n\n"
        "Respond in JSON with keys: 'thought_process', 'final_answer'. "
        "'thought_process' should contain your ReAct chain of thought. "
        "'final_answer' should contain your concluding recommendation or verdict."
    )

    raw_output = call_openai(system_prompt, user_prompt)

    # Attempt to parse JSON. Fallback to treating entire output as one string.
    try:
        parsed = json.loads(raw_output)
        thought = parsed.get("thought_process", raw_output)
        final = parsed.get("final_answer", raw_output)
    except json.JSONDecodeError:
        thought = raw_output
        final = raw_output

    return {"thought_process": thought, "final_answer": final, "raw": raw_output}

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Thesis Topic Evaluator", layout="centered")

st.title("🎓 Thesis Topic Evaluator")

st.markdown(
    """
Enter a proposed Bachelor's thesis topic in Software Engineering. 
The system uses specialised agents (ReAct) to evaluate **scope**, **clarity**, **literature support**, and **practical feasibility**, and then delivers an overall verdict.
    """
)

topic = st.text_area("Proposed Thesis Topic", height=150, placeholder="e.g. Employing Machine Learning for Automated Code Review in Continuous Integration Pipelines")

if st.button("Evaluate"):
    if not topic.strip():
        st.warning("Please input a thesis topic.")
        st.stop()

    with st.spinner("Running multi-agent evaluation…"):
        results = {}
        for key in ["scope", "critic", "literature", "feasibility"]:
            results[key] = run_agent(key, topic)

        # Prepare a combined summary for the judge agent
        judge_context_lines = []
        for k, v in results.items():
            judge_context_lines.append(f"{k.upper()} AGENT SAID:\n{v['raw']}")
        judge_input = "\n\n".join(judge_context_lines)

        judge_user_prompt = (
            "You are given the following agent outputs. Provide your synthesised verdict as instructed.\n\n" + judge_input + "\n"
            "Respond in JSON with keys: 'decision' and 'justification'."
        )
        judge_raw = call_openai(AGENT_SYSTEM_PROMPTS["judge"], judge_user_prompt, temperature=0.2)
        try:
            judge_parsed = json.loads(judge_raw)
        except json.JSONDecodeError:
            judge_parsed = {"decision": "UNKNOWN", "justification": judge_raw}

    # Display results in expandable sections
    st.success("Evaluation complete.")

    for label, key in [
        ("Scope Assessment", "scope"),
        ("Clarity Critique", "critic"),
        ("Literature Availability", "literature"),
        ("Practical Feasibility", "feasibility"),
    ]:
        with st.expander(label, expanded=False):
            st.subheader("Reasoning")
            st.markdown(results[key]["thought_process"])
            st.subheader("Conclusion")
            st.markdown(results[key]["final_answer"])

    st.markdown("---")
    st.header("🏁 Overall Verdict")
    st.subheader(judge_parsed.get("decision", "UNKNOWN"))
    st.markdown(judge_parsed.get("justification", ""))

else:
    st.info("Enter a topic and press **Evaluate** to begin.") 