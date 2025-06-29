import os
import json
import re
from typing import Dict, List, Generator, Optional

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
    "refiner": (
        "You are the TOPIC REFINER. Given a Bachelor thesis topic in Software Engineering and the feedback from evaluation agents, you must propose an improved topic that addresses all raised concerns while keeping the original domain intent. "
        "Respond ONLY with the refined topic sentence – do not include any additional commentary."
    ),
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def call_openai(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Non-streaming completion used internally when full response is fine."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def call_openai_stream(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Generator[str, None, None]:
    """Yield content chunks from the OpenAI stream."""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _extract_json(text: str) -> Optional[dict]:
    """Attempt to locate and parse the first JSON object in the text."""
    match = re.search(r"{.*}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def run_agent_stream(agent_key: str, topic: str, reasoning_placeholder: st.empty) -> Dict[str, str]:
    """Execute an agent with streaming, updating the given placeholder live."""
    system_prompt = AGENT_SYSTEM_PROMPTS[agent_key]
    user_prompt = (
        "Evaluate the following thesis topic:\n" + topic + "\n\n"
        "Respond in JSON with keys: 'thought_process', 'final_answer'. "
        "'thought_process' should contain your ReAct chain of thought. "
        "'final_answer' should contain your concluding recommendation or verdict."
    )

    collected = ""
    for token in call_openai_stream(system_prompt, user_prompt):
        collected += token
        reasoning_placeholder.markdown(collected + "▌")

    # Finished streaming – remove cursor
    reasoning_placeholder.markdown(collected)

    # Parse JSON if possible (flexible extraction)
    parsed = _extract_json(collected)
    if parsed is not None:
        thought = parsed.get("thought_process", collected)
        final = parsed.get("final_answer", collected)
    else:
        thought = collected
        final = collected

    if debug_mode:
        print(f"\n[{agent_key.upper()} RAW OUTPUT]\n{collected}\n")

    return {"thought_process": thought, "final_answer": final, "raw": collected}

# -----------------------------------------------------------------------------
# Non-stream helpers (defined early for linter; full definitions appear later)
# -----------------------------------------------------------------------------

def run_agent(agent_key: str, topic: str) -> Dict[str, str]:  # noqa: F401
    """Run agent without streaming, returning parsed content."""
    system_prompt = AGENT_SYSTEM_PROMPTS[agent_key]
    user_prompt = (
        "Evaluate the following thesis topic:\n" + topic + "\n\n"
        "Respond in JSON with keys: 'thought_process', 'final_answer'. "
        "'thought_process' should contain your ReAct chain of thought. "
        "'final_answer' should contain your concluding recommendation or verdict."
    )

    raw_output = call_openai(system_prompt, user_prompt)

    parsed = _extract_json(raw_output)
    if parsed is not None:
        thought = parsed.get("thought_process", raw_output)
        final = parsed.get("final_answer", raw_output)
    else:
        thought = raw_output
        final = raw_output

    if debug_mode:
        print(f"\n[{agent_key.upper()} RAW OUTPUT]\n{raw_output}\n")

    return {"thought_process": thought, "final_answer": final, "raw": raw_output}


def evaluate_topic(topic: str) -> Dict[str, Dict[str, str]]:  # noqa: F401
    """Evaluate the topic with all specialised agents and judge."""
    results: Dict[str, Dict[str, str]] = {}
    for key in ["scope", "critic", "literature", "feasibility"]:
        results[key] = run_agent(key, topic)

    judge_context_lines = [f"{k.upper()} AGENT SAID:\n{v['raw']}" for k, v in results.items()]
    judge_input = "\n\n".join(judge_context_lines)

    judge_prompt = (
        "You are given the following agent outputs. Provide your synthesised verdict as instructed.\n\n" + judge_input + "\n"
        "Respond in JSON with keys: 'decision' and 'justification'."
    )
    judge_raw = call_openai(AGENT_SYSTEM_PROMPTS["judge"], judge_prompt, temperature=0.2)

    judge_parsed_obj = _extract_json(judge_raw) or {}
    judge_parsed = {
        "decision": judge_parsed_obj.get("decision", "UNKNOWN"),
        "justification": judge_parsed_obj.get("justification", judge_raw),
        "raw": judge_raw,
    }

    if debug_mode:
        print("\n[JUDGE RAW OUTPUT]\n" + judge_raw + "\n")

    results["judge"] = judge_parsed
    return results


def refine_topic(original_topic: str, evaluation_results: Dict[str, Dict[str, str]]) -> str:  # noqa: F401
    """Ask the refiner agent for a better topic."""
    feedback_lines = []
    for k in ["scope", "critic", "literature", "feasibility", "judge"]:
        raw = evaluation_results[k]["raw"]
        feedback_lines.append(f"{k.upper()} FEEDBACK:\n{raw}")

    refiner_prompt_user = (
        "Original topic:\n" + original_topic + "\n\n" + "\n\n".join(feedback_lines) + "\n\n"
        "Provide ONLY one improved topic that addresses all concerns."
    )

    refined_topic = call_openai(AGENT_SYSTEM_PROMPTS["refiner"], refiner_prompt_user, temperature=0.4)

    refined_topic = refined_topic.strip().split("\n")[0]

    if debug_mode:
        print("\n[REFINER OUTPUT]\n" + refined_topic + "\n")

    return refined_topic

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Thesis Topic Evaluator", layout="centered")

# Add debug toggle in sidebar
debug_mode = st.sidebar.checkbox("🔧 Debug mode (print raw outputs)")

st.title("🎓 Thesis Topic Evaluator")

st.markdown(
    """
Enter a proposed Bachelor's thesis topic in Software Engineering. 
The system uses specialised agents (ReAct) to evaluate **scope**, **clarity**, **literature support**, and **practical feasibility**, and then delivers an overall verdict.
    """
)

# Input controls ----------------------------------------------------------------

topic = st.text_area("Proposed Thesis Topic", height=150, placeholder="e.g. Employing Machine Learning for Automated Code Review in Continuous Integration Pipelines")

auto_iterate = st.checkbox("🔄 Automatically refine until approved (max 3 iterations)")

if st.button("Evaluate & Iterate" if auto_iterate else "Evaluate"):
    if not topic.strip():
        st.warning("Please input a thesis topic.")
        st.stop()

    with st.spinner("Running evaluation…"):
        results = {}

        # ----------- FIRST ITERATION (with streaming) -------------------------
        st.subheader("Iteration 1: Original Topic")
        st.markdown(f"**Topic:** _{topic}_")

        # Pre-create expanders and placeholders so we can update them while streaming
        agent_placeholders: Dict[str, Dict[str, st.empty]] = {}
        for label, key in [
            ("Scope Assessment", "scope"),
            ("Clarity Critique", "critic"),
            ("Literature Availability", "literature"),
            ("Practical Feasibility", "feasibility"),
        ]:
            with st.expander(label, expanded=True):
                reason_ph = st.empty()
                conclusion_ph = st.empty()
            agent_placeholders[key] = {"reason": reason_ph, "conclusion": conclusion_ph}

        # Run each agent streamingly
        for key in ["scope", "critic", "literature", "feasibility"]:
            phs = agent_placeholders[key]
            results[key] = run_agent_stream(key, topic, phs["reason"])
            # After completion, fill conclusion
            phs["reason"].markdown(results[key]["thought_process"])
            phs["conclusion"].markdown(f"**Conclusion:** {results[key]['final_answer']}")

        # Judge evaluation for first iteration (reuse existing logic)
        judge_context_lines = [f"{k.upper()} AGENT SAID:\n{v['raw']}" for k, v in results.items()]
        judge_input = "\n\n".join(judge_context_lines)

        judge_user_prompt = (
            "You are given the following agent outputs. Provide your synthesised verdict as instructed.\n\n" + judge_input + "\n"
            "Respond in JSON with keys: 'decision' and 'justification'."
        )
        judge_raw = call_openai(AGENT_SYSTEM_PROMPTS["judge"], judge_user_prompt, temperature=0.2)

        judge_parsed_obj = _extract_json(judge_raw) or {}
        judge_parsed = {
            "decision": judge_parsed_obj.get("decision", "UNKNOWN"),
            "justification": judge_parsed_obj.get("justification", judge_raw),
            "raw": judge_raw,
        }

        if debug_mode:
            print("\n[JUDGE RAW OUTPUT]\n" + judge_raw + "\n")

    # The expanders are already populated live; no need to re-render here

    st.success("Iteration 1 complete.")

    current_topic = topic
    current_results = {**results, "judge": judge_parsed}

    # ------------------- Auto-Refinement Loop ----------------------
    if auto_iterate:
        max_iters = 3
        iter_num = 1

        while iter_num < max_iters and current_results["judge"]["decision"] != "APPROVE":
            iter_num += 1
            st.markdown("---")
            st.subheader(f"Iteration {iter_num}: Refinement")

            # Refine topic
            current_topic = refine_topic(current_topic, current_results)
            st.markdown(f"**Proposed topic:** _{current_topic}_")

            # ----------- STREAMING EVALUATION FOR REFINED TOPIC -------------
            agent_placeholders_ref: Dict[str, Dict[str, st.empty]] = {}
            for label, key in [
                ("Scope Assessment", "scope"),
                ("Clarity Critique", "critic"),
                ("Literature Availability", "literature"),
                ("Practical Feasibility", "feasibility"),
            ]:
                with st.expander(label, expanded=True):
                    reason_ph = st.empty()
                    conclusion_ph = st.empty()
                agent_placeholders_ref[key] = {"reason": reason_ph, "conclusion": conclusion_ph}

            # Run agents with streaming
            current_results = {}
            for key in ["scope", "critic", "literature", "feasibility"]:
                phs = agent_placeholders_ref[key]
                current_results[key] = run_agent_stream(key, current_topic, phs["reason"])
                phs["reason"].markdown(current_results[key]["thought_process"])
                phs["conclusion"].markdown(f"**Conclusion:** {current_results[key]['final_answer']}")

            # Judge evaluation
            judge_ctx_lines = [f"{k.upper()} AGENT SAID:\n{v['raw']}" for k, v in current_results.items()]
            judge_input2 = "\n\n".join(judge_ctx_lines)
            judge_user_prompt2 = (
                "You are given the following agent outputs. Provide your synthesised verdict as instructed.\n\n" + judge_input2 + "\n"
                "Respond in JSON with keys: 'decision' and 'justification'."
            )
            judge_raw2 = call_openai(AGENT_SYSTEM_PROMPTS["judge"], judge_user_prompt2, temperature=0.2)
            judge_obj2 = _extract_json(judge_raw2) or {}
            current_results["judge"] = {
                "decision": judge_obj2.get("decision", "UNKNOWN"),
                "justification": judge_obj2.get("justification", judge_raw2),
                "raw": judge_raw2,
            }

            st.markdown(f"### Verdict: {current_results['judge']['decision']}")
            st.markdown(current_results["judge"]["justification"])

        st.markdown("---")
        st.header("🏁 Final Verdict")
        st.markdown(f"**Final topic:** _{current_topic}_")
        st.subheader(current_results["judge"]["decision"])
        st.markdown(current_results["judge"]["justification"])

    else:
        st.markdown("---")
        st.header("🏁 Overall Verdict")
        st.markdown(f"**Topic:** _{topic}_")
        st.subheader(judge_parsed.get("decision", "UNKNOWN"))
        st.markdown(judge_parsed.get("justification", ""))

else:
    st.info("Enter a topic and press **Evaluate** to begin.") 