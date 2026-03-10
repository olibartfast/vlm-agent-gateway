"""
Parsing utilities for ReAct steps and monitoring output.
"""

import json
import re


def _parse_react_step(text: str) -> tuple[str | None, str | None, dict, str | None]:
    """
    Parse one ReAct step from model output.

    Returns:
        (thought, action, action_input_dict, final_answer)
        where final_answer is non-None only when the agent is done.
    """
    final_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return None, None, {}, final_match.group(1).strip()

    thought_match = re.search(
        r"Thought\s*:\s*(.+?)(?=Action\s*:|$)", text, re.DOTALL | re.IGNORECASE
    )
    action_match = re.search(r"Action\s*:\s*(\w+)", text, re.IGNORECASE)
    input_match = re.search(
        r"Action Input\s*:\s*(\{.*?\}|\S.*?)(?=\nObservation|\nThought|\nAction|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""

    try:
        action_input = json.loads(input_match.group(1).strip()) if input_match else {}
    except (json.JSONDecodeError, AttributeError):
        action_input = {}

    return thought, action, action_input, None


def parse_monitor_output(text: str) -> dict:
    """
    Parse the structured output from the monitoring agent.

    Returns a dict with keys: alert, summary, confidence, recommended_action, thought
    """
    result = {
        "alert": False,
        "summary": "",
        "confidence": "LOW",
        "recommended_action": "",
        "thought": "",
    }

    thought_m = re.search(
        r"Thought:\s*(.+?)(?=Alert:|$)", text, re.DOTALL | re.IGNORECASE
    )
    if thought_m:
        result["thought"] = thought_m.group(1).strip()

    alert_m = re.search(r"Alert:\s*(YES|NO)", text, re.IGNORECASE)
    if alert_m:
        result["alert"] = alert_m.group(1).strip().upper() == "YES"

    summary_m = re.search(
        r"Summary:\s*(.+?)(?=Confidence:|Alert:|$)", text, re.DOTALL | re.IGNORECASE
    )
    if summary_m:
        result["summary"] = summary_m.group(1).strip()

    conf_m = re.search(r"Confidence:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    if conf_m:
        result["confidence"] = conf_m.group(1).strip().upper()

    action_m = re.search(
        r"Recommended Action:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE
    )
    if action_m:
        result["recommended_action"] = action_m.group(1).strip()

    return result
