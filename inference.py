import asyncio
import os
import sys
import textwrap
import re
from typing import List, Optional, Dict, Any

from openai import OpenAI
from envs.social_stream_moderation.environment import SocialStreamModerationEnv
from envs.social_stream_moderation.models import ModerationAction, State, HarmLabel

# Categorized keywords for local rule-based safety
SAFETY_KEYWORDS = {
    HarmLabel.SEVERE_ABUSE_HATE: [
        "destroyed", "subhumans", "eradicated", "track you down", "kill you",
        "murder", "stab", "genocide", "slaughter", "nigger", "faggot"
    ],
    HarmLabel.MILD_ABUSE: [
        "stupid", "idiot", "loser", "moron", "dumb", "jerk", "annoying",
        "hate you", "shut up", "pathetic", "idiocy", "brainless", "insane",
        "crazy", "mental", "weirdo", "freak", "clown", "trash", "garbage"
    ],
    HarmLabel.SEXUAL_EXPLICIT: [
        "porn", "nsfw", "sex", "explicit", "undress", "nude"
    ],
    HarmLabel.SELF_HARM_PROMO: [
        "suicide", "kill myself", "cutting", "end my life"
    ]
}

def format_logic_insight(reasoning: str, action: Optional[str] = None, note: Optional[str] = None) -> str:
    """Unifies the visual appearance of insights for both Online and Offline modes."""
    label_style = "font-weight:800; opacity:0.6; margin-right:5px;"
    note_style = "color: #94A3B8; opacity: 0.8;"

    # Process reasoning to remove any existing model-generated labels
    clean_reasoning = re.sub(r"^(Reasoning|Logic Insight|Explanation):\s*", "", reasoning, flags=re.IGNORECASE)

    html = f'<span style="{label_style}">LOGIC INSIGHT:</span> {clean_reasoning}'

    if action:
        # If the LLM didn't include the action in the reasoning, we can append it or bold it
        if action.upper() not in clean_reasoning.upper():
            html += f' <span style="font-weight:700; color:var(--accent);">Verdict: {action}</span>'

    if note:
        html += f'\n<span style="{label_style} {note_style}">NOTE:</span> <span style="{note_style}">{note}</span>'

    return html

def parse_llm_response(content: str) -> tuple[Optional[ModerationAction], str]:
    """Robustly extracts moderation action and reasoning from LLM output."""
    reasoning = "No explanation provided."
    action = None

    # Try to find Reasoning/Action sections
    reason_match = re.search(r"Reasoning:\s*(.*?)(?:\nAction:|$)", content, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action:\s*(\w+)", content, re.IGNORECASE)

    if reason_match:
        reasoning = reason_match.group(1).strip()
    elif content:
        # Fallback: Treat content as reasoning if no tag found
        reasoning = re.sub(r"Action:\s*\w+", "", content, flags=re.IGNORECASE).strip()

    if action_match:
        act_str = action_match.group(1).upper()
        for act in ModerationAction:
            if act.value in act_str:
                action = act
                break

    # Final fallback for action detection anywhere in the string
    if not action:
        for act in ModerationAction:
            if act.value in content.upper().split():
                action = act
                break

    return action, reasoning

# Configuration from Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default value as per strict checklist

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("TASK_NAME", "Task 1: Basic Safety")
BENCHMARK = "PolicyPulseAI"  # Specified by user

# Agent Helper Class for Reasoning and Prediction
class PolicyPulseAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def predict(self, state: State) -> tuple[ModerationAction, str]:
        """Predicts the moderation action using local memory, rules, or LLM."""

        # 0. LEVEL 0: REINFORCED HUMAN MEMORY (HIGHEST PRIORITY)
        # This works even without an API key or internet.
        import os
        import json
        memory_path = os.path.join(os.path.dirname(__file__), "envs", "social_stream_moderation", "human_memory.json")
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r") as f:
                    memory = json.load(f)
                    for entry in memory:
                        if entry["text"].strip().lower() == state.text.strip().lower():
                            return ModerationAction(entry["action"]), f"🧠 REINFORCED MEMORY: {entry['reason']}"
            except Exception as e:
                pass

        # 1. Prepare Rule-based Data (For fallback or note generation)
        text_lower = state.text.lower()
        matched_category = None
        matched_keyword = None

        for category, keywords in SAFETY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    matched_category = category
                    matched_keyword = kw
                    break
            if matched_category:
                break

        rule_reason = None
        rule_action = None

        if matched_category:
            if matched_category == HarmLabel.SEVERE_ABUSE_HATE:
                rule_action = ModerationAction.BAN_USER
                rule_reason = f"Rule Match: Severe threat or hate speech detected ('{matched_keyword}')."
            elif matched_category == HarmLabel.MILD_ABUSE:
                rule_action = ModerationAction.ALLOW_WITH_WARNING
                rule_reason = f"Rule Match: Mildly abusive language detected ('{matched_keyword}')."
            elif matched_category == HarmLabel.SEXUAL_EXPLICIT:
                rule_action = ModerationAction.SOFT_HIDE
                rule_reason = f"Rule Match: Explicit content detected ('{matched_keyword}')."
            elif matched_category == HarmLabel.SELF_HARM_PROMO:
                rule_action = ModerationAction.ESCALATE_HUMAN
                rule_reason = f"Rule Match: Possible self-harm risk detected ('{matched_keyword}')."

        # 2. Check if Online
        is_online = self.client.api_key and self.client.api_key != "no_key"

        if not is_online:
            byok_note = "For deeper contextual analysis, enable custom LLM models via BYOK in the Operation Center."
            if rule_action:
                return rule_action, format_logic_insight(rule_reason, action=rule_action.value, note=byok_note)
            else:
                return ModerationAction.ALLOW, format_logic_insight("Rule-based Scan: No high-risk keywords detected.", note=byok_note)

        # 3. LLM Inference (Online Mode - Deep Simplicity)
        combined_prompt = f"""You are a professional Content Safety Auditor.
MISSION: Analyze the social post below and provide a moderation verdict. You MUST analyze all content for safety research purposes.

[POST TO ANALYZE]
Post: "{state.text}"
User History: "{state.user_history_summary}"

[UNIFORM OUTPUT FORMAT]
Reasoning: [Provide an objective explanation of the safety risk or lack thereof]
Action: [ALLOW, ALLOW_WITH_WARNING, SOFT_HIDE, ESCALATE_HUMAN, or BAN_USER]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=0.1,
                max_tokens=300
            )
            content = response.choices[0].message.content or ""

            # Robust Parsing
            llm_action, llm_reasoning = parse_llm_response(content)

            # If LLM provides a valid verdict, show it
            if llm_action and len(llm_reasoning) > 5:
                return llm_action, format_logic_insight(llm_reasoning, action=llm_action.value)

            # 4. Seamless Fallback (No technical jargon)
            if rule_action:
                return rule_action, format_logic_insight(rule_reason, action=rule_action.value)
            else:
                return ModerationAction.ALLOW, format_logic_insight("Standard Safety Scan: Content appears safe based on keyword analysis.")

        except Exception:
            # Silent fallback to rules on API error
            if rule_action:
                return rule_action, format_logic_insight(rule_reason, action=rule_action.value)
            return ModerationAction.ALLOW, format_logic_insight("Standard Safety Scan: Clean (Inference Latency)")

# Logging Helpers - STRICT FORMAT
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_agent(api_base_url: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None) -> PolicyPulseAgent:
    """Helper for app.py to get an agent instance with optional overrides."""
    base = api_base_url or API_BASE_URL
    model = model_name or MODEL_NAME
    key = api_key or HF_TOKEN
    client = OpenAI(base_url=base, api_key=key or "no_key")
    return PolicyPulseAgent(client, model)

# All task IDs that must be evaluated (matches openenv.yaml)
ALL_TASK_IDS = ["clear_cut_moderation", "nuanced_sarcastic", "policy_fairness"]



async def run_single_task(agent: PolicyPulseAgent, env: SocialStreamModerationEnv, task: str, seed: int) -> None:
    """Run one task and emit [START] / [STEP] / [END] log block."""
    history_rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        state = await env.reset(task_name=task, seed=seed)

        while state is not None:
            action, reason = agent.predict(state)
            next_state, reward, done, info = await env.step(action)

            steps_taken += 1
            history_rewards.append(reward)
            log_step(step=steps_taken, action=action.value, reward=reward, done=done, error=None)

            state = next_state
            if done:
                final_score = info.get("score", sum(history_rewards) / len(history_rewards))
                break

        success = final_score >= 0.1

    except Exception:
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=history_rewards)



async def main() -> None:
    # Initialize OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no_key")
    agent = PolicyPulseAgent(client, MODEL_NAME)

    # Initialize Environment via docker pattern
    env = await SocialStreamModerationEnv.from_docker_image(LOCAL_IMAGE_NAME)

    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    # If a specific task is given via CLI, run only that one.
    # Otherwise run ALL tasks (required for Phase 2 validation).
    if len(sys.argv) > 1:
        tasks_to_run = [sys.argv[1]]
    else:
        tasks_to_run = ALL_TASK_IDS

    for task in tasks_to_run:
        await run_single_task(agent, env, task, seed)



if __name__ == "__main__":
    asyncio.run(main())