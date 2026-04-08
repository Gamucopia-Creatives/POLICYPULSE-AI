import os
import sys
import json
import random
from typing import Optional, List, Dict, Any
from envs.social_stream_moderation.environment import SocialStreamModerationEnv
from envs.social_stream_moderation.models import ModerationAction, State

# Simple Rule-based Fallback Agent
class RuleBasedAgent:
    def predict(self, state: State) -> tuple[ModerationAction, str]:
        text = state.text.lower()
        history = state.user_history_summary
        
        # Obvious severe markers
        severe_keywords = ["destroyed", "subhumans", "eradicated", "track you down", "hurt your family", "kill you", "find and hit", "murder", "search and destroy", "stab", "death at you", "find you"]
        sexual_keywords = ["explicit", "sexual", "hardcore", "adult videos", "porn"]
        self_harm_keywords = ["can't take it anymore", "end it all", "cut yourself", "suicide"]
        
        for kw in severe_keywords:
            if kw in text:
                return ModerationAction.BAN_USER, f"Rule Match: Severe Violence/Threat detected ('{kw}')"
        
        for kw in sexual_keywords:
            if kw in text:
                return ModerationAction.BAN_USER, f"Rule Match: Explicit Sexual Content detected ('{kw}')"
                
        for kw in self_harm_keywords:
            if kw in text:
                return ModerationAction.ESCALATE_HUMAN, f"Rule Match: Self-harm risk detected ('{kw}') - Sending for Human Review"
                
        # Mild markers
        mild_keywords = ["idiot", "shut the hell up", "stupid", "dumbest", "moron", "useless"]
        for kw in mild_keywords:
            if kw in text:
                if history == "no_prior_violations":
                    return ModerationAction.ALLOW_WITH_WARNING, f"Rule Match: Mild insult ('{kw}'). Account in good standing - Warning issued."
                else:
                    return ModerationAction.SOFT_HIDE, f"Rule Match: Repeat mild offense ('{kw}'). User history shows prior violations - Applying Soft Hide."
                    
        return ModerationAction.ALLOW, "Rule Match: No policy violations detected. Content is safe."


class LLMAgent:
    def __init__(self, api_base_url: str, model_name: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name
        
    def predict(self, state: State) -> tuple[ModerationAction, str]:
        prompt = f"""You are an AI content moderator.
Analyze this post: "{state.text}"
User History: "{state.user_history_summary}"

Policies:
- ALLOW: Neutral content
- ALLOW_WITH_WARNING: Mild insults
- SOFT_HIDE: Borderline
- ESCALATE_HUMAN: Hate, self-harm
- BAN_USER: Severe violations

Write your thought process and then state the action.

Example:
Reasoning: This contains a violent threat.
Action: BAN_USER

Provide your response below:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Robust check for safety blocks / empty responses
            choices = getattr(response, "choices", [])
            if not choices or len(choices) == 0:
                return ModerationAction.ALLOW, "LLM Error: Request was rejected by the model provider (likely a safety filter block)."
                
            content = choices[0].message.content
            if content is None:
                return ModerationAction.ALLOW, "LLM Error: Received empty response from model API (Safety violation block or token truncation)."
                
            # Escape HTML characters so they don't break the web UI
            llm_output = content.strip().replace('<', '&lt;').replace('>', '&gt;')
            
            # Map response to enum robustly, catching common truncations
            action_mapping = {
                "ALLOW_WITH_WARNING": ModerationAction.ALLOW_WITH_WARNING,
                "ESCALATE_HUMAN": ModerationAction.ESCALATE_HUMAN,
                "BAN_USER": ModerationAction.BAN_USER,
                "SOFT_HIDE": ModerationAction.SOFT_HIDE,
                "WARNING": ModerationAction.ALLOW_WITH_WARNING,
                "ESCALATE": ModerationAction.ESCALATE_HUMAN,
                "BAN": ModerationAction.BAN_USER,
                "HIDE": ModerationAction.SOFT_HIDE,
                "ALLOW": ModerationAction.ALLOW
            }
            
            # The LLM's raw output is passed directly to the UI as the insight
            for key, action in action_mapping.items():
                if f"Action: {key}" in llm_output or f"Action: {key}".upper() in llm_output.upper():
                    return action, llm_output
            
            # Fallback if "Action: X" format wasn't strictly followed but the word is just in the text
            action_str = llm_output.upper()
            for key, action in action_mapping.items():
                if key in action_str:
                    return action, llm_output
                    
            return ModerationAction.ALLOW, llm_output
        except Exception as e:
            return ModerationAction.ALLOW, f"LLM Error: {str(e)}"

def get_agent(api_base_url: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None):
    # Check for LLM config. Fallback to passed arguments, then os.environ
    base_url = api_base_url or os.environ.get("API_BASE_URL")
    model = model_name or os.environ.get("MODEL_NAME")
    token = api_key or os.environ.get("HF_TOKEN", "fake_key")
    
    if base_url and model:
        return LLMAgent(base_url, model, token)
    else:
        return RuleBasedAgent()

def run_episode(task_name: str, seed: Optional[int] = None):
    env = SocialStreamModerationEnv()
    
    agent = get_agent()
        
    state = env.reset(task_name=task_name, seed=seed)
    
    # [START] marker - Strictly formatted
    print(f"[START] task={task_name} | seed={seed}")
    
    step_idx = 0
    done = False
    
    while not done:
        action, reason = agent.predict(state)
        # Store state ID before step
        state_id = state.post_id
        next_state, reward, done, info = env.step(action)
        
        # [STEP] marker - Strictly formatted
        print(f"[STEP] step={step_idx} | state={state_id} | action={action.value} | reward={reward:.4f} | reason={reason}")
        
        state = next_state
        step_idx += 1
        
    # [END] marker - Strictly formatted
    final_score = info.get("final_episode_score", 0.0)
    print(f"[END] score={final_score:.4f}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "clear_cut_moderation"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_episode(task, seed)
