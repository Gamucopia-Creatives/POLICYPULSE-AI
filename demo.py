import os
import json
import random
from envs.social_stream_moderation.environment import SocialStreamModerationEnv
from envs.social_stream_moderation.models import ModerationAction, PolicyMode

def compare_policies():
    env = SocialStreamModerationEnv()
    seed = 101
    
    # We'll use a specific post that is 'MILD_ABUSE' (e.g., "You are an absolute idiot")
    # For a deterministic comparison, we'll force the policy mode.
    
    print("="*40)
    print("DEMO: Comparing Platform Policy Regimes")
    print("="*40)
    
    # Custom post for demo
    demo_post = {
        "post_id": "demo_mild_abuse",
        "text": "You are an absolute idiot for thinking that.",
        "harm_label": "MILD_ABUSE",
        "user_history_summary": "no_prior_violations",
        "context_type": "reply_heated",
        "user_group": "A"
    }
    
    # Mock data loading for the demo
    env.reset(task_name="clear_cut_moderation", seed=seed)
    env.episode_posts = [type('Post', (), demo_post)] # Simple hack
    
    # Case 1: LENIENT Mode
    print("\n[Scenario 1: LENIENT Policy Mode]")
    print(f"Content: '{demo_post['text']}'")
    env.policy_mode = PolicyMode.LENIENT
    # In lenient mode, a common action would be ALLOW or WARNING
    action = ModerationAction.ALLOW_WITH_WARNING
    _, reward, _, _ = env.step(action)
    print(f"Action: {action.value} | Reward: {reward}")
    print("Interpretation: Lenient modes prioritize user expression; mild insults get a warning.")

    # Case 2: STRICT Mode
    env.reset(task_name="clear_cut_moderation", seed=seed)
    env.episode_posts = [type('Post', (), demo_post)]
    print("\n[Scenario 2: STRICT Policy Mode]")
    print(f"Content: '{demo_post['text']}'")
    env.policy_mode = PolicyMode.STRICT
    # In strict mode, ALLOWING mild abuse is penalized (-0.2 in our grader)
    # A strict action would be BAN_USER or SOFT_HIDE
    action_allow = ModerationAction.ALLOW
    _, reward_allow, _, _ = env.step(action_allow)
    print(f"Action: ALLOW | Reward: {reward_allow}")
    
    env.reset(task_name="clear_cut_moderation", seed=seed)
    env.episode_posts = [type('Post', (), demo_post)]
    env.policy_mode = PolicyMode.STRICT
    action_ban = ModerationAction.BAN_USER
    _, reward_ban, _, _ = env.step(action_ban)
    print(f"Action: BAN_USER | Reward: {reward_ban}")
    print("Interpretation: Strict modes prioritize safety; 'ALLOWING' even mild abuse is heavily penalized.")
    
    print("\nConclusion: The environment successfully models different platform priorities.")

if __name__ == "__main__":
    compare_policies()
