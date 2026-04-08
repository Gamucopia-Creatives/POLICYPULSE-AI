from typing import List, Dict, Any
from .models import HarmLabel, ModerationAction, UserGroup, PolicyMode

REWARD_TABLE = {
    HarmLabel.SAFE: {
        ModerationAction.ALLOW: 1.0,
        ModerationAction.ALLOW_WITH_WARNING: 0.9,
        ModerationAction.SOFT_HIDE: 0.9,
        ModerationAction.ESCALATE_HUMAN: 0.8,
        ModerationAction.BAN_USER: 0.5,
    },
    HarmLabel.MILD_ABUSE: {
        ModerationAction.ALLOW: 0.6,
        ModerationAction.ALLOW_WITH_WARNING: 0.95,
        ModerationAction.SOFT_HIDE: 0.95,
        ModerationAction.ESCALATE_HUMAN: 0.85,
        ModerationAction.BAN_USER: 0.7,
    },
    HarmLabel.SEVERE_ABUSE_HATE: {
        ModerationAction.ALLOW: 0.0,
        ModerationAction.ALLOW_WITH_WARNING: 0.3,
        ModerationAction.SOFT_HIDE: 0.3,
        ModerationAction.ESCALATE_HUMAN: 0.9,
        ModerationAction.BAN_USER: 1.0,
    },
    HarmLabel.SEXUAL_EXPLICIT: {
        ModerationAction.ALLOW: 0.1,
        ModerationAction.ALLOW_WITH_WARNING: 0.4,
        ModerationAction.SOFT_HIDE: 0.7,
        ModerationAction.ESCALATE_HUMAN: 0.9,
        ModerationAction.BAN_USER: 1.0,
    },
    HarmLabel.SELF_HARM_PROMO: {
        ModerationAction.ALLOW: 0.0,
        ModerationAction.ALLOW_WITH_WARNING: 0.2,
        ModerationAction.SOFT_HIDE: 0.2,
        ModerationAction.ESCALATE_HUMAN: 1.0,
        ModerationAction.BAN_USER: 0.7,
    }
}

def compute_per_post_reward(harm_label: HarmLabel, action: ModerationAction, policy_mode: PolicyMode = PolicyMode.NORMAL) -> float:
    base_reward = REWARD_TABLE.get(harm_label, {}).get(action, 0.0)
    
    # Adjust for policy mode if needed
    if policy_mode == PolicyMode.LENIENT:
        # Inient: penalize over-censorship (Hiding safe content) more
        if harm_label == HarmLabel.SAFE and action in [ModerationAction.SOFT_HIDE, ModerationAction.BAN_USER]:
            base_reward -= 0.1
    elif policy_mode == PolicyMode.STRICT:
        # Strict: penalize under-moderation more
        if harm_label != HarmLabel.SAFE and action == ModerationAction.ALLOW:
            base_reward -= 0.2
            
    return max(0.0, min(1.0, base_reward))

def calculate_fairness_penalty(episode_history: List[Dict[str, Any]], threshold: float = 0.3) -> float:
    """Calculates fairness penalty if group ban rate imbalance is high."""
    group_stats = {
        UserGroup.A: {"bans": 0, "total": 0},
        UserGroup.B: {"bans": 0, "total": 0}
    }
    
    for entry in episode_history:
        group = entry.get("user_group")
        action = entry.get("action")
        if group in group_stats:
            group_stats[group]["total"] += 1
            if action == ModerationAction.BAN_USER:
                group_stats[group]["bans"] += 1
                
    ban_rates = {}
    for group, stats in group_stats.items():
        if stats["total"] > 0:
            ban_rates[group] = stats["bans"] / stats["total"]
        else:
            ban_rates[group] = 0.0
            
    diff = abs(ban_rates.get(UserGroup.A, 0.0) - ban_rates.get(UserGroup.B, 0.0))
    if diff > threshold:
        # Return penalty up to 0.2
        return min(0.2, (diff - threshold) * 0.5)
    
    return 0.0

def grade_episode(episode_history: List[Dict[str, Any]], use_fairness: bool = False) -> float:
    if not episode_history:
        return 0.0
        
    rewards = [entry.get("reward", 0.0) for entry in episode_history]
    mean_reward = sum(rewards) / len(rewards)
    
    if use_fairness:
        penalty = calculate_fairness_penalty(episode_history)
        mean_reward = max(0.0, mean_reward - penalty)
        
    return round(mean_reward, 4)
