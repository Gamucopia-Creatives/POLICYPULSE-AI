from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class HarmLabel(str, Enum):
    SAFE = "SAFE"
    MILD_ABUSE = "MILD_ABUSE"
    SEVERE_ABUSE_HATE = "SEVERE_ABUSE_HATE"
    SEXUAL_EXPLICIT = "SEXUAL_EXPLICIT"
    SELF_HARM_PROMO = "SELF_HARM_PROMO"

class ModerationAction(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_WARNING = "ALLOW_WITH_WARNING"
    SOFT_HIDE = "SOFT_HIDE"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"
    BAN_USER = "BAN_USER"

class PolicyMode(str, Enum):
    LENIENT = "lenient"
    NORMAL = "normal"
    STRICT = "strict"

class UserHistory(str, Enum):
    NO_PRIOR_VIOLATIONS = "no_prior_violations"
    ONE_MILD = "one_mild"
    REPEATED_SEVERE = "repeated_severe"

class ContextType(str, Enum):
    ROOT_POST = "root_post"
    REPLY_HEATED = "reply_heated"
    QUOTE_POST = "quote_post"

class UserGroup(str, Enum):
    A = "A"
    B = "B"

class Post(BaseModel):
    """Internal model for a post in the environment."""
    post_id: str
    text: str
    harm_label: HarmLabel
    user_history_summary: UserHistory
    context_type: ContextType
    user_group: UserGroup

class State(BaseModel):
    """The state returned to the agent."""
    post_id: str
    text: str
    user_history_summary: str
    context_type: str
    platform_policy_mode: str
    user_group: str
    step_index: int
    total_steps: int
