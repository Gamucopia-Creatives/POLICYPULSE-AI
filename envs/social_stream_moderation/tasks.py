from pydantic import BaseModel
from typing import List, Optional
from .models import PolicyMode

class TaskConfig(BaseModel):
    name: str
    difficulty: str
    data_file: str
    episode_length: int
    policy_mode: PolicyMode
    use_fairness: bool = False

TASKS = {
    "clear_cut_moderation": TaskConfig(
        name="clear_cut_moderation",
        difficulty="easy",
        data_file="data_easy.json",
        episode_length=10,
        policy_mode=PolicyMode.NORMAL,
        use_fairness=False
    ),
    "nuanced_sarcastic": TaskConfig(
        name="nuanced_sarcastic",
        difficulty="medium",
        data_file="data_medium.json",
        episode_length=15,
        policy_mode=PolicyMode.NORMAL,
        use_fairness=False
    ),
    "policy_fairness": TaskConfig(
        name="policy_fairness",
        difficulty="hard",
        data_file="data_hard.json",
        episode_length=20,
        policy_mode=PolicyMode.NORMAL, # Can vary via config if needed or randomized in reset
        use_fairness=True
    )
}
