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
    grader_id: str = "basic_safety_grader"

TASKS = {
    "Task 1: Basic Safety": TaskConfig(
        name="Task 1: Basic Safety",
        difficulty="easy",
        data_file="data_easy.json",
        episode_length=10,
        policy_mode=PolicyMode.NORMAL,
        use_fairness=False,
        grader_id="basic_safety_grader"
    ),
    "Task 2: Context & Nuance": TaskConfig(
        name="Task 2: Context & Nuance",
        difficulty="medium",
        data_file="data_medium.json",
        episode_length=15,
        policy_mode=PolicyMode.NORMAL,
        use_fairness=False,
        grader_id="context_nuance_grader"
    ),
    "Task 3: Fairness & Bias": TaskConfig(
        name="Task 3: Fairness & Bias",
        difficulty="hard",
        data_file="data_hard.json",
        episode_length=20,
        policy_mode=PolicyMode.NORMAL,
        use_fairness=True,
        grader_id="fairness_bias_grader"
    )
}
