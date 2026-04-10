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

# Legacy aliases used by validate_submission.py and inference.py CLI.
# Kept in a separate dict so that TASKS.values() only yields the 3 canonical
# entries (avoids duplicates in the /tasks API endpoint).
TASK_ALIASES = {
    "clear_cut_moderation": "Task 1: Basic Safety",
    "nuanced_sarcastic": "Task 2: Context & Nuance",
    "policy_fairness": "Task 3: Fairness & Bias",
}


def resolve_task(name: str) -> TaskConfig:
    """Look up a task by canonical name or legacy alias."""
    if name in TASKS:
        return TASKS[name]
    canonical = TASK_ALIASES.get(name)
    if canonical and canonical in TASKS:
        return TASKS[canonical]
    raise KeyError(f"Unknown task: {name}")
