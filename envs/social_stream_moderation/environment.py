import json
import os
import random
from typing import List, Dict, Any, Tuple, Optional
from .models import HarmLabel, ModerationAction, State, PolicyMode, Post, UserGroup
from .tasks import TASKS, TaskConfig, resolve_task
from .graders import compute_per_post_reward, grade_episode, get_grader

class SocialStreamModerationEnv:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = os.path.dirname(__file__)
        self.data_dir = data_dir
        self.current_task: Optional[TaskConfig] = None
        self.episode_posts: List[Post] = []
        self.step_index = 0
        self.done = False
        self.episode_history: List[Dict[str, Any]] = []
        self.policy_mode = PolicyMode.NORMAL
        self._grader = None

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None):
        """Standard OpenEnv V4 interface for initializing the environment."""
        # For local project structure, we just return an instance.
        return cls()
        
    async def reset(self, task_name: str = "clear_cut_moderation", seed: Optional[int] = None) -> State:
        """Resets the environment with a given task and seed."""
        if seed is not None:
            random.seed(seed)
            
        try:
            self.current_task = resolve_task(task_name)
        except KeyError:
            raise ValueError(f"Task {task_name} not found in TASKS.")
        data_path = os.path.join(self.data_dir, self.current_task.data_file)
        
        with open(data_path, "r") as f:
            all_posts = json.load(f)
            
        # Sample posts for the episode
        sampled_posts = random.sample(all_posts, min(len(all_posts), self.current_task.episode_length))
        self.episode_posts = [Post(**p) for p in sampled_posts]
        
        # Reset state
        self.step_index = 0
        self.done = False
        self.episode_history = []
        self.policy_mode = self.current_task.policy_mode

        # Initialise the grader for this task
        self._grader = get_grader(self.current_task.grader_id)
        self._grader.reset()

        return self._get_state()
        
    def _get_state(self) -> State:
        """Returns the current state representation."""
        if self.step_index >= len(self.episode_posts):
            return None # Should not happen if done correctly
            
        post = self.episode_posts[self.step_index]
        return State(
            post_id=post.post_id,
            text=post.text,
            user_history_summary=post.user_history_summary.value,
            context_type=post.context_type.value,
            platform_policy_mode=self.policy_mode.value,
            user_group=post.user_group.value,
            step_index=self.step_index,
            total_steps=len(self.episode_posts)
        )
    def state(self) -> State:
        """Standard OpenEnv interface to return the current observation."""
        return self._get_state()   

    async def step(self, action: ModerationAction) -> Tuple[Optional[State], float, bool, Dict[str, Any]]:
        """Processes one moderation action."""
        if self.done:
            raise RuntimeError("Episode is already finished. Call reset() first.")
            
        current_post = self.episode_posts[self.step_index]
        
        # Validate action
        if not isinstance(action, ModerationAction):
            try:
                action = ModerationAction(action)
            except ValueError:
                # Default to soft hide or warning if invalid
                action = ModerationAction.ALLOW_WITH_WARNING
                
        # Compute reward
        reward = compute_per_post_reward(current_post.harm_label, action, self.policy_mode)
        
        # Log to history for final grading (include context_type and
        # policy_mode so context-aware graders can use them)
        self.episode_history.append({
            "post_id": current_post.post_id,
            "harm_label": current_post.harm_label,
            "user_group": current_post.user_group,
            "context_type": current_post.context_type,
            "policy_mode": self.policy_mode,
            "action": action,
            "reward": reward
        })
        
        # Increment step
        self.step_index += 1
        
        # Check if done
        if self.step_index >= len(self.episode_posts):
            self.done = True
            
        next_state = self._get_state() if not self.done else None
        
        # Return next_state, reward, done, info
        info = {
            "ground_truth_label": current_post.harm_label,
            "action_taken": action.value,
            "reward": reward
        }
        
        if self.done:
            # Use the task-specific grader when available
            if self._grader is not None:
                final_score = self._grader.grade(self.episode_history)
            else:
                final_score = grade_episode(self.episode_history, self.current_task.use_fairness)
            info["score"] = final_score
            info["grader_id"] = self.current_task.grader_id
            
        return next_state, reward, self.done, info
