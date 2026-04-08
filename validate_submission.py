import subprocess
import os

def run_test_task(task_name: str, seed: int = 42):
    print(f"Testing Task: {task_name}...")
    try:
        # Run inference.py locally and capture output
        result = subprocess.run(
            ['python3', 'inference.py', task_name, str(seed)],
            capture_output=True,
            text=True,
            timeout=30 # Safety timeout
        )
        output = result.stdout
        
        # Validation checks
        has_start = f"[START] task={task_name} | seed={seed}" in output
        has_step = "[STEP] step=0" in output
        has_end = f"[END]" in output
        
        if has_start and has_step and has_end:
            print(f"  [PASS] {task_name} | Logs formatted correctly.")
        else:
            print(f"  [FAIL] {task_name} | Logs missing markers!")
            print(f"Output received:\n{output}")
            
    except Exception as e:
        print(f"  [FAIL] {task_name} | Error: {e}")

if __name__ == "__main__":
    print("-" * 40)
    print("🔍 PolicyPulse AI: Pre-Submission Validator")
    print("-" * 40)
    
    # 1. Check file existence
    required_files = ["inference.py", "app.py", "openenv.yaml", "Dockerfile", "README.md", "requirements.txt"]
    for f in required_files:
        if os.path.exists(f):
            print(f"  [PASS] {f} exists in root.")
        else:
            print(f"  [FAIL] {f} is missing from root!")

    # 2. Check for Envs directory
    if os.path.exists("envs/social_stream_moderation"):
        print("  [PASS] Environment package found.")
    else:
        print("  [FAIL] envs/social_stream_moderation is missing!")

    # 3. Test tasks with inference.py
    print("\n📦 Testing Baseline Tasks...")
    run_test_task("clear_cut_moderation")
    run_test_task("nuanced_sarcastic")
    run_test_task("policy_fairness")
    
    print("-" * 40)
    print("Validator Finished. If all [PASS], you are ready for submission!")
    print("-" * 40)
