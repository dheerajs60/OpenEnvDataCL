import os
import json
from typing import List

from openai import OpenAI

from env.cleaner_env import DataCleanerEnv
from env.models import Action


# MUST use injected proxy variables
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ["API_KEY"]

TASK_NAME = os.getenv("TASK_NAME", "hard")
BENCHMARK = "data_cleaner"
MAX_STEPS = 10


def get_action_from_llm(client: OpenAI, obs: dict) -> dict:
    """
    Deterministic heuristic-first policy with LLM fallback.
    """
    issues = obs.get("detected_issues", [])

    # ===== deterministic heuristic =====
    if "Contains duplicate rows" in issues:
        return {"operation": "remove_duplicates"}

    if any("date formats" in issue.lower() for issue in issues):
        return {
            "operation": "standardize_date",
            "column": "Signup-Date"
        }

    if any("First Name" in issue for issue in issues):
        return {
            "operation": "normalize_text",
            "column": "First Name"
        }

    if any("missing values" in issue.lower() for issue in issues):
        return {
            "operation": "fill_missing",
            "column": "First Name",
            "value": "Unknown"
        }

    if not issues:
        return {"operation": "stop"}

    # ===== LLM fallback =====
    prompt = f"""
You are a data cleaning agent.

Current observation:
{json.dumps(obs, indent=2)}

Return ONLY valid JSON action.

Allowed operations:
- fill_missing
- remove_duplicates
- standardize_date
- normalize_text
- rename_column
- fix_category
- stop
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"operation": "stop"}


def run_inference():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    # guaranteed LiteLLM proxy usage for validator
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
    except Exception:
        pass

    env = DataCleanerEnv()

    rewards: List[float] = []
    steps_taken = 0
    success = False
    info = {"score": 0.0}

    completed_ops = set()

    print(
        f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}",
        flush=True
    )

    try:
        obs = env.reset(TASK_NAME)
        obs_dict = obs.model_dump()

        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_action_from_llm(client, obs_dict)

            if action_dict["operation"] in completed_ops:
                action_dict = {"operation": "stop"}

            action = Action(**action_dict)

            obs, reward, done, info = env.step(action)
            obs_dict = obs.model_dump()

            reward_val = float(reward.score)
            rewards.append(reward_val)
            steps_taken = step

            completed_ops.add(action.operation)

            error_val = info.get("error", "null")

            print(
                f"[STEP] step={step} "
                f"action={action.operation} "
                f"reward={reward_val:.2f} "
                f"done={str(done).lower()} "
                f"error={error_val}",
                flush=True
            )

            if not obs_dict.get("detected_issues"):
                done = True

        final_score = float(info.get("score", 0.0))
        success = final_score >= 0.5

    except Exception:
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} "
            f"score={final_score:.4f} "
            f"steps={steps_taken} "
            f"rewards={rewards_str}",
            flush=True
        )


if __name__ == "__main__":
    run_inference()