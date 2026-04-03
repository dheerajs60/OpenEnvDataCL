import os
import json
from openai import OpenAI
from env.cleaner_env import DataCleanerEnv
from env.models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "fake-token")

def get_action_from_llm(obs: dict) -> dict:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    prompt = f"""
You are a data cleaning agent.
Observation:
{json.dumps(obs, indent=2)}

Return a JSON action exactly in this structure:
{{"operation": "...", "column": "...", "value": "..."}}
Allowed operations: fill_missing, remove_duplicates, standardize_date, normalize_text, rename_column, fix_category, stop.
If you are done or don't know what to do, output {{"operation": "stop"}}.
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"operation": "stop"}

def run_inference():
    task_diff = "hard"
    print(f"[START] task={task_diff} env=data_cleaner model={MODEL_NAME}")
    
    env = DataCleanerEnv()
    try:
        obs = env.reset(task_diff)
        obs_dict = obs.model_dump(by_alias=True) if hasattr(obs, 'model_dump') else hasattr(obs, 'dict') and obs.dict(by_alias=True) or obs
        
        done = False
        step_count = 1
        rewards = []
        
        while not done:
            action_dict = get_action_from_llm(obs_dict)
            action = Action(**action_dict)
            
            obs, reward, done, info = env.step(action)
            obs_dict = obs.model_dump(by_alias=True) if hasattr(obs, 'model_dump') else hasattr(obs, 'dict') and obs.dict(by_alias=True) or obs
            
            reward_val = reward.score if hasattr(reward, 'score') else reward.get('score', 0.0)
            rewards.append(f"{reward_val:.2f}")
            
            error_val = info.get("error", "null")
            print(f"[STEP] step={step_count} action={action.operation} reward={reward_val:.2f} done={str(done).lower()} error={error_val}")
            
            step_count += 1
            if done:
                break
                
        print(f"[END] success=true steps={step_count-1} rewards={','.join(rewards)}")
    except Exception as e:
        print(f"[END] success=false error={str(e)}")

if __name__ == "__main__":
    run_inference()
