import json
import os
import re
from datasets import load_dataset


def preprocess_data():
    ds = load_dataset("nvidia/OpenMathReasoning", split="train", streaming=True)

    output_file = "data/grpo_math_v1.jsonl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = 0
    with open(output_file, "w") as f:
        for item in ds:
            count += 1
            question = item.get('question') or item.get('problem')
            answer = item.get('answer') or item.get('solution')

            if not question or not answer:
                continue

            entry = {
                "system": "你是一个数学解题专家。请一步步推理，并在最后把最终答案放在 \\boxed{} 中。",
                "query": question,
                "response": answer
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if count >= 500:
                break

    print(f"[Info] Data preparation completed! Saved {count} items to {output_file}")

if __name__ == "__main__":
    preprocess_data()