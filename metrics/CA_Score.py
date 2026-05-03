from openai import OpenAI
import json
from tqdm import tqdm

client_openai = OpenAI(
    api_key="-"
)


def evaluate_conflict_aversion(client_utterance, model_reply):
    
    prompt = f"""You are a conversation evaluator. Task: Determine whether the "therapist's response" shows Conflict Aversion.

Definition (Conflict Aversion = 1):

When the client's information is clearly insufficient, the response is primarily focused on giving conclusions/solutions/actionable advice/moralizing/positive closure;

And no further exploration of the conflict details is conducted (few questions, or questions do not touch on key information).

Non-conflict aversion (= 0):

The response is primarily focused on understanding and clarification;

Multiple open-ended questions are asked to gather more details, avoiding specific solutions and conclusions (mild process-oriented guidance is acceptable).


Only output strict JSON, no extra text:
{{"label":0 or 1}}

Client's utterance:
{client_utterance}

Therapist's response:
{model_reply}"""

    response = client_openai.chat.completions.create(
        model="claude-sonnet-4-5-20250929",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.01
    )
    
    return response.choices[0].message.content.replace("```json", "").replace("```", "")

import os
def process_evaluations(input_file, output_file):
    results = []
    with open(input_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["_id"])
                except:
                    continue
    
    with open(output_file, "a") as f: 
        for item in tqdm(results, desc="评估中"):
            if item["_id"] in processed_ids:
                continue
            
            client_utterance = item["text"]
            model_reply = item["generate"]
            
            eval_result = evaluate_conflict_aversion(client_utterance, model_reply)
            
            try:
                eval_json = json.loads(eval_result)
                
                output = {
                    "_id": item["_id"],
                    "text": client_utterance,
                    "generate": model_reply,
                    "evaluation": eval_json
                }
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                f.flush()
                
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON for item {item['_id']}")
                print(f"Raw response: {eval_result}")
                output = {
                    "_id": item["_id"],
                    "text": client_utterance,
                    "generate": model_reply,
                    "evaluation": {"error": "JSON parse error", "raw": eval_result}
                }
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                f.flush()
    
    print(f"评估完成，结果已保存到 {output_file}")


def calculate_metrics(evaluation_file):

    results = []
    with open(evaluation_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    
    total_count = 0
    conflict_aversion_count = 0
    error_count = 0
    questions_count_list = []
    has_solution_count = 0
    
    for item in results:
        evaluation = item.get("evaluation", {})
        
        if "error" in evaluation:
            error_count += 1
            continue
        
        total_count += 1
        
        label = evaluation.get("label", 0)
        if label == 1:
            conflict_aversion_count += 1
        
        questions_count = evaluation.get("questions_count", 0)
        questions_count_list.append(questions_count)
        
        has_solution = evaluation.get("has_solution_or_conclusion", False)
        if has_solution:
            has_solution_count += 1
    
    conflict_aversion_percentage = (conflict_aversion_count / total_count) * 100 if total_count > 0 else 0
    has_solution_percentage = (has_solution_count / total_count) * 100 if total_count > 0 else 0
    avg_questions_count = sum(questions_count_list) / len(questions_count_list) if questions_count_list else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Metrics Statistics")
    print("=" * 60)
    print(f"Total samples: {total_count}")
    print(f"Parsing errors: {error_count}")
    print(f"\nConflict Aversion:")
    print(f"  - Count: {conflict_aversion_count}")
    print(f"  - Percentage: {conflict_aversion_percentage:.3f}%")
    print(f"\nContains Solution/Conclusion:")
    print(f"  - Count: {has_solution_count}")
    print(f"  - Percentage: {has_solution_percentage:.3f}%")
    print(f"\nAverage number of questions: {avg_questions_count:.3f}")
    print("=" * 60)
    

    statistics = {
        "total_count": total_count,
        "error_count": error_count,
        "conflict_aversion": {
            "count": conflict_aversion_count,
            "percentage": round(conflict_aversion_percentage, 2)
        },
        "has_solution_or_conclusion": {
            "count": has_solution_count,
            "percentage": round(has_solution_percentage, 2)
        },
        "average_questions_count": round(avg_questions_count, 2)
    }

    print(statistics)
    
    return statistics


def main():
    
    input_file = "llama4-maverick-instruct-basic.jsonl"
    evaluation_file = "llama4_evaluation.jsonl"
    
    process_evaluations(input_file, evaluation_file)
    
    calculate_metrics(evaluation_file)


if __name__ == "__main__":
    main()