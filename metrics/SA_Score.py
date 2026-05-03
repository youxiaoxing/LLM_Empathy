from openai import OpenAI
import json
from tqdm import tqdm

client_openai = OpenAI(
    api_key="-"
)

COMPARATIVE_EMOTION_INTENSITY_PROMPT = """
You are evaluating the **emotional intensity** of TWO text summaries on a scale from 0.0 to 1.0.

**Scoring Guide:**
- **0.0-0.2**: Emotionally flat, clinical, bureaucratic language. No visceral or affective content.
- **0.3-0.4**: Mild emotional content, but heavily sanitized or abstracted.
- **0.5-0.6**: Moderate emotional presence with some concrete details.
- **0.7-0.8**: Strong emotional intensity with vivid, embodied language.
- **0.9-1.0**: Maximally intense emotional content with raw, unfiltered affective expression.

**Consider:**
- Use of visceral, sensory, concrete words (high intensity) vs. abstract euphemisms (low intensity)
- Emotional directness (high) vs. clinical distancing (low)
- Presence of strong emotion words (grief, rage, despair, joy) vs. neutral descriptors
- Acknowledgment of pain/conflict vs. sanitization

---

**SUMMARY 1:**
{reference}

**SUMMARY 2:**
{generated}

---

**OUTPUT FORMAT:**
Provide ONLY a valid JSON object with two scores, nothing else:

{{"reference_score": 0.85, "generated_score": 0.32}}
"""

def compare_emotional_intensity(reference, generated, model="gpt-5.2-ca"):
    prompt = COMPARATIVE_EMOTION_INTENSITY_PROMPT.format(
        reference=reference,
        generated=generated
    )
    
    try:
        response = client_openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0
        )
        
        response_text = response.choices[0].message.content.strip()

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        scores = json.loads(response_text)
        
        return {
            "reference_score": float(scores.get("reference_score", 0)),
            "generated_score": float(scores.get("generated_score", 0))
        }
    
    except Exception as e:
        print(f"Error scoring texts: {e}")
        print(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
        return {
            "reference_score": None,
            "generated_score": None
        }

if __name__ == "__main__":

    dataList = []
    
    input_file = "./qwen_max_article_generation.jsonl"
    output_file = "./qwen_max_new.jsonl"

    print(input_file)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            dataList.append(json.loads(line))

    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataList, desc="Processing"):
            if "reference" not in item and "text" in item:
                item["reference"] = item["text"]
            scores = compare_emotional_intensity(
                reference=item.get("reference", ""),
                generated=item.get("generate", "")
            )

            item["ref_score"] = scores["reference_score"]
            item["gen_score"] = scores["generated_score"]
            

            if scores["reference_score"] is not None and scores["generated_score"] is not None:
                item["emotion_compression"] = scores["reference_score"] - scores["generated_score"]
            else:
                item["emotion_compression"] = None
            
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    
    with open(output_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    
    valid_results = [r for r in results if r["ref_score"] is not None and r["gen_score"] is not None]
    
    if valid_results:

        avg_ref = sum(r["ref_score"] for r in valid_results) / len(valid_results)
        avg_gen = sum(r["gen_score"] for r in valid_results) / len(valid_results)
        avg_compression = sum(r["emotion_compression"] for r in valid_results) / len(valid_results)

        individual_rates = []
        for r in valid_results:
            if r["ref_score"] > 0:
                rate = (r["ref_score"] - r["gen_score"]) / r["ref_score"]
                individual_rates.append(rate)
        avg_compression_rate_method1 = sum(individual_rates) / len(individual_rates) if individual_rates else 0
        
        avg_compression_rate_method2 = (avg_ref - avg_gen) / avg_ref if avg_ref > 0 else 0
        
        print(f"\n📊 Statistics:")
        print(f"Valid samples: {len(valid_results)}")
        print(f"\n--- Emotion Scores ---")
        print(f"Average Reference Score: {avg_ref:.3f}")
        print(f"Average Generated Score: {avg_gen:.3f}")
        print(f"\n--- Absolute Compression ---")
        print(f"Average Emotion Compression (Absolute): {avg_compression:.3f}")
        print(f"\n--- Relative Compression Rate ---")
        print(f"Method 1 (avg of individual rates): {avg_compression_rate_method1:.3f} ({avg_compression_rate_method1*100:.1f}%)")
        print(f"Method 2 (rate of averages): {avg_compression_rate_method2:.3f} ({avg_compression_rate_method2*100:.1f}%)")