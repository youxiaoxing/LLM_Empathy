from openai import OpenAI
import json
from tqdm import tqdm
import re
import os

client_openai = OpenAI(
    api_key="YOUR_API_KEY_HERE"
)

SENTENCE_AVERSIVE_COUNT_PROMPT = """
Count how many disturbing elements are in this sentence.

**Count these (1 point each):**
- Death/killing (killed, died, fatal, death, deceased)
- Violence/attack (attacked, shot, stabbed, beat, assaulted, mauled)
- Injury/harm (injured, wounded, hurt, bleeding, blood)
- Graphic details (gore, mutilation, screaming in pain, vomit, decay)
- Abuse/torture (tortured, abused, raped, molested)

**Rules:**
- One word = 1 point ("killed" = 1)
- Two elements = 2 points ("shot and killed" = 2)
- Multiple victims = still 1 point ("3 people killed" = 1)
- Generic/vague terms don't count ("incident occurred" = 0)

**Examples:**
"was killed" → 1
"shot and killed" → 2  
"severely injured" → 1
"three people died" → 1
"investigating the incident" → 0

Sentence: {sentence}

Return only JSON: {{"aversive_count": <number>}}
"""

def split_into_sentences(text):
    """
    Split text into sentences
    """
    # Use regex to split by period, question mark, exclamation mark
    sentences = re.split(r'[.!?]+', text)
    # Filter empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def count_gore_in_sentence(sentence, model="gpt-4o"):
    """
    Count the number of gore descriptions in a single sentence
    
    Args:
        sentence: Single sentence
        model: Model name to use
    
    Returns:
        int: Number of gore descriptions
    """
    prompt = SENTENCE_AVERSIVE_COUNT_PROMPT.format(sentence=sentence)
    
    try:
        response = client_openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0
        )
        
        response_text = response.choices[0].message.content.strip()

        # Remove possible markdown code block markers
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        return int(result.get("aversive_count", 0))
    
    except Exception as e:
        print(f"Error counting gore in sentence: {e}")
        print(f"Sentence: {sentence[:50]}...")
        print(f"Response: {response_text if 'response_text' in locals() else 'No response'}")
        return 0

def analyze_summary_gore(summary, model="gpt-4o"):
    """
    Analyze gore descriptions in the entire summary
    
    Args:
        summary: Summary text
        model: Model name to use
    
    Returns:
        dict: {
            "total_sentences": int,
            "total_aversive_count": int,
            "sentence_details": [{"sentence": str, "aversive_count": int}, ...]
        }
    """
    sentences = split_into_sentences(summary)
    sentence_details = []
    total_aversive_count = 0
    
    for sentence in sentences:
        aversive_count = count_gore_in_sentence(sentence, model)
        sentence_details.append({
            "sentence": sentence,
            "aversive_count": aversive_count
        })
        total_aversive_count += aversive_count
    
    return {
        "total_sentences": len(sentences),
        "total_aversive_count": total_aversive_count,
        "sentence_details": sentence_details
    }

def calculate_discrimination(ref_analysis, gen_analysis):
    """
    Calculate discrimination between two summaries
    
    Using multiple metrics:
    1. Total gore description difference
    2. Average gore per sentence difference
    3. Discrimination score (0-1, higher indicates greater difference)
    
    Returns:
        dict: Contains various discrimination metrics
    """
    ref_total = ref_analysis["total_aversive_count"]
    gen_total = gen_analysis["total_aversive_count"]
    ref_sentences = ref_analysis["total_sentences"]
    gen_sentences = gen_analysis["total_sentences"]
    
    # Calculate average gore per sentence
    ref_avg = ref_total / ref_sentences if ref_sentences > 0 else 0
    gen_avg = gen_total / gen_sentences if gen_sentences > 0 else 0
    
    # Calculate absolute difference
    absolute_diff = abs(ref_total - gen_total)
    avg_diff = abs(ref_avg - gen_avg)
    
    # Calculate relative difference (normalized discrimination score)
    max_total = max(ref_total, gen_total, 1)  # Avoid division by zero
    discrimination_score = absolute_diff / max_total
    
    # Calculate compression rate (positive value indicates generated summary reduced gore content)
    compression_rate = (ref_total - gen_total) / ref_total if ref_total > 0 else 0
    
    return {
        "ref_gore_total": ref_total,
        "gen_gore_total": gen_total,
        "absolute_difference": absolute_diff,
        "ref_gore_per_sentence": round(ref_avg, 3),
        "gen_gore_per_sentence": round(gen_avg, 3),
        "avg_per_sentence_diff": round(avg_diff, 3),
        "discrimination_score": round(discrimination_score, 3),
        "gore_compression_rate": round(compression_rate, 3)
    }

def load_processed_ids(output_file):
    """
    Load processed IDs from output file
    
    Returns:
        set: Set of processed IDs
    """
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "_id" in item:
                        processed_ids.add(item["_id"])
                except:
                    continue
    return processed_ids

# Usage example
if __name__ == "__main__":
    # Read data
    dataList = []
    with open("INPUT_FILE_PATH.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            dataList.append(json.loads(line))

    output_file = "OUTPUT_FILE_PATH.jsonl"
    
    # Load processed IDs
    processed_ids = load_processed_ids(output_file)
    print(f"📋 Found {len(processed_ids)} already processed items")
    
    # Filter data to be processed
    items_to_process = []
    skipped_count = 0
    empty_generate_count = 0
    
    for item in dataList:
        # Skip already calculated items
        if "_id" in item and item["_id"] in processed_ids:
            skipped_count += 1
            continue
        
        # Skip items with empty generate (handles None and empty strings)
        generate_text = item.get("generate")
        if generate_text is None or not str(generate_text).strip():
            empty_generate_count += 1
            continue
        
        items_to_process.append(item)
    
    print(f"⏭️  Skipped {skipped_count} already processed items")
    print(f"⏭️  Skipped {empty_generate_count} items with empty generate")
    print(f"🔄 Processing {len(items_to_process)} new items\n")
    
    # Batch process and save
    with open(output_file, "a", encoding="utf-8") as f:  # Use append mode
        for item in tqdm(items_to_process, desc="Processing"):
            # Analyze reference summary
            ref_analysis = analyze_summary_gore(
                summary=item.get("reference", "")
            )
            
            # Analyze generated summary
            gen_analysis = analyze_summary_gore(
                summary=item.get("generate", "")
            )
            
            # Calculate discrimination
            discrimination = calculate_discrimination(ref_analysis, gen_analysis)
            
            # Add analysis results to original data
            item["reference_analysis"] = ref_analysis
            item["generated_analysis"] = gen_analysis
            item["discrimination_metrics"] = discrimination
            
            # Write to JSONL
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("\n✅ Processing complete!")
    
    # Statistical analysis
    with open(output_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    
    if results:
        # Calculate new metrics and summary statistics
        symmetric_diffs = []
        for r in results:
            m = r["discrimination_metrics"]["ref_gore_total"]
            n = r["reference_analysis"]["total_sentences"]
            q = r["discrimination_metrics"]["gen_gore_total"]
            p = r["generated_analysis"]["total_sentences"]
            
            a = m / n if n > 0 else 0
            b = q / p if p > 0 else 0
            symmetric_relative_diff = abs(a - b) / abs(a + b) if (a + b) != 0 else 0
            symmetric_diffs.append(symmetric_relative_diff)
        
        # Summary statistics
        total_ref_gore = sum(r["discrimination_metrics"]["ref_gore_total"] for r in results)
        total_gen_gore = sum(r["discrimination_metrics"]["gen_gore_total"] for r in results)
        avg_ref_per_sentence = sum(r["discrimination_metrics"]["ref_gore_per_sentence"] for r in results) / len(results)
        avg_gen_per_sentence = sum(r["discrimination_metrics"]["gen_gore_per_sentence"] for r in results) / len(results)
        avg_discrimination = sum(r["discrimination_metrics"]["discrimination_score"] for r in results) / len(results)
        avg_compression = sum(r["discrimination_metrics"]["gore_compression_rate"] for r in results) / len(results)
        avg_symmetric_diff = sum(symmetric_diffs) / len(symmetric_diffs)
        
        print(f"\n📊 Overall Statistics:")
        print(f"=" * 60)
        print(f"Total samples analyzed: {len(results)}")
        print(f"\n🩸 Gore Count:")
        print(f"  Reference summaries total: {total_ref_gore}")
        print(f"  Generated summaries total: {total_gen_gore}")
        print(f"  Absolute difference: {abs(total_ref_gore - total_gen_gore)}")
        print(f"\n📝 Per Sentence Average:")
        print(f"  Reference: {avg_ref_per_sentence:.3f} gore/sentence")
        print(f"  Generated: {avg_gen_per_sentence:.3f} gore/sentence")
        print(f"\n📏 Discrimination & Compression:")
        print(f"  Average discrimination score: {avg_discrimination:.3f} (0-1 scale)")
        print(f"  Average gore compression rate: {avg_compression:.3f} ({avg_compression*100:.1f}%)")
        print(f"  Average symmetric relative difference: {avg_symmetric_diff:.3f} (0-1 scale)")
        print(f"\n💡 Interpretation:")
        if avg_compression > 0:
            print(f"  ✓ Generated summaries reduce gore content by {avg_compression*100:.1f}% on average")
        elif avg_compression < 0:
            print(f"  ✗ Generated summaries increase gore content by {abs(avg_compression)*100:.1f}% on average")
        else:
            print(f"  = Generated summaries maintain similar gore levels")