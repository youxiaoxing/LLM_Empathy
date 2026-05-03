from openai import OpenAI
import json
from tqdm import tqdm


client_openai = OpenAI(
    api_key="-"
)

with open("./dataset/wmt23.json", "r", encoding='utf-8') as f:
    dataList = json.load(f)

def translate_chapter(zh_text, en_reference):
    response = client_openai.chat.completions.create(
            # model="gemini-3-pro-preview",
            # model="gpt-5.2",
            # model="deepseek-v3.2",
            # model="claude-sonnet-4-5-20250929",
            # model="kimi-k2-0905-preview",
            # model="qwen3-vl-235b-a22b-instruct",
            # model="glm-4.7",
            # model="grok-4-1-fast-non-reasoning-latest",
            # model="doubao-seed-1-8-251228",
            model="meta-llama/llama-4-maverick",
        
        messages=[
            {
                "role": "system",
                "content": """You are a professional literary translator specializing in Chinese-to-English translation. 
                Your task is to translate Chinese novels into fluent, natural English."""
            },
            {
                "role": "user",
                "content": f"""Please translate the following Chinese novel chapter into English.

Chinese Text:
{zh_text}

Please provide only the English translation without any explanations or notes."""
            }
        ],
        temperature=0.01
    )
    
    return response.choices[0].message.content


output_file = "./result/wmt23_translation_llama4_results.jsonl"


with open(output_file, "w", encoding='utf-8') as f:
    for item in tqdm(dataList, desc="Translation progress"):
        book_id = item["book_id"]
        chapter_id = item["chapter_id"]
        zh_text = item["zh"]
        en_reference = item["en"]
        
        # 调用翻译函数
        try:
            generated_translation = translate_chapter(zh_text, en_reference)
            
            # 写入结果
            result = {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "source": zh_text,
                "reference": en_reference,
                "generated": generated_translation
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        except Exception as e:
            error_result = {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "error": str(e)
            }
            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

print(f"\nTranslation complete! Results saved to: {output_file}")