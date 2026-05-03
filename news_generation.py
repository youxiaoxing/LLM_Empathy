from openai import OpenAI
import json
from tqdm import tqdm


client_openai = OpenAI(
    api_key="-"
)


with open("./dataset/newsroom.json", "r") as f:
    dataList = json.load(f)

def gen_article(summary, reference_article):
    length = len(reference_article.split(" "))
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
            # model="meta-llama/llama-4-maverick",
            model="qwen/qwen3-max",

            messages = [
                {
                    "role": "user",
                    "content": f"""As a seasoned news editor, please help me expand the following news summary into a full news article, which should contain approximately {length} words.
                    
                    News Summary:
                    {summary}
                     """
                }
            ],
            temperature=0.01
        )

    return response.choices[0].message.content


with open("./result/article_generation.jsonl", "a") as f:
    for item in tqdm(dataList): 
        summary = item["summary"]
        reference_article = item["text"]

        generated_article = gen_article(summary, reference_article)

        f.write(json.dumps({"_id": item["_id"], "reference": reference_article, "generate": generated_article}) + "\n")