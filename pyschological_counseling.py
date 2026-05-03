from openai import OpenAI
import json
from tqdm import tqdm

client_openai = OpenAI(
    api_key="-"
)

with open("./dataset/data.json", "r") as f:
    dataList = json.load(f)

def gen_sum(article):

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

            messages = [
                {
                    "role": "user",
                    "content": f""" You are a psychological counselor. Below is the client's narrative. Please respond as a professional psychological counselor.
                    
                    {article}
                     """
                }
            ],
            temperature=0.01
        )

    return response.choices[0].message.content.replace("```json", "").replace("```", "")


with open("./result/llama4-maverick.jsonl", "a") as f:
    for item in tqdm(dataList): 
        article = item["text"]

        gen_summary = gen_sum(article)

        f.write(json.dumps({"_id": item["_id"], "text": article, "generate": gen_summary}) + "\n")