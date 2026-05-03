from openai import OpenAI
import numpy as np

client = OpenAI(
    base_url="-",
    api_key="-"
)

def calculate_similarity(text1, text2):
    response = client.embeddings.create(
        model="Qwen3-Embedding-4B",
        input=[text1, text2]
    )
    
    emb1 = np.array(response.data[0].embedding)
    emb2 = np.array(response.data[1].embedding)
    
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    return similarity

import json

total_similarity = 0
file_path = "./qwen_max_article_generation.jsonl"
print(file_path)
with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)
        similarity = calculate_similarity(data["reference"], data["generate"])
        # print(similarity)
        total_similarity += similarity

print(total_similarity / 50)