import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def compute_embedding(text, client, model_name):
    response = client.embeddings.create(input=[text], model=model_name)
    return np.array(response.data[0].embedding)

def load_experiences(file_path, N=None, client=None, embedding_model=None):
    experiences = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if N is not None and i >= N:
                break
            exp = json.loads(line.strip())
            for pair in exp.get('pair', []):
                experiences.append(pair)
    
    for exp in experiences:
        exp['neg_embedding'] = compute_embedding(exp['negative'], client, embedding_model)
    return experiences

def construct_queries(generated, x, model_name, client):
    prompt = f"Generate {x} concise queries to extract potential errors in legal language style (word choice and sentence structure) from a negative example database. Point out specific problematic words and sentences.\n\n{generated}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    queries = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()][:x]
    return queries

def find_top_pairs(query, experiences, y, client, embedding_model):
    q_emb = compute_embedding(query, client, embedding_model)
    sims = [cosine_similarity([q_emb], [exp['neg_embedding']])[0][0] for exp in experiences]
    top_indices = np.argsort(sims)[-y:][::-1]
    top_pairs = [(experiences[i]['positive'], experiences[i]['negative']) for i in top_indices]
    return top_pairs

def score_generated(generated, pairs, model_name, aspects, client):
    pair_str = "\n".join([f"负面示例: {neg}\n正面示例: {pos}\n" for pos, neg in pairs])
    aspect_results = {}
    for aspect, info in aspects.items():
        asp_name = info['name']
        asp_desc = info['desc']
        prompt = f"""使用提供的法律语言风格的正面和负面示例作为参考，对以下生成的文本在{asp_name}方面进行评估，从0到10分（打分务必极其严格，尽可能多地找出模型的缺陷，体现法律文书的严谨性和模型表现差距，不能全都打7分和8分，必要时可以勇敢打低分。）。{asp_desc}。

示例：
{pair_str}

生成的文本：
{generated}

严格输出为JSON：{{"score": 分数, "reason": "理由"}} 而不添加任何额外文本。
理由部分请精确到具体的词语和句子使用缺陷，不要笼统的描述，根据你识别出的缺陷个数确定分数。
评分量表（0–10 分）：
10 分：无缺陷。
7-9 分：存在 1-2 处缺陷。
4-6 分：存在 3-4 处缺陷。
3-5 分：存在 4-5 处缺陷。
0-2 分：存在 6 处及以上缺陷。
格式大致如：缺陷1:模型表现为……实际文书中……；缺陷2:……；缺陷3:……；缺陷4:……；打分：…分
"""
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        output = response.choices[0].message.content
        try:
            result = json.loads(output)
            aspect_results[aspect] = {"score": result['score'], "reason": result['reason']}
        except (json.JSONDecodeError, KeyError):
            import re
            score_match = re.search(r'"score":\s*(\d+)', output)
            reason_match = re.search(r'"reason":\s*"(.*?)"', output, re.DOTALL)
            score = int(score_match.group(1)) if score_match else 0
            reason = reason_match.group(1) if reason_match else ""
            aspect_results[aspect] = {"score": score, "reason": reason}
    return aspect_results

def process(input_jsonl, output_dir, exp_library, generation_model, embedding_model, x, y, N, generation_client, embedding_client):
    experiences = load_experiences(exp_library, N, embedding_client, embedding_model)
    aspects = {
        "noun": {"name": "名词", "desc": "评估名词的使用是否准确、专业且符合法律文体"},
        "verb": {"name": "动词", "desc": "评估动词的选择是否精确、正式，避免口语化表达"},
        "adjective": {"name": "形容词", "desc": "评估形容词的使用是否适度、中立，避免主观情感色彩"},
        "small_words": {"name": "小词（数词、量词、代词、副词、介词、助词）", "desc": "评估这些功能词是否规范、简洁，避免冗余或不当使用"},
        "sentence_coherence": {"name": "句子衔接连贯", "desc": "评估句子间逻辑连接是否顺畅、条理清晰"},
        "sentence_structure": {"name": "句子结构", "desc": "评估句子结构是否复杂适当、符合法律文书的规范"},
        "intra_sentence_collocation": {"name": "句内搭配", "desc": "评估句内词语搭配是否自然、准确，避免搭配错误"}
    }
    output_file = os.path.join(output_dir, f"scores_x{x}_y{y}_N{N}.jsonl")
    with open(input_jsonl, 'r') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            generated = data['generated']
            queries = construct_queries(generated, x, generation_model, generation_client)
            all_pairs = []
            for q in queries:
                all_pairs.extend(find_top_pairs(q, experiences, y, embedding_client, embedding_model))
            all_pairs = list(set(all_pairs))
            aspect_results = score_generated(generated, all_pairs, generation_model, aspects, generation_client)
            result = {"index": data['index'], "aspects": aspect_results}
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    generation_client = OpenAI(
        base_url=os.getenv('GENERATION_BASE_URL'),
        api_key=os.getenv('GENERATION_API_KEY'),
    )
    embedding_client = OpenAI(
        base_url=os.getenv('EMBEDDING_BASE_URL'),
        api_key=os.getenv('EMBEDDING_API_KEY'),
    )
    embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    generation_model = os.getenv('GENERATION_MODEL', 'gpt-4o-mini')
    input_jsonl = "data/test_samples.jsonl"
    output_dir = "output/subjective_scores"
    exp_library = "data/examples.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    ablations = [(5,5), (5,10), (10,5), (10,10)]
    steps = [100, 500, 1000, 2000, 4000]
    from multiprocessing import Pool
    def run_ablation(args):
        x, y, N = args
        process(input_jsonl, output_dir, exp_library, generation_model, embedding_model, x, y, N, generation_client, embedding_client)
    combinations = [(x, y, N) for x, y in ablations for N in steps]
    with Pool(processes=5) as pool:
        pool.map(run_ablation, combinations) 