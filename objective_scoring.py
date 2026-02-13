import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from linguistic_features.feature_extractor import extract_all_features
from tqdm import tqdm
from multiprocessing import Pool

def load_experiences(file_path, N=None):
    experiences = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if N is not None and i >= N:
                break
            exp = json.loads(line.strip())
            for pair in exp.get('pair', []):
                experiences.append(pair)
    return experiences

def dict_features_to_array(features_list):
    feature_names = set()
    for features in features_list:
        for feature_dict in features:
            feature_names.update(feature_dict.keys())
    feature_names = sorted(list(feature_names))
    
    X = np.zeros((len(features_list), len(feature_names)))
    
    for i, features in enumerate(features_list):
        flat_dict = {}
        for feature_dict in features:
            flat_dict.update(feature_dict)
        for j, name in enumerate(feature_names):
            X[i, j] = flat_dict.get(name, 0.0)
    
    return X, feature_names

def process(input_jsonl, output_dir, exp_library, N, k):
    experiences = load_experiences(exp_library, N)
    
    positives = [pair['positive'] for pair in experiences]
    negatives = [pair['negative'] for pair in experiences]
    
    all_texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    
    all_features = []
    for text in tqdm(all_texts, desc="Extracting features"):
        features = extract_all_features(text)
        all_features.append(features)
    
    X, feature_names = dict_features_to_array(all_features)
    y = np.array(labels)
    
    lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    lr.fit(X, y)
    
    coef_abs = np.abs(lr.coef_[0])
    top_indices = np.argsort(coef_abs)[::-1][:k]
    top_features = [feature_names[i] for i in top_indices]
    X_top = X[:, top_indices]
    
    lr_top = LogisticRegression(penalty=None, solver='lbfgs', random_state=42)
    lr_top.fit(X_top, y)
    
    model_data = {
        "feature_names": top_features,
        "coefficients": lr_top.coef_[0].tolist(),
        "intercept": lr_top.intercept_[0]
    }
    model_file = os.path.join(output_dir, f"model_N{N}_k{k}.json")
    with open(model_file, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)
    
    output_file = os.path.join(output_dir, f"scores_N{N}_k{k}.jsonl")
    with open(input_jsonl, 'r') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            generated = data['generated']
            index = data['index']
            
            features = extract_all_features(generated)
            flat_dict = {}
            for d in features:
                flat_dict.update(d)
            
            X_new = np.zeros((1, k))
            for j, name in enumerate(top_features):
                X_new[0, j] = flat_dict.get(name, 0.0)
            
            linear_pred = lr_top.intercept_[0] + np.dot(lr_top.coef_[0], X_new[0])
            reg_score = 1 / (1 + np.exp(-linear_pred))
            
            result = {"index": index, "reg_score": reg_score}
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_jsonl = "data/test_samples.jsonl"
    output_dir = "output/objective_scores"
    exp_library = "data/examples.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    steps_N = [100, 500, 1000, 2000, 4000]
    steps_k = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    def run_ablation(args):
        N, k = args
        process(input_jsonl, output_dir, exp_library, N, k)
    
    combinations = [(N, k) for N in steps_N for k in steps_k]
    with Pool(processes=5) as pool:
        pool.map(run_ablation, combinations) 