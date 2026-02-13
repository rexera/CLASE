import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from feature_extractor import extract_all_features
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import hashlib

def get_file_hash(file_path):
    """Calculate hash of file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        return hashlib.md5(content.encode()).hexdigest()

def process_single_file(file_path, label):
    # Check cache first
    cache_dir = 'feature_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(cache_dir, f"{file_hash}.json")
    
    try:
        if os.path.exists(cache_file):
            print(f"Loading cached features for {file_path}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
        else:
            print(f"Computing features for {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                features = extract_all_features(text)
            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # If there's an error with cache, compute features again
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            features = extract_all_features(text)
    
    return features, label

def dict_features_to_array(features_list):
    """Convert list of feature dictionaries to numpy array"""
    # Get all unique feature names
    feature_names = set()
    for features in features_list:
        for feature_dict in features:
            feature_names.update(feature_dict.keys())
    feature_names = sorted(list(feature_names))
    
    # Create numpy array
    X = np.zeros((len(features_list), len(feature_names)))
    
    # Fill in values
    for i, features in enumerate(features_list):
        for feature_dict in features:
            for j, name in enumerate(feature_names):
                if name in feature_dict:
                    X[i, j] = feature_dict[name]
    
    return X, feature_names

def process_text_files():
    # Initialize lists to store features and labels
    all_features = []
    labels = []
    
    # Process gist files (label 0)
    gist_dir = 'gist_txts'
    gist_files = [os.path.join(gist_dir, f) for f in os.listdir(gist_dir) if f.endswith('.txt')]
    
    # Process reason files (label 1)
    reason_dir = 'reason_txts'
    reason_files = [os.path.join(reason_dir, f) for f in os.listdir(reason_dir) if f.endswith('.txt')]
    
    # Combine all files and their labels
    all_files = gist_files + reason_files
    all_labels = [0] * len(gist_files) + [1] * len(reason_files)
    
    print(f"\nProcessing {len(all_files)} files with 10 parallel workers...")
    
    # Create a pool of 10 workers
    num_workers = min(10, cpu_count())
    
    # Process files in chunks to show progress
    chunk_size = 100  # Process 100 files at a time
    results = []
    
    for i in tqdm(range(0, len(all_files), chunk_size), desc="Processing chunks"):
        chunk_files = all_files[i:i + chunk_size]
        chunk_labels = all_labels[i:i + chunk_size]
        
        with Pool(num_workers) as pool:
            chunk_results = pool.starmap(process_single_file, zip(chunk_files, chunk_labels))
            results.extend(chunk_results)
    
    # Unpack results
    all_features, labels = zip(*results)
    
    return list(all_features), list(labels)

def analyze_features():
    print("\nExtracting features from text files...")
    # Get features and labels
    all_features, labels = process_text_files()
    
    print("\nConverting features to numpy array...")
    # Convert features to numpy array
    X, feature_names = dict_features_to_array(all_features)
    y = np.array(labels)
    
    print("\nPerforming logistic regression analysis...")
    # Create and fit logistic regression model
    lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    lr.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr.coef_[0],
        'Absolute_Coefficient': np.abs(lr.coef_[0])
    })
    
    # Sort by absolute coefficient value
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)
    
    # Select significant features (non-zero coefficients)
    significant_features = feature_importance[feature_importance['Coefficient'] != 0]
    
    return significant_features

if __name__ == '__main__':
    significant_features = analyze_features()
    print("\nSignificant Features:")
    print(significant_features)
    
    # Save results to CSV
    significant_features.to_csv('significant_features.csv', index=False)
    print("\nResults saved to 'significant_features.csv'") 