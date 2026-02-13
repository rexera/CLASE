'''93	Percentage of conjunctions per document（文档中连词占比）	篇章特征	计算连词数量占文档总词数的比例
94	Number of unique conjunctions per document（文档中唯一连词数量）	篇章特征	统计文档中不重复的连词数量
95	Percentage of unique conjunctions per document（文档中唯一连词占比）	篇章特征	计算唯一连词数量占总词数的比例
96	Average number of conjunctions per sentence（文档中每句平均连词数量）	篇章特征	统计文档中每句的连词数量，求平均值
97	Average number of unique conjunctions per sentence（文档中每句平均唯一连词数量）	篇章特征	统计文档中每句不重复的连词数量，求平均值
98	Percentage of pronouns per document（文档中代词占比）	篇章特征	计算代词数量占文档总词数的比例
99	Number of unique pronouns per document（文档中唯一代词数量）	篇章特征	统计文档中不重复的代词数量
100	Percentage of unique pronouns per document（文档中唯一代词占比）	篇章特征	计算唯一代词数量占总词数的比例'''

import jieba
import jieba.posseg as pseg
import re
from typing import List, Dict

def feature_93_to_100(text: str) -> List[Dict[str, float]]:
    """
    Calculate features 93-100 for a given text.
    
    Args:
        text: Input text string
        
    Returns:
        List of dictionaries containing feature values
    """
    # Tokenize text into words with POS tags
    words_with_tags = list(pseg.cut(text))
    total_words = len(words_with_tags)
    
    # Split into sentences
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Initialize counters
    conjunctions = []
    unique_conjunctions = set()
    pronouns = []
    unique_pronouns = set()
    
    # Count conjunctions and pronouns using POS tags
    for word, tag in words_with_tags:
        # Conjunctions: c (连词)
        if tag == 'c':
            conjunctions.append(word)
            unique_conjunctions.add(word)
        # Pronouns: r (代词)
        if tag == 'r':
            pronouns.append(word)
            unique_pronouns.add(word)
    
    # Calculate features
    features = []
    
    # Feature 93: Percentage of conjunctions per document
    features.append({'93': len(conjunctions) / total_words if total_words > 0 else 0})
    
    # Feature 94: Number of unique conjunctions per document
    features.append({'94': len(unique_conjunctions)})
    
    # Feature 95: Percentage of unique conjunctions per document
    features.append({'95': len(unique_conjunctions) / total_words if total_words > 0 else 0})
    
    # Feature 96: Average number of conjunctions per sentence
    conj_per_sentence = []
    for sentence in sentences:
        words = list(pseg.cut(sentence))
        conj_count = sum(1 for word, tag in words if tag == 'c')
        conj_per_sentence.append(conj_count)
    features.append({'96': sum(conj_per_sentence) / len(sentences) if sentences else 0})
    
    # Feature 97: Average number of unique conjunctions per sentence
    unique_conj_per_sentence = []
    for sentence in sentences:
        words = list(pseg.cut(sentence))
        unique_conj = set(word for word, tag in words if tag == 'c')
        unique_conj_per_sentence.append(len(unique_conj))
    features.append({'97': sum(unique_conj_per_sentence) / len(sentences) if sentences else 0})
    
    # Feature 98: Percentage of pronouns per document
    features.append({'98': len(pronouns) / total_words if total_words > 0 else 0})
    
    # Feature 99: Number of unique pronouns per document
    features.append({'99': len(unique_pronouns)})
    
    # Feature 100: Percentage of unique pronouns per document
    features.append({'100': len(unique_pronouns) / total_words if total_words > 0 else 0})
    
    return features

def main():
    # Read example text
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Calculate features
    features = feature_93_to_100(text)
    
    # Print results
    for feature in features:
        for key, value in feature.items():
            print(f"Feature {key}: {value:.4f}")

if __name__ == "__main__":
    main()