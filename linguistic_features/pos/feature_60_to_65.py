'''60	Total number of adverbs per document（文档中副词的总数）	词性特征	统计文档中所有副词的数量
61	Total number of unique adverbs per document（文档中唯一副词的总数）	词性特征	统计文档中不重复的副词数量
62	Percentage of adverbs per document（文档中副词占比）	词性特征	计算文档中副词数量占总词数的比例
63	Percentage of unique adverbs per document（文档中唯一副词占比）	词性特征	计算文档中唯一副词数量占总词数的比例
64	Average number of adverbs per sentence（文档中每句平均副词数量）	词性特征	统计文档中每句的副词数量，求平均值
65	Average number of unique adverbs per sentence（文档中每句平均唯一副词数量）	词性特征	统计文档中每句不重复的副词数量，求平均值'''

import jieba
import jieba.posseg as pseg
from typing import List, Dict

def feature_60_to_65(text: str) -> List[Dict[str, float]]:
    """
    Extract features 60-65 related to adverbs in the text.
    
    Args:
        text: Input text string
        
    Returns:
        List of dictionaries containing feature values
    """
    # Segment text into sentences
    sentences = [s.strip() for s in text.split('。') if s.strip()]
    
    # Get all words and their POS tags
    words_with_pos = pseg.cut(text)
    words = []
    pos_tags = []
    for word, pos in words_with_pos:
        words.append(word)
        pos_tags.append(pos)
    
    # Count total words and adverbs
    total_words = len(words)
    adverbs = [word for word, pos in zip(words, pos_tags) if pos.startswith('d')]
    unique_adverbs = list(set(adverbs))
    total_adverbs = len(adverbs)
    total_unique_adverbs = len(unique_adverbs)
    
    # Calculate adverbs per sentence
    adverbs_per_sentence = []
    unique_adverbs_per_sentence = []
    for sentence in sentences:
        sent_words = list(pseg.cut(sentence))
        sent_adverbs = [word for word, pos in sent_words if pos.startswith('d')]
        sent_unique_adverbs = list(set(sent_adverbs))
        adverbs_per_sentence.append(len(sent_adverbs))
        unique_adverbs_per_sentence.append(len(sent_unique_adverbs))
    
    # Calculate features
    features = [
        {'60': float(total_adverbs)},  # Total number of adverbs
        {'61': float(total_unique_adverbs)},  # Total number of unique adverbs
        {'62': float(total_adverbs / total_words) if total_words > 0 else 0.0},  # Percentage of adverbs
        {'63': float(total_unique_adverbs / total_words) if total_words > 0 else 0.0},  # Percentage of unique adverbs
        {'64': float(sum(adverbs_per_sentence) / len(sentences)) if sentences else 0.0},  # Average adverbs per sentence
        {'65': float(sum(unique_adverbs_per_sentence) / len(sentences)) if sentences else 0.0}  # Average unique adverbs per sentence
    ]
    
    return features

def main():
    # Read test file
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Calculate features
    features = feature_60_to_65(text)
    
    # Print results
    for feature in features:
        for key, value in feature.items():
            print(f"Feature {key}: {value:.4f}")

if __name__ == "__main__":
    main()

