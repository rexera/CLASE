'''8	Average number of characters per word per document （文档中每词平均汉字数）	浅层特征	文档中所有词的总汉字数除以词数
9	Average number of characters per unique word per document （文档中每个唯一词的平均汉字数）	浅层特征	去重后统计所有词的总汉字数，除以唯一词的数量
10	Number of two-character words per document （文档中双字词的数量）	浅层特征	统计文档中所有由两个字符组成的词的数量
11	Number of three-character words per document（文档中三字词的数量）	浅层特征	统计文档中所有由三个字符组成的词的数量
12	Number of four-character words per document（文档中四字词的数量）	浅层特征	统计文档中所有由四个字符组成的词的数量
13	Number of words longer than four characters per document（文档中超过四字的词的数量）	浅层特征	统计文档中所有长度超过四个字符的词的数量
14	Number of unique words per document（文档中唯一词的数量）	浅层特征	统计文档中不重复词的数量
15	Number of unique two-character words per document（文档中唯一双字词的数量）	浅层特征	统计文档中不重复的双字词的数量
16	Number of unique three-character words per document（文档中唯一三字词的数量）	浅层特征	统计文档中不重复的三字词的数量
17	Number of unique four-character words per document（文档中唯一四字词的数量）	浅层特征	统计文档中不重复的四字词的数量
18	Number of unique words longer than four characters per document（文档中唯一超过四字的词的数量）	浅层特征	统计文档中不重复的长度超过四个字符的词的数量'''

import jieba
from typing import List, Dict
import os

def feature_8_to_18(text: str) -> List[Dict[str, float]]:
    """
    Calculate features 8-18 for a given Chinese text.
    
    Args:
        text (str): Input Chinese text
        
    Returns:
        List[Dict[str, float]]: List of dictionaries containing feature values
    """
    # Segment the text using jieba
    words = list(jieba.cut(text))
    
    # Calculate total characters and words
    total_chars = sum(len(word) for word in words)
    total_words = len(words)
    unique_words = set(words)
    unique_words_count = len(unique_words)
    
    # Feature 8: Average characters per word
    feature_8 = total_chars / total_words if total_words > 0 else 0
    
    # Feature 9: Average characters per unique word
    feature_9 = sum(len(word) for word in unique_words) / unique_words_count if unique_words_count > 0 else 0
    
    # Features 10-13: Count words by length
    two_char_words = [w for w in words if len(w) == 2]
    three_char_words = [w for w in words if len(w) == 3]
    four_char_words = [w for w in words if len(w) == 4]
    more_than_four_char_words = [w for w in words if len(w) > 4]
    
    feature_10 = len(two_char_words)
    feature_11 = len(three_char_words)
    feature_12 = len(four_char_words)
    feature_13 = len(more_than_four_char_words)
    
    # Feature 14: Number of unique words
    feature_14 = unique_words_count
    
    # Features 15-18: Count unique words by length
    unique_two_char_words = {w for w in unique_words if len(w) == 2}
    unique_three_char_words = {w for w in unique_words if len(w) == 3}
    unique_four_char_words = {w for w in unique_words if len(w) == 4}
    unique_more_than_four_char_words = {w for w in unique_words if len(w) > 4}
    
    feature_15 = len(unique_two_char_words)
    feature_16 = len(unique_three_char_words)
    feature_17 = len(unique_four_char_words)
    feature_18 = len(unique_more_than_four_char_words)
    
    return [
        {'8': feature_8},
        {'9': feature_9},
        {'10': feature_10},
        {'11': feature_11},
        {'12': feature_12},
        {'13': feature_13},
        {'14': feature_14},
        {'15': feature_15},
        {'16': feature_16},
        {'17': feature_17},
        {'18': feature_18}
    ]

def main():
    # Read example.txt from the root directory
    example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'example.txt')
    with open(example_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Calculate features
    features = feature_8_to_18(text)
    
    # Print results
    for feature in features:
        for feature_id, value in feature.items():
            print(f"Feature {feature_id}: {value:.4f}")

if __name__ == "__main__":
    main()