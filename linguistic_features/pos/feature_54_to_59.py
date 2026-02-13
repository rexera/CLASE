'''54	Total number of idioms per document（文档中成语的总数）	词性特征	统计文档中所有成语的数量
55	Total number of unique idioms per document（文档中唯一成语的总数）	词性特征	统计文档中不重复的成语数量
56	Percentage of idioms per document（文档中成语占比）	词性特征	计算成语数量占总词数的比例
57	Percentage of unique idioms per document（文档中唯一成语占比）	词性特征	计算文档中唯一成语数量占总词数的比例
58	Average number of idioms per sentence（文档中每句平均成语数量）	词性特征	统计文档中每句的成语数量，求平均值
59	Average number of unique idioms per sentence（文档中每句平均唯一成语数量）	词性特征	统计文档中每句不重复的成语数量，求平均值

从idioms.json中读取成语
'''

import json
import re
from typing import List, Dict

def load_idioms() -> set:
    """Load idioms from idioms.json file."""
    try:
        with open('/root/mayiran/CLASE/linguistic_features/_resources/idioms.json', 'r', encoding='utf-8') as f:
            idioms_list = json.load(f)
            return set(idioms_list)
    except FileNotFoundError:
        print("Warning: idioms.json not found. Using empty set of idioms.")
        return set()
    except json.JSONDecodeError:
        print("Warning: Invalid JSON format in idioms.json. Using empty set of idioms.")
        return set()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using Chinese punctuation."""
    # Split by common Chinese sentence endings
    sentences = re.split('[。！？]', text)
    # Remove empty sentences and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def count_words(text: str) -> int:
    """Count the number of Chinese characters in the text."""
    return len([c for c in text if '\u4e00' <= c <= '\u9fff'])

def find_idioms_in_text(text: str, idioms: set) -> List[str]:
    """Find all idioms in the given text."""
    found_idioms = []
    # Sort idioms by length in descending order to handle overlapping idioms
    sorted_idioms = sorted(idioms, key=len, reverse=True)
    
    for idiom in sorted_idioms:
        # Find all occurrences of the idiom
        start = 0
        while True:
            start = text.find(idiom, start)
            if start == -1:
                break
                
            # Check if it's a complete idiom (not part of a longer word)
            end = start + len(idiom)
            is_complete = True
            
            # Check if the idiom is at the start of text
            if start > 0:
                prev_char = text[start - 1]
                # If previous character is a Chinese character, it's not a complete idiom
                if '\u4e00' <= prev_char <= '\u9fff':
                    is_complete = False
            
            # Check if the idiom is at the end of text
            if end < len(text):
                next_char = text[end]
                # If next character is a Chinese character, it's not a complete idiom
                if '\u4e00' <= next_char <= '\u9fff':
                    is_complete = False
            
            if is_complete:
                found_idioms.append(idiom)
            
            start = end
    
    return found_idioms

def feature_54_to_59(text: str) -> List[Dict[str, float]]:
    """
    Calculate idiom-related features (54-59) for the given text.
    Returns a list of dictionaries containing the feature values.
    """
    # Load idioms
    idioms = load_idioms()
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    num_sentences = len(sentences)
    
    # Count total words
    total_words = count_words(text)
    
    # Initialize counters
    total_idioms = 0
    unique_idioms = set()
    idioms_per_sentence = []
    unique_idioms_per_sentence = []
    
    # Process each sentence
    for sentence in sentences:
        # Find idioms in the sentence
        sentence_idioms = find_idioms_in_text(sentence, idioms)
        sentence_unique_idioms = set(sentence_idioms)
        
        # Update counters
        total_idioms += len(sentence_idioms)
        unique_idioms.update(sentence_unique_idioms)
        idioms_per_sentence.append(len(sentence_idioms))
        unique_idioms_per_sentence.append(len(sentence_unique_idioms))
    
    # Calculate features
    features = [
        {'54': float(total_idioms)},  # Total number of idioms
        {'55': float(len(unique_idioms))},  # Total number of unique idioms
        {'56': float(total_idioms / total_words) if total_words > 0 else 0.0},  # Percentage of idioms
        {'57': float(len(unique_idioms) / total_words) if total_words > 0 else 0.0},  # Percentage of unique idioms
        {'58': float(sum(idioms_per_sentence) / num_sentences) if num_sentences > 0 else 0.0},  # Average idioms per sentence
        {'59': float(sum(unique_idioms_per_sentence) / num_sentences) if num_sentences > 0 else 0.0}  # Average unique idioms per sentence
    ]
    
    return features

def main():
    """Test the feature extraction with example.txt."""
    try:
        with open('example.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        features = feature_54_to_59(text)
        for feature in features:
            print(feature)
            
    except FileNotFoundError:
        print("Error: example.txt not found.")

if __name__ == "__main__":
    main()

