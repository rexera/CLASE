'''19	Number of sentences per document（文档中的句子数量）	浅层特征	统计文档中的句子数量
20	Average number of words per sentence per document（文档中每句平均词数）	浅层特征	计算文档中每个句子的词数，求平均值
21	Average number of characters per sentence per document（文档中每句平均汉字数）	浅层特征	统计文档中每个句子的汉字数，求平均值
22	Average number of punctuation marks per sentence per document（文档中每句平均标点符号数）	浅层特征	统计文档中每个句子的标点符号数量，求平均值
23	Total number of characters per document（文档中的汉字总数）	浅层特征	统计文档中所有汉字的数量
24	Total number of punctuation marks per document（文档中的标点符号总数）	浅层特征	统计文档中所有标点符号的数量'''

import re

def split_chinese_sentences(text):
    """
    Split Chinese text into sentences based on punctuation marks.
    """
    # Define sentence-ending punctuation marks in Chinese
    sentence_endings = r'[。！？]'
    
    # Split text into sentences
    sentences = re.split(sentence_endings, text)
    
    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def feature_19_to_24(text):
    """
    Extract features 19-24 from the input text.
    
    Args:
        text (str): Input text document
        
    Returns:
        list: List of dictionaries containing feature values
    """
    # Split text into sentences
    sentences = split_chinese_sentences(text)
    
    # Calculate features
    num_sentences = len(sentences)
    
    # Initialize counters
    total_words = 0
    total_chars = 0
    total_punctuation = 0
    sentence_word_counts = []
    sentence_char_counts = []
    sentence_punctuation_counts = []
    
    # Process each sentence
    for sentence in sentences:
        # Count words (split by whitespace)
        words = sentence.split()
        word_count = len(words)
        sentence_word_counts.append(word_count)
        total_words += word_count
        
        # Count Chinese characters
        char_count = len(re.findall(r'[\u4e00-\u9fff]', sentence))
        sentence_char_counts.append(char_count)
        total_chars += char_count
        
        # Count punctuation marks
        punctuation_count = len(re.findall(r'[，。！？；：、]', sentence))
        sentence_punctuation_counts.append(punctuation_count)
        total_punctuation += punctuation_count
    
    # Calculate averages
    avg_words_per_sentence = total_words / num_sentences if num_sentences > 0 else 0
    avg_chars_per_sentence = total_chars / num_sentences if num_sentences > 0 else 0
    avg_punctuation_per_sentence = total_punctuation / num_sentences if num_sentences > 0 else 0
    
    # Create feature dictionary
    features = [
        {'19': num_sentences},
        {'20': avg_words_per_sentence},
        {'21': avg_chars_per_sentence},
        {'22': avg_punctuation_per_sentence},
        {'23': total_chars},
        {'24': total_punctuation}
    ]
    
    return features

def main():
    # Read example.txt
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract features
    features = feature_19_to_24(text)
    
    # Print results
    for feature in features:
        for key, value in feature.items():
            print(f"Feature {key}: {value:.2f}")

if __name__ == "__main__":
    main()

