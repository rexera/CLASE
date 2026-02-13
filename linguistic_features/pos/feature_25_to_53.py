'''25	Average number of functional words per sentence（文档中每句平均功能词数量）	词性特征	统计文档中每句的功能词数量，求平均值
26	Average number of unique functional words per sentence（文档中每句平均唯一功能词数量）	词性特征	统计文档中每句不重复的功能词数量，求平均值
27	Percentage of functional words per document（文档中功能词占比）	词性特征	计算功能词数量占文档所有词数量的比例
28	Percentage of unique functional words per document（文档中唯一功能词占比）	词性特征	计算文档中唯一功能词数量占所有词数量的比例
29	Total number of unique functional words per document（文档中唯一功能词的总数）	词性特征	统计文档中所有不重复的功能词数量
30	Total number of adjectives per document（文档中形容词的总数）	词性特征	统计文档中所有形容词的数量
31	Total number of unique adjectives per document（文档中唯一形容词的总数）	词性特征	统计文档中不重复的形容词数量
32	Percentage of adjectives per document（文档中形容词占比）	词性特征	计算文档中形容词数量占总词数的比例
33	Percentage of unique adjectives per document（文档中唯一形容词占比）	词性特征	计算文档中唯一形容词数量占总词数的比例
34	Average number of adjectives per sentence（文档中每句平均形容词数量）	词性特征	统计文档中每句的形容词数量，求平均值
35	Average number of unique adjectives per sentence（文档中每句平均唯一形容词数量）	词性特征	统计文档中每句不重复的形容词数量，求平均值
36	Total number of verbs per document（文档中动词的总数）	词性特征	统计文档中所有动词的数量
37	Total number of unique verbs per document（文档中唯一动词的总数）	词性特征	统计文档中不重复的动词数量
38	Percentage of verbs per document（文档中动词占比）	词性特征	计算文档中动词数量占总词数的比例
39	Percentage of unique verbs per document（文档中唯一动词占比）	词性特征	计算文档中唯一动词数量占总词数的比例
40	Average number of verbs per sentence（文档中每句平均动词数量）	词性特征	统计文档中每句的动词数量，求平均值
41	Average number of unique verbs per sentence（文档中每句平均唯一动词数量）	词性特征	统计文档中每句不重复的动词数量，求平均值
42	Total number of nouns per document（文档中名词的总数）	词性特征	统计文档中所有名词的数量
43	Total number of unique nouns per document（文档中唯一名词的总数）	词性特征	统计文档中不重复的名词数量
44	Percentage of nouns per document（文档中名词占比）	词性特征	计算文档中名词数量占总词数的比例
45	Percentage of unique nouns per document（文档中唯一名词占比）	词性特征	计算文档中唯一名词数量占总词数的比例
46	Average number of nouns per sentence（文档中每句平均名词数量）	词性特征	统计文档中每句的名词数量，求平均值
47	Average number of unique nouns per sentence（文档中每句平均唯一名词数量）	词性特征	统计文档中每句不重复的名词数量，求平均值
48	Total number of content words per document（文档中实词的总数）	词性特征	统计文档中所有实词的数量
49	Total number of unique content words per document（文档中唯一实词的总数）	词性特征	统计文档中不重复的实词数量
50	Percentage of content words per document（文档中实词占比）	词性特征	计算文档中实词数量占总词数的比例
51	Percentage of unique content words per document（文档中唯一实词占比）	词性特征	计算文档中唯一实词数量占总词数的比例
52	Average number of content words per sentence（文档中每句平均实词数量）	词性特征	统计文档中每句的实词数量，求平均值
53	Average number of unique content words per sentence（文档中每句平均唯一实词数量）	词性特征	统计文档中每句不重复的实词数量，求平均值'''

import jieba
import jieba.posseg as pseg
from typing import List, Dict, Union
import re

def split_into_sentences(text: str) -> List[str]:
    """Split Chinese text into sentences."""
    # Common Chinese sentence endings
    sentence_endings = r'[。！？]'
    sentences = re.split(f'({sentence_endings})', text)
    # Combine the sentence with its ending
    return [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]

def get_word_pos(text: str) -> List[tuple]:
    """Get words and their POS tags from text."""
    words_with_pos = pseg.cut(text)
    return [(word, pos) for word, pos in words_with_pos]

def feature_25_to_53(text: str) -> List[Dict[str, float]]:
    """
    Calculate features 25-53 from the input text.
    Returns a list of dictionaries containing feature values.
    """
    # Split text into sentences
    sentences = split_into_sentences(text)
    
    # Get all words and their POS tags
    all_words_pos = get_word_pos(text)
    
    # Initialize counters
    total_words = len(all_words_pos)
    total_sentences = len(sentences)
    
    # Define POS categories
    functional_words = {'d', 'p', 'c', 'u', 'y', 'e', 'o', 'h', 'k', 'x', 'w', 'PERIOD', 'w'}
    adjectives = {'a', 'an', 'Ag', 'a', 'an', 'Ag'}
    verbs = {'v', 'vd', 'vn', 'v', 'vd', 'vn', 'Vg'}
    nouns = {'n', 'nr', 'ns', 'nt', 'nz', 'n', 'nr', 'ns', 'nt', 'nz', 'Ng'}
    content_words = set(adjectives) | set(verbs) | set(nouns)
    
    # Calculate document-level statistics
    doc_stats = {
        'functional': {'total': 0, 'unique': set()},
        'adjectives': {'total': 0, 'unique': set()},
        'verbs': {'total': 0, 'unique': set()},
        'nouns': {'total': 0, 'unique': set()},
        'content': {'total': 0, 'unique': set()}
    }
    
    # Calculate sentence-level statistics
    sentence_stats = {
        'functional': [],
        'adjectives': [],
        'verbs': [],
        'nouns': [],
        'content': []
    }
    
    # Process each sentence
    for sentence in sentences:
        sent_words_pos = get_word_pos(sentence)
        sent_stats = {
            'functional': {'total': 0, 'unique': set()},
            'adjectives': {'total': 0, 'unique': set()},
            'verbs': {'total': 0, 'unique': set()},
            'nouns': {'total': 0, 'unique': set()},
            'content': {'total': 0, 'unique': set()}
        }
        
        for word, pos in sent_words_pos:
            # Update document statistics
            if pos in functional_words:
                doc_stats['functional']['total'] += 1
                doc_stats['functional']['unique'].add(word)
                sent_stats['functional']['total'] += 1
                sent_stats['functional']['unique'].add(word)
            if pos in adjectives:
                doc_stats['adjectives']['total'] += 1
                doc_stats['adjectives']['unique'].add(word)
                sent_stats['adjectives']['total'] += 1
                sent_stats['adjectives']['unique'].add(word)
            if pos in verbs:
                doc_stats['verbs']['total'] += 1
                doc_stats['verbs']['unique'].add(word)
                sent_stats['verbs']['total'] += 1
                sent_stats['verbs']['unique'].add(word)
            if pos in nouns:
                doc_stats['nouns']['total'] += 1
                doc_stats['nouns']['unique'].add(word)
                sent_stats['nouns']['total'] += 1
                sent_stats['nouns']['unique'].add(word)
            if pos in content_words:
                doc_stats['content']['total'] += 1
                doc_stats['content']['unique'].add(word)
                sent_stats['content']['total'] += 1
                sent_stats['content']['unique'].add(word)
        
        # Store sentence statistics
        for category in sentence_stats:
            sentence_stats[category].append({
                'total': sent_stats[category]['total'],
                'unique': len(sent_stats[category]['unique'])
            })
    
    # Calculate features
    features = []
    
    # Feature 25: Average number of functional words per sentence
    features.append({'25': sum(s['total'] for s in sentence_stats['functional']) / total_sentences})
    
    # Feature 26: Average number of unique functional words per sentence
    features.append({'26': sum(s['unique'] for s in sentence_stats['functional']) / total_sentences})
    
    # Feature 27: Percentage of functional words per document
    features.append({'27': doc_stats['functional']['total'] / total_words})
    
    # Feature 28: Percentage of unique functional words per document
    features.append({'28': len(doc_stats['functional']['unique']) / total_words})
    
    # Feature 29: Total number of unique functional words per document
    features.append({'29': len(doc_stats['functional']['unique'])})
    
    # Feature 30: Total number of adjectives per document
    features.append({'30': doc_stats['adjectives']['total']})
    
    # Feature 31: Total number of unique adjectives per document
    features.append({'31': len(doc_stats['adjectives']['unique'])})
    
    # Feature 32: Percentage of adjectives per document
    features.append({'32': doc_stats['adjectives']['total'] / total_words})
    
    # Feature 33: Percentage of unique adjectives per document
    features.append({'33': len(doc_stats['adjectives']['unique']) / total_words})
    
    # Feature 34: Average number of adjectives per sentence
    features.append({'34': sum(s['total'] for s in sentence_stats['adjectives']) / total_sentences})
    
    # Feature 35: Average number of unique adjectives per sentence
    features.append({'35': sum(s['unique'] for s in sentence_stats['adjectives']) / total_sentences})
    
    # Feature 36: Total number of verbs per document
    features.append({'36': doc_stats['verbs']['total']})
    
    # Feature 37: Total number of unique verbs per document
    features.append({'37': len(doc_stats['verbs']['unique'])})
    
    # Feature 38: Percentage of verbs per document
    features.append({'38': doc_stats['verbs']['total'] / total_words})
    
    # Feature 39: Percentage of unique verbs per document
    features.append({'39': len(doc_stats['verbs']['unique']) / total_words})
    
    # Feature 40: Average number of verbs per sentence
    features.append({'40': sum(s['total'] for s in sentence_stats['verbs']) / total_sentences})
    
    # Feature 41: Average number of unique verbs per sentence
    features.append({'41': sum(s['unique'] for s in sentence_stats['verbs']) / total_sentences})
    
    # Feature 42: Total number of nouns per document
    features.append({'42': doc_stats['nouns']['total']})
    
    # Feature 43: Total number of unique nouns per document
    features.append({'43': len(doc_stats['nouns']['unique'])})
    
    # Feature 44: Percentage of nouns per document
    features.append({'44': doc_stats['nouns']['total'] / total_words})
    
    # Feature 45: Percentage of unique nouns per document
    features.append({'45': len(doc_stats['nouns']['unique']) / total_words})
    
    # Feature 46: Average number of nouns per sentence
    features.append({'46': sum(s['total'] for s in sentence_stats['nouns']) / total_sentences})
    
    # Feature 47: Average number of unique nouns per sentence
    features.append({'47': sum(s['unique'] for s in sentence_stats['nouns']) / total_sentences})
    
    # Feature 48: Total number of content words per document
    features.append({'48': doc_stats['content']['total']})
    
    # Feature 49: Total number of unique content words per document
    features.append({'49': len(doc_stats['content']['unique'])})
    
    # Feature 50: Percentage of content words per document
    features.append({'50': doc_stats['content']['total'] / total_words})
    
    # Feature 51: Percentage of unique content words per document
    features.append({'51': len(doc_stats['content']['unique']) / total_words})
    
    # Feature 52: Average number of content words per sentence
    features.append({'52': sum(s['total'] for s in sentence_stats['content']) / total_sentences})
    
    # Feature 53: Average number of unique content words per sentence
    features.append({'53': sum(s['unique'] for s in sentence_stats['content']) / total_sentences})
    
    return features

def main():
    """Test the feature extraction with example.txt"""
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    features = feature_25_to_53(text)
    for feature in features:
        print(feature)

if __name__ == '__main__':
    main()