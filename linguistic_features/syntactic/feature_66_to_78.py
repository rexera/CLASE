'''66	Average number of noun phrases per sentence（文档中每句平均名词短语数量）	句法特征	统计文档中每句的名词短语数量，求平均值
67	Average number of verbal phrases per sentence（文档中每句平均动词短语数量）	句法特征	统计文档中每句的动词短语数量，求平均值
68	Total number of noun phrases per document（文档中名词短语的总数）	句法特征	统计文档中所有的名词短语数量
69	Total number of verbal phrases per document（文档中动词短语的总数）	句法特征	统计文档中所有的动词短语数量
70	Total number of prepositional phrases per document（文档中介词短语的总数）	句法特征	统计文档中所有的介词短语数量
71	Average length of noun phrases per document（文档中名词短语的平均长度）	句法特征	统计文档中所有名词短语的长度，计算平均值
72	Average length of verbal phrases per document（文档中动词短语的平均长度）	句法特征	统计文档中所有动词短语的长度，计算平均值
73	Average length of prepositional phrases per document（文档中介词短语的平均长度）	句法特征	统计文档中所有介词短语的长度，计算平均值
74	Average number of sentences with clauses per document（文档中包含从句的句子平均数量）	句法特征	统计文档中含从句的句子数量，计算平均值
75	Percentage of sentences without clauses per document（文档中无从句句子的比例）	句法特征	统计文档中无从句句子占总句子数的比例
76	Average number of clauses per sentence（文档中每句的平均从句数量）	句法特征	统计文档中每句的从句数量，计算平均值
77	Average number of sentences per document（文档中句子的平均数量）	句法特征	统计文档中所有句子的数量，计算平均值
78	Average height of parse tree per document（文档中语法解析树的平均高度）	句法特征	统计文档中每个句子的语法树高度，计算平均值'''

import jieba
import jieba.posseg as pseg
from nltk import Tree
from nltk.parse import CoreNLPParser
from typing import List, Dict
import re

def feature_66_to_78(text: str) -> List[Dict[str, float]]:
    """
    Extract syntactic features 66-78 from Chinese text.
    
    Args:
        text (str): Input Chinese text
        
    Returns:
        List[Dict[str, float]]: List of feature dictionaries, where each dictionary
        contains a single feature value with its corresponding number as key
    """
    # Split text into sentences (simple Chinese sentence splitting)
    sentences = re.split(r'[。！？!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Initialize counters
    total_np_count = 0
    total_vp_count = 0
    total_pp_count = 0
    total_np_length = 0
    total_vp_length = 0
    total_pp_length = 0
    sentences_with_clauses = 0
    total_clauses = 0
    total_tree_height = 0
    
    for sentence in sentences:
        # Use jieba for word segmentation and POS tagging
        words = pseg.cut(sentence)
        
        # Count phrases
        np_count = 0
        vp_count = 0
        pp_count = 0
        np_length = 0
        vp_length = 0
        pp_length = 0
        clause_count = 0
        
        # Simple phrase detection based on POS tags
        current_phrase = []
        current_phrase_type = None
        
        for word, flag in words:
            # Noun phrases (nouns and their modifiers)
            if flag.startswith('n') or flag in ['a', 'b', 'f', 'm', 'q', 'r', 's', 't', 'v', 'z']:
                if current_phrase_type != 'np':
                    if current_phrase:
                        if current_phrase_type == 'np':
                            np_count += 1
                            np_length += len(current_phrase)
                        elif current_phrase_type == 'vp':
                            vp_count += 1
                            vp_length += len(current_phrase)
                        elif current_phrase_type == 'pp':
                            pp_count += 1
                            pp_length += len(current_phrase)
                    current_phrase = []
                current_phrase_type = 'np'
                current_phrase.append(word)
            
            # Verb phrases (verbs and their auxiliaries)
            elif flag.startswith('v'):
                if current_phrase_type != 'vp':
                    if current_phrase:
                        if current_phrase_type == 'np':
                            np_count += 1
                            np_length += len(current_phrase)
                        elif current_phrase_type == 'vp':
                            vp_count += 1
                            vp_length += len(current_phrase)
                        elif current_phrase_type == 'pp':
                            pp_count += 1
                            pp_length += len(current_phrase)
                    current_phrase = []
                current_phrase_type = 'vp'
                current_phrase.append(word)
            
            # Prepositional phrases (prepositions and their objects)
            elif flag == 'p':
                if current_phrase_type != 'pp':
                    if current_phrase:
                        if current_phrase_type == 'np':
                            np_count += 1
                            np_length += len(current_phrase)
                        elif current_phrase_type == 'vp':
                            vp_count += 1
                            vp_length += len(current_phrase)
                        elif current_phrase_type == 'pp':
                            pp_count += 1
                            pp_length += len(current_phrase)
                    current_phrase = []
                current_phrase_type = 'pp'
                current_phrase.append(word)
            
            # Clause detection (based on conjunctions and certain particles)
            elif flag in ['c', 'u']:
                clause_count += 1
            
            else:
                if current_phrase:
                    current_phrase.append(word)
        
        # Handle the last phrase
        if current_phrase:
            if current_phrase_type == 'np':
                np_count += 1
                np_length += len(current_phrase)
            elif current_phrase_type == 'vp':
                vp_count += 1
                vp_length += len(current_phrase)
            elif current_phrase_type == 'pp':
                pp_count += 1
                pp_length += len(current_phrase)
        
        # Update totals
        total_np_count += np_count
        total_vp_count += vp_count
        total_pp_count += pp_count
        total_np_length += np_length
        total_vp_length += vp_length
        total_pp_length += pp_length
        total_clauses += clause_count
        if clause_count > 0:
            sentences_with_clauses += 1
        
        # Simple tree height estimation based on clause count
        total_tree_height += clause_count + 1
    
    # Calculate averages and percentages
    num_sentences = len(sentences)
    
    features = [
        {'66': total_np_count / num_sentences if num_sentences > 0 else 0},  # Average NP per sentence
        {'67': total_vp_count / num_sentences if num_sentences > 0 else 0},  # Average VP per sentence
        {'68': float(total_np_count)},  # Total NP count
        {'69': float(total_vp_count)},  # Total VP count
        {'70': float(total_pp_count)},  # Total PP count
        {'71': total_np_length / total_np_count if total_np_count > 0 else 0},  # Average NP length
        {'72': total_vp_length / total_vp_count if total_vp_count > 0 else 0},  # Average VP length
        {'73': total_pp_length / total_pp_count if total_pp_count > 0 else 0},  # Average PP length
        {'74': sentences_with_clauses / num_sentences if num_sentences > 0 else 0},  # Average sentences with clauses
        {'75': (num_sentences - sentences_with_clauses) / num_sentences if num_sentences > 0 else 0},  # Percentage of sentences without clauses
        {'76': total_clauses / num_sentences if num_sentences > 0 else 0},  # Average clauses per sentence
        {'77': float(num_sentences)},  # Average sentences per document
        {'78': total_tree_height / num_sentences if num_sentences > 0 else 0}  # Average parse tree height
    ]
    
    return features

def main():
    # Read example text
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract features
    features = feature_66_to_78(text)
    
    # Print results
    for feature in features:
        for key, value in feature.items():
            print(f"Feature {key}: {value:.4f}")

if __name__ == "__main__":
    main()
