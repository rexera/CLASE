'''79	Total number of entities per document（文档中实体的总数）	篇章特征	统计文档中所有命名实体的数量
80	Total number of unique entities per document（文档中唯一实体的总数）	篇章特征	统计文档中不重复的实体数量
81	Percentage of entities per document（文档中实体占比）	篇章特征	计算实体数量占文档所有词数量的比例
82	Percentage of unique entities per document（文档中唯一实体占比）	篇章特征	计算唯一实体数量占文档所有词数量的比例
83	Average number of entities per sentence（文档中每句平均实体数量）	篇章特征	统计文档中每句的实体数量，求平均值
84	Average number of unique entities per sentence（文档中每句平均唯一实体数量）	篇章特征	统计文档中每句不重复的实体数量，求平均值
85	Percentage of named entities per document（文档中命名实体占比）	篇章特征	计算命名实体数量占文档总词数的比例
86	Average number of named entities per sentence（文档中每句平均命名实体数量）	篇章特征	统计文档中每句的命名实体数量，求平均值
87	Percentage of named entities against total entities per document（文档中命名实体占所有实体的比例）	篇章特征	计算命名实体数量占所有实体数量的比例
88	Percentage of nouns per document（文档中名词占比）	篇章特征	计算名词数量占文档所有词数量的比例
89	Percentage of Not-NE nouns per document（文档中非命名实体名词占比）	篇章特征	计算非命名实体名词数量占文档所有词数量的比例
90	Average number of nouns per sentence（文档中每句平均名词数量）	篇章特征	统计文档中每句的名词数量，求平均值
91	Average number of Not-NE nouns per sentence（文档中每句平均非命名实体名词数量）	篇章特征	统计文档中每句的非命名实体名词数量，求平均值
92	Average number of Not-Entity nouns per sentence（文档中每句平均非实体名词数量）	篇章特征	统计文档中每句的非实体名词数量，求平均值'''

import jieba
import jieba.posseg as pseg
from typing import List, Dict

def feature_79_to_92(text: str) -> List[Dict[str, float]]:
    """
    Calculate features 79-92 for a given Chinese text.
    
    Features:
    79. Total number of entities per document
    80. Total number of unique entities per document
    81. Percentage of entities per document
    82. Percentage of unique entities per document
    83. Average number of entities per sentence
    84. Average number of unique entities per sentence
    85. Percentage of named entities per document
    86. Average number of named entities per sentence
    87. Percentage of named entities against total entities per document
    88. Percentage of nouns per document
    89. Percentage of Not-NE nouns per document
    90. Average number of nouns per sentence
    91. Average number of Not-NE nouns per sentence
    92. Average number of Not-Entity nouns per sentence
    
    Args:
        text (str): Input Chinese text
        
    Returns:
        List[Dict[str, float]]: List of dictionaries containing feature values
    """
    # Split text into sentences
    sentences = [s.strip() for s in text.split('。') if s.strip()]
    
    # Initialize counters
    total_words = 0
    total_entities = 0
    unique_entities = set()
    total_named_entities = 0
    total_nouns = 0
    total_not_ne_nouns = 0
    total_not_entity_nouns = 0
    
    # Process each sentence
    for sentence in sentences:
        # Get words and their POS tags
        words = list(pseg.cut(sentence))
        total_words += len(words)
        
        # Count entities and nouns
        sentence_entities = set()
        sentence_named_entities = 0
        sentence_nouns = 0
        sentence_not_ne_nouns = 0
        sentence_not_entity_nouns = 0
        
        for word, flag in words:
            if flag.startswith('nr') or flag.startswith('ns') or flag.startswith('nt'):
                total_entities += 1
                sentence_entities.add(word)
                unique_entities.add(word)
                if flag.startswith('nr'):
                    total_named_entities += 1
                    sentence_named_entities += 1
            elif flag.startswith('n'):
                total_nouns += 1
                sentence_nouns += 1
                # Not-NE nouns are nouns that are not named entities (nr, ns, nt)
                if not (flag.startswith('nr') or flag.startswith('ns') or flag.startswith('nt')):
                    total_not_ne_nouns += 1
                    sentence_not_ne_nouns += 1
                # Not-Entity nouns are nouns that are not any type of entity (nr, ns, nt, nz, nl)
                if not (flag.startswith('nr') or flag.startswith('ns') or flag.startswith('nt') or 
                       flag.startswith('nz') or flag.startswith('nl')):
                    total_not_entity_nouns += 1
                    sentence_not_entity_nouns += 1
    
    # Calculate features
    features = []
    
    # Feature 79: Total number of entities
    features.append({'79': float(total_entities)})
    
    # Feature 80: Total number of unique entities
    features.append({'80': float(len(unique_entities))})
    
    # Feature 81: Percentage of entities
    features.append({'81': float(total_entities) / total_words if total_words > 0 else 0.0})
    
    # Feature 82: Percentage of unique entities
    features.append({'82': float(len(unique_entities)) / total_words if total_words > 0 else 0.0})
    
    # Feature 83: Average number of entities per sentence
    features.append({'83': float(total_entities) / len(sentences) if sentences else 0.0})
    
    # Feature 84: Average number of unique entities per sentence
    features.append({'84': float(len(unique_entities)) / len(sentences) if sentences else 0.0})
    
    # Feature 85: Percentage of named entities
    features.append({'85': float(total_named_entities) / total_words if total_words > 0 else 0.0})
    
    # Feature 86: Average number of named entities per sentence
    features.append({'86': float(total_named_entities) / len(sentences) if sentences else 0.0})
    
    # Feature 87: Percentage of named entities against total entities
    features.append({'87': float(total_named_entities) / total_entities if total_entities > 0 else 0.0})
    
    # Feature 88: Percentage of nouns
    features.append({'88': float(total_nouns) / total_words if total_words > 0 else 0.0})
    
    # Feature 89: Percentage of Not-NE nouns
    features.append({'89': float(total_not_ne_nouns) / total_words if total_words > 0 else 0.0})
    
    # Feature 90: Average number of nouns per sentence
    features.append({'90': float(total_nouns) / len(sentences) if sentences else 0.0})
    
    # Feature 91: Average number of Not-NE nouns per sentence
    features.append({'91': float(total_not_ne_nouns) / len(sentences) if sentences else 0.0})
    
    # Feature 92: Average number of Not-Entity nouns per sentence
    features.append({'92': float(total_not_entity_nouns) / len(sentences) if sentences else 0.0})
    
    return features

def main():
    # Read example text
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Calculate features
    features = feature_79_to_92(text)
    
    # Print results
    for feature in features:
        for key, value in feature.items():
            print(f"Feature {key}: {value:.4f}")

if __name__ == "__main__":
    main()

