import json
from typing import List, Dict
import jieba
import jieba.posseg as pseg
from collections import Counter
import numpy as np
from .shallow.feature_1_to_3 import feature_1_to_3
from .shallow.feature_4_to_7 import feature_4_to_7
from .shallow.feature_8_to_18 import feature_8_to_18
from .shallow.feature_19_to_24 import feature_19_to_24
from .pos.feature_25_to_53 import feature_25_to_53
from .pos.feature_54_to_59 import feature_54_to_59
from .pos.feature_60_to_65 import feature_60_to_65
from .syntactic.feature_66_to_78 import feature_66_to_78
from .discourse.feature_79_to_92 import feature_79_to_92
from .discourse.feature_93_to_100 import feature_93_to_100

def extract_all_features(text: str) -> List[Dict[str, float]]:
    """
    Extract all 100 features from the input text.
    
    Args:
        text (str): Input Chinese text
        
    Returns:
        List[Dict[str, float]]: List of dictionaries containing feature values
    """
    # Initialize Jieba
    jieba.initialize()
    
    # Extract features from each category
    features = []
    
    # Shallow features (1-24)
    features.extend(feature_1_to_3(text))
    features.extend(feature_4_to_7(text))
    features.extend(feature_8_to_18(text))
    features.extend(feature_19_to_24(text))
    
    # POS features (25-65)
    features.extend(feature_25_to_53(text))
    features.extend(feature_54_to_59(text))
    features.extend(feature_60_to_65(text))
    
    # Syntactic features (66-78)
    features.extend(feature_66_to_78(text))
    
    # Discourse features (79-100)
    features.extend(feature_79_to_92(text))
    features.extend(feature_93_to_100(text))
    
    return features

def main():
    # Read example text
    with open('/root/mayiran/CLASE/example.txt', 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Extract all features
    features = extract_all_features(text)
    
    # Print features
    for feature in features:
        print(feature)

if __name__ == '__main__':
    main() 