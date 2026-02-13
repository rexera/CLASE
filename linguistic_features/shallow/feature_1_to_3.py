'''Combined features for character frequency analysis
1. Percentage of most-common characters (top 3500)
2. Percentage of second-most-common characters (3500-6500)
3. Percentage of all common characters (top 6500)'''

import json
from collections import Counter
from typing import Dict, List, Set
import os
from pathlib import Path

def load_common_characters(json_path: str, start_idx: int = 0, end_idx: int = 3500) -> Set[str]:
    """Load Chinese characters from character.json file within specified range.
    
    Args:
        json_path: Path to the character.json file
        start_idx: Starting index of characters to load
        end_idx: Ending index of characters to load
        
    Returns:
        Set of characters within the specified range
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        char_data = json.load(f)
        return set(char_data[start_idx:end_idx])

def calculate_character_percentage(text: str, common_chars: Set[str]) -> float:
    """Calculate the percentage of common characters in the text.
    
    Args:
        text: Input text to analyze
        common_chars: Set of common characters to check against
        
    Returns:
        Decimal value between 0 and 1 representing the proportion of common characters
    """
    if not text:
        return 0.0
    
    char_count = Counter(text)
    total_chars = sum(char_count.values())
    common_char_count = sum(count for char, count in char_count.items() 
                          if char in common_chars)
    
    return common_char_count / total_chars if total_chars > 0 else 0.0

def calculate_second_most_common_percentage(text: str, character_set: Set[str]) -> float:
    """Calculate the percentage of second-most-common characters in the text.
    
    Args:
        text: Input text to analyze
        character_set: Set of characters to consider
        
    Returns:
        Percentage of second-most-common characters (0-1)
    """
    filtered_chars = [char for char in text if char in character_set]
    
    if not filtered_chars:
        return 0.0
        
    char_counts = Counter(filtered_chars)
    
    if len(char_counts) < 2:
        return 0.0
        
    second_most_common_count = sorted(char_counts.values(), reverse=True)[1]
    total_chars = len(filtered_chars)
    
    return second_most_common_count / total_chars

def feature_1_to_3(text: str, json_path: str = "/root/mayiran/CLASE/linguistic_features/_resources/characters.json") -> List[Dict[str, float]]:
    """Calculate all three character frequency features.
    
    Args:
        text: Input text to analyze
        json_path: Path to the characters.json file
        
    Returns:
        List of dictionaries containing the three feature values
    """
    # Feature 1: Most common characters (top 3500)
    common_chars_3500 = load_common_characters(json_path, 0, 3500)
    feature1 = calculate_character_percentage(text, common_chars_3500)
    
    # Feature 2: Second most common characters (3500-6500)
    chars_3500_6500 = load_common_characters(json_path, 3500, 6500)
    feature2 = calculate_second_most_common_percentage(text, chars_3500_6500)
    
    # Feature 3: All common characters (top 6500)
    common_chars_6500 = load_common_characters(json_path, 0, 6500)
    feature3 = calculate_character_percentage(text, common_chars_6500)
    
    return [
        {'1': feature1},
        {'2': feature2},
        {'3': feature3}
    ]

def main():
    """Test the feature extraction using example.txt."""
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path for example.txt
    example_path = os.path.join("example.txt")
    
    # Read example.txt
    with open(example_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract features
    results = feature_1_to_3(text)
    for result in results:
        for key, value in result.items():
            print(f"Feature {key}: {value:.4f}")

if __name__ == "__main__":
    main() 