'''Combined features 4-7 related to character stroke counts
Features:
4. Percentage of low-stroke-count characters (1-5 strokes)
5. Percentage of medium-stroke-count characters (6-15 strokes)
6. Percentage of high-stroke-count characters (16+ strokes)
7. Average number of strokes per character'''

import json
from typing import Dict, List, Tuple

def load_stroke_data(stroke_file: str) -> Dict[str, int]:
    """Load character stroke data from JSON file.
    
    Args:
        stroke_file: Path to the JSON file containing character stroke data
        
    Returns:
        Dictionary mapping characters to their stroke counts
    """
    with open(stroke_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_stroke_ratios(text: str, stroke_data: Dict[str, int]) -> Tuple[float, float, float, float]:
    """Calculate all stroke-related ratios and averages.
    
    Args:
        text: Input text to analyze
        stroke_data: Dictionary mapping characters to their stroke counts
        
    Returns:
        Tuple containing (low_ratio, medium_ratio, high_ratio, avg_strokes)
    """
    total_chars = 0
    low_stroke_chars = 0
    medium_stroke_chars = 0
    high_stroke_chars = 0
    total_strokes = 0
    
    for char in text:
        if char in stroke_data:
            total_chars += 1
            strokes = stroke_data[char]
            total_strokes += strokes
            
            if strokes <= 5:
                low_stroke_chars += 1
            elif 6 <= strokes <= 15:
                medium_stroke_chars += 1
            else:  # strokes >= 16
                high_stroke_chars += 1
    
    if total_chars == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    low_ratio = low_stroke_chars / total_chars
    medium_ratio = medium_stroke_chars / total_chars
    high_ratio = high_stroke_chars / total_chars
    avg_strokes = total_strokes / total_chars
    
    return low_ratio, medium_ratio, high_ratio, avg_strokes

def feature_4_to_7(text: str, stroke_file: str = '/root/mayiran/CLASE/linguistic_features/_resources/char_strokes.json') -> List[Dict[str, float]]:
    """Calculate all stroke-related features (4-7) for the given text.
    
    Args:
        text: Input text to analyze
        stroke_file: Path to the JSON file containing character stroke data
        
    Returns:
        List of dictionaries containing the features:
        [{'4': low_stroke_ratio}, {'5': medium_stroke_ratio}, 
         {'6': high_stroke_ratio}, {'7': avg_strokes}]
    """
    stroke_data = load_stroke_data(stroke_file)
    low_ratio, medium_ratio, high_ratio, avg_strokes = calculate_stroke_ratios(text, stroke_data)
    
    return [
        {'4': low_ratio},
        {'5': medium_ratio},
        {'6': high_ratio},
        {'7': avg_strokes}
    ]

def main():
    """Test the feature extraction with example.txt"""
    # Read example text
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Calculate features
    results = feature_4_to_7(text)
    
    # Print results
    for result in results:
        feature_num = list(result.keys())[0]
        value = result[feature_num]
        print(f"Feature {feature_num}: {value:.4f}")

if __name__ == '__main__':
    main() 