import re
import pandas as pd

'''
Cleaning the originalScore column of the Kaggle dataset and standardizing the 
wide variety of score evaluation systems. 
'''
def standardize_score(score):
    try: 
        if pd.isna(score):
            return None
        
        score = str(score).strip().lower()
        score = score.strip('"\'')

        # Remove dates (e.g. '2021-01-01', '01/01/2021')
        if re.search(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}' , score):
            return None

        # Normalize leading-dot decimals e.g. '.5/4' -> '0.5/4'
        if re.match(r'^\.\d', score):
            score = '0' + score

        # Handle fractions e.g. '3/4', '2.5/5', '0.5/4', '7/10', '4 of 5'
        fraction = re.search(r'(\d*\.?\d+)\s*(?:/|of)\s*(\d*\.?\d+)', score)
        if fraction:
            numerator, denominator = float(fraction.group(1)), float(fraction.group(2))
            if denominator == 0:
                return None
            return round((numerator / denominator) * 100, 1)
        
        # Handle percentages e.g. '80%'
        percent = re.match(r'(\d+\.?\d*)\s*%', score)
        if percent:
            return float(percent.group(1))
        
        # Handle letter grades e.g. 'A', 'B+', 'C-'
        letter = re.match(r'^([a-f])([+-]?)$', score)
        if letter:
            grade_map = {'a': 95, 'b': 70, 'c': 50, 'd': 30, 'f': 5} # parameters to be tinkered with
            modifier = {'': 0, '+': 5, '-': -5}
            return grade_map[letter.group(1)] + modifier[letter.group(2)]

        # Handle letter grades with word modifiers e.g. 'A minus', 'B plus', 'B-plus', 'C minus'
        letter_word = re.match(r'^([a-f])[- ](plus|minus)$', score)
        if letter_word:
            grade_map = {'a': 95, 'b': 70, 'c': 50, 'd': 30, 'f': 5}
            modifier = {'plus': 5, 'minus': -5}
            return grade_map[letter_word.group(1)] + modifier[letter_word.group(2)]
        
        # Handle text e.g. 'not recommended', 'recommended'
        if 'strongly not recommended' in score:
            return 10.0
        if 'not recommended' in score:
            return 30.0
        if 'recommended' in score:
            return 70.0
        if 'strongly recommended' in score:
            return 90.0
        
        return None
    
    except Exception:
        return None