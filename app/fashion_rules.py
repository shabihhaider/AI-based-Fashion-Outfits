# fashion_rules.py - Advanced Fashion Rule Framework
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class FashionItem:
    name: str
    category: str
    formality: str
    style_category: str
    season_tags: List[str]
    fabric_type: str
    color_family: str

class AdvancedFormalityClassifier:
    """PhD-level formality classification with comprehensive rule engine"""
    
    def __init__(self):
        # Comprehensive keyword dictionaries based on fashion theory
        self.formality_rules = {
            'formal': {
                'tops': [
                    'dress shirt', 'oxford shirt', 'button-down', 'dress blouse',
                    'silk blouse', 'tailored shirt', 'formal shirt', 'business shirt',
                    'blazer', 'suit jacket', 'tuxedo jacket', 'sport coat',
                    'formal sweater', 'cashmere sweater', 'merino wool'
                ],
                'bottoms': [
                    'dress pants', 'suit pants', 'trousers', 'slacks', 'chinos',
                    'tailored pants', 'wool pants', 'formal skirt', 'pencil skirt',
                    'A-line skirt', 'midi skirt', 'dress shorts'
                ],
                'footwear': [
                    'oxford shoes', 'derby shoes', 'loafers', 'monk strap',
                    'dress boots', 'chelsea boots', 'pumps', 'heels',
                    'pointed toe', 'leather shoes', 'dress flats'
                ],
                'outerwear': [
                    'blazer', 'suit jacket', 'overcoat', 'trench coat',
                    'wool coat', 'cashmere coat', 'pea coat', 'formal vest'
                ]
            },
            'casual': {
                'tops': [
                    't-shirt', 'tee', 'polo shirt', 'henley', 'tank top',
                    'graphic tee', 'hoodie', 'sweatshirt', 'casual sweater',
                    'cardigan', 'flannel', 'denim shirt'
                ],
                'bottoms': [
                    'jeans', 'denim', 'shorts', 'cargo pants', 'sweatpants',
                    'joggers', 'leggings', 'yoga pants', 'casual skirt',
                    'mini skirt', 'denim skirt', 'athletic shorts'
                ],
                'footwear': [
                    'sneakers', 'running shoes', 'canvas shoes', 'sandals',
                    'flip-flops', 'athletic shoes', 'casual boots', 'hiking boots',
                    'slip-on shoes', 'boat shoes'
                ],
                'outerwear': [
                    'hoodie', 'zip-up hoodie', 'bomber jacket', 'denim jacket',
                    'windbreaker', 'track jacket', 'casual vest', 'fleece jacket'
                ]
            },
            'smart_casual': {
                'tops': [
                    'polo shirt', 'button-up shirt', 'knit shirt', 'sweater',
                    'cardigan', 'blouse', 'nice t-shirt', 'long-sleeve tee'
                ],
                'bottoms': [
                    'chinos', 'khakis', 'dark jeans', 'casual pants',
                    'smart shorts', 'tailored shorts', 'casual skirt'
                ],
                'footwear': [
                    'loafers', 'boat shoes', 'clean sneakers', 'casual boots',
                    'desert boots', 'canvas shoes', 'ballet flats'
                ],
                'outerwear': [
                    'cardigan', 'light jacket', 'casual blazer', 'zip-up sweater'
                ]
            }
        }
        
        # Fabric-based formality indicators
        self.fabric_formality = {
            'formal': ['wool', 'cashmere', 'silk', 'linen', 'cotton poplin', 'twill'],
            'casual': ['jersey', 'fleece', 'terry', 'denim', 'canvas'],
            'athletic': ['polyester', 'spandex', 'moisture-wicking', 'athletic mesh']
        }
        
        # Style indicators
        self.style_indicators = {
            'business': ['business', 'professional', 'office', 'corporate', 'executive'],
            'casual': ['casual', 'relaxed', 'everyday', 'weekend', 'leisure'],
            'athletic': ['athletic', 'sport', 'gym', 'workout', 'running', 'yoga'],
            'formal': ['formal', 'evening', 'cocktail', 'black-tie', 'white-tie']
        }
    
    def classify_item(self, item_name: str, category: str, description: str = "") -> FashionItem:
        """Comprehensive item classification with confidence scoring"""
        
        text = f"{item_name} {description}".lower()
        category = category.lower()
        
        # Determine formality level
        formality_scores = {}
        for formality_level, categories in self.formality_rules.items():
            score = 0
            if category in categories:
                for keyword in categories[category]:
                    if keyword in text:
                        score += 1
            formality_scores[formality_level] = score
        
        # Add fabric-based scoring
        for fabric_level, fabrics in self.fabric_formality.items():
            for fabric in fabrics:
                if fabric in text:
                    if fabric_level == 'formal':
                        formality_scores['formal'] = formality_scores.get('formal', 0) + 0.5
                    elif fabric_level == 'casual':
                        formality_scores['casual'] = formality_scores.get('casual', 0) + 0.5
        
        # Determine primary formality
        if not any(formality_scores.values()):
            formality = 'neutral'
        else:
            formality = max(formality_scores, key=formality_scores.get)
        
        # Determine style category
        style_scores = {}
        for style, keywords in self.style_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text)
            style_scores[style] = score
        
        style_category = max(style_scores, key=style_scores.get) if any(style_scores.values()) else 'general'
        
        # Extract fabric and color (simplified)
        fabric_type = self._extract_fabric(text)
        color_family = self._extract_color(text)
        season_tags = self._extract_season_tags(text)
        
        return FashionItem(
            name=item_name,
            category=category,
            formality=formality,
            style_category=style_category,
            season_tags=season_tags,
            fabric_type=fabric_type,
            color_family=color_family
        )
    
    def _extract_fabric(self, text: str) -> str:
        fabric_keywords = ['cotton', 'wool', 'silk', 'polyester', 'linen', 'cashmere', 
                          'denim', 'leather', 'suede', 'canvas', 'fleece']
        for fabric in fabric_keywords:
            if fabric in text:
                return fabric
        return 'unknown'
    
    def _extract_color(self, text: str) -> str:
        color_keywords = ['black', 'white', 'blue', 'red', 'green', 'yellow', 
                         'navy', 'gray', 'brown', 'pink', 'purple', 'orange']
        for color in color_keywords:
            if color in text:
                return color
        return 'unknown'
    
    def _extract_season_tags(self, text: str) -> List[str]:
        season_keywords = {
            'winter': ['winter', 'wool', 'cashmere', 'fleece', 'thermal'],
            'summer': ['summer', 'cotton', 'linen', 'lightweight', 'breathable'],
            'spring': ['spring', 'light', 'transitional'],
            'fall': ['fall', 'autumn', 'layering']
        }
        tags = []
        for season, keywords in season_keywords.items():
            if any(keyword in text for keyword in keywords):
                tags.append(season)
        return tags or ['all-season']

class FashionCompatibilityEngine:
    """Advanced compatibility engine with formal fashion rules"""
    
    def __init__(self):
        self.classifier = AdvancedFormalityClassifier()
        
        # Formality compatibility matrix (PhD-level fashion theory)
        self.formality_matrix = {
            ('formal', 'formal'): 1.0,
            ('formal', 'smart_casual'): 0.7,
            ('formal', 'casual'): 0.2,  # Major penalty - prevents formal shirt + casual shorts
            ('smart_casual', 'smart_casual'): 0.9,
            ('smart_casual', 'formal'): 0.7,
            ('smart_casual', 'casual'): 0.6,
            ('casual', 'casual'): 0.9,
            ('casual', 'smart_casual'): 0.6,
            ('casual', 'formal'): 0.2,  # Major penalty
            ('neutral', 'formal'): 0.8,
            ('neutral', 'smart_casual'): 0.8,
            ('neutral', 'casual'): 0.8,
            ('formal', 'neutral'): 0.8,
            ('smart_casual', 'neutral'): 0.8,
            ('casual', 'neutral'): 0.8,
            ('neutral', 'neutral'): 0.7
        }
        
        # Occasion-based rules
        self.occasion_rules = {
            'business': {
                'required_formality': 'formal',
                'acceptable_combinations': [('formal', 'formal'), ('formal', 'smart_casual')],
                'forbidden_items': ['shorts', 'flip-flops', 'tank tops', 'graphic tees']
            },
            'casual': {
                'required_formality': 'casual',
                'acceptable_combinations': [('casual', 'casual'), ('smart_casual', 'casual')],
                'encouraged_items': ['jeans', 'sneakers', 't-shirts']
            }
        }
    
    def compute_outfit_compatibility(self, items: List[Dict], weather_context: Dict = None) -> Dict:
        """Compute comprehensive outfit compatibility with detailed scoring"""
        
        if len(items) < 2:
            return {'score': 0.0, 'explanation': 'Insufficient items for outfit'}
        
        # Classify all items
        classified_items = []
        for item in items:
            fashion_item = self.classifier.classify_item(
                item.get('name', ''),
                item.get('slot', ''),
                item.get('description', '')
            )
            classified_items.append(fashion_item)
        
        # Core compatibility scoring
        formality_score = self._compute_formality_compatibility(classified_items)
        color_harmony_score = self._compute_color_harmony(classified_items)
        seasonal_score = self._compute_seasonal_appropriateness(classified_items, weather_context)
        style_coherence_score = self._compute_style_coherence(classified_items)
        
        # Weighted final score (based on fashion theory importance)
        final_score = (
            0.40 * formality_score +      # Most important: formality matching
            0.25 * seasonal_score +       # Weather appropriateness
            0.20 * style_coherence_score + # Style consistency
            0.15 * color_harmony_score    # Color coordination
        )
        
        return {
            'score': final_score,
            'formality_analysis': self._analyze_formality_mix(classified_items),
            'individual_scores': {
                'formality': formality_score,
                'seasonal': seasonal_score,
                'style': style_coherence_score,
                'color': color_harmony_score
            }
        }
    
    def _compute_formality_compatibility(self, items: List[FashionItem]) -> float:
        """Compute formality compatibility using fashion theory matrix"""
        if len(items) < 2:
            return 1.0
        
        pairwise_scores = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1, item2 = items[i], items[j]
                pair_key = (item1.formality, item2.formality)
                reverse_key = (item2.formality, item1.formality)
                
                score = self.formality_matrix.get(pair_key, 
                        self.formality_matrix.get(reverse_key, 0.5))
                
                # Apply category-specific rules
                if item1.category == 'tops' and item2.category == 'bottoms':
                    score *= 1.2  # Emphasize top-bottom pairing
                
                pairwise_scores.append(score)
        
        return np.mean(pairwise_scores) if pairwise_scores else 0.5
    
    def _compute_color_harmony(self, items: List[FashionItem]) -> float:
        """Color harmony based on color theory"""
        colors = [item.color_family for item in items if item.color_family != 'unknown']
        if len(colors) < 2:
            return 0.8
        
        harmony_score = 0.0
        neutral_colors = {'black', 'white', 'gray', 'navy', 'brown', 'beige'}
        
        for i, color1 in enumerate(colors):
            for color2 in colors[i+1:]:
                if color1 == color2:
                    harmony_score += 0.8
                elif color1 in neutral_colors or color2 in neutral_colors:
                    harmony_score += 0.9
                else:
                    harmony_score += 0.6
        
        pairs_count = len(colors) * (len(colors) - 1) // 2
        return harmony_score / max(pairs_count, 1)
    
    def _compute_seasonal_appropriateness(self, items: List[FashionItem], weather_context: Dict) -> float:
        """Weather and season appropriateness scoring"""
        if not weather_context:
            return 1.0
        
        weather_hint = weather_context.get('hint', 'mild')
        temperature = weather_context.get('temp_c', 20)
        
        scores = []
        for item in items:
            item_score = 1.0
            
            if weather_hint == 'hot' or temperature > 25:
                if any(tag in ['winter'] for tag in item.season_tags):
                    item_score *= 0.6
                if item.fabric_type in ['wool', 'cashmere', 'fleece']:
                    item_score *= 0.5
                    
            elif weather_hint == 'cold' or temperature < 10:
                if any(tag in ['summer'] for tag in item.season_tags):
                    item_score *= 0.6
                if item.fabric_type in ['wool', 'cashmere', 'fleece']:
                    item_score *= 1.3
                    
            scores.append(item_score)
        
        return np.mean(scores)
    
    def _compute_style_coherence(self, items: List[FashionItem]) -> float:
        """Style coherence scoring"""
        style_categories = [item.style_category for item in items]
        
        if len(set(style_categories)) == 1:
            return 0.9  # All same style
        elif len(set(style_categories)) == len(style_categories):
            return 0.4  # All different styles
        else:
            return 0.7  # Some style consistency
    
    def _analyze_formality_mix(self, items: List[FashionItem]) -> Dict:
        """Analyze formality distribution and identify issues"""
        formality_counts = {}
        for item in items:
            formality_counts[item.formality] = formality_counts.get(item.formality, 0) + 1
        
        analysis = {
            'formality_distribution': formality_counts,
            'primary_formality': max(formality_counts, key=formality_counts.get),
            'mixed_formality': len(formality_counts) > 1,
            'problematic_pairs': []
        }
        
        # Identify problematic formal-casual combinations
        if 'formal' in formality_counts and 'casual' in formality_counts:
            formal_items = [item for item in items if item.formality == 'formal']
            casual_items = [item for item in items if item.formality == 'casual']
            
            for formal_item in formal_items:
                for casual_item in casual_items:
                    analysis['problematic_pairs'].append({
                        'formal_item': formal_item.name,
                        'casual_item': casual_item.name,
                        'issue': 'Formal-casual mismatch'
                    })
        
        return analysis