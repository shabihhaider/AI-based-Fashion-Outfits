# enhanced_wardrobe_integration.py
from fashion_rules import FashionCompatibilityEngine, AdvancedFormalityClassifier
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedWardrobeRecommender:
    """Enhanced wardrobe recommender with fashion rule integration"""
    
    def __init__(self, base_engine, config):
        self.base_engine = base_engine  # Your existing AdvancedRecommendationEngine
        self.fashion_engine = FashionCompatibilityEngine()
        self.config = config
        
        # Cache for classified items
        self.item_classification_cache = {}
        
    def enhanced_wardrobe_recommend(
        self,
        cat_lists: Dict[str, List],
        feats_map: Dict[str, torch.Tensor],
        names_map: Dict[str, List[str]],
        weather_context: Optional[Dict] = None,
        topk: int = 10,
        formality_weight: float = 0.4,
        enable_strict_rules: bool = True
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Enhanced wardrobe recommendation with fashion rules"""
        
        # Step 1: Generate base combinations using existing visual similarity
        base_combos = self._generate_base_combinations(
            cat_lists, feats_map, names_map, weather_context
        )
        
        # Step 2: Apply fashion rule filtering and scoring
        enhanced_combos = []
        rule_analytics = {
            'total_combinations': len(base_combos),
            'rule_filtered': 0,
            'fashion_scores': [],
            'formality_violations': 0,
            'weather_violations': 0
        }
        
        for combo_score, parts, base_explanation in base_combos:
            # Convert to fashion rule format
            fashion_items = []
            for part in parts:
                item_dict = {
                    'name': part.get('name', ''),
                    'slot': part.get('slot', ''),
                    'description': ''
                }
                fashion_items.append(item_dict)
            
            # Summer hard-ban: skip any outerwear combo when it's hot
            if (weather_context or {}).get("hint") == "hot" and any(p.get("slot") == "outerwear" for p in parts):
                rule_analytics['weather_violations'] += 1
                continue

            # Compute fashion compatibility
            fashion_result = self.fashion_engine.compute_outfit_compatibility(
                fashion_items, weather_context
            )
            
            fashion_score = fashion_result['score']
            fashion_explanation = fashion_result
            
            # Combine scores with configurable weighting
            if enable_strict_rules:
                # Strict mode: Heavy penalty for rule violations
                if fashion_score < 0.3:  # Major formality violation
                    rule_analytics['rule_filtered'] += 1
                    continue  # Skip this combination entirely
                
                combined_score = (
                    (1 - formality_weight) * combo_score +
                    formality_weight * fashion_score
                )
            else:
                # Soft mode: Weighted combination
                combined_score = (
                    0.6 * combo_score +
                    0.4 * fashion_score
                )
            
            # Track analytics
            rule_analytics['fashion_scores'].append(fashion_score)
            if fashion_explanation.get('formality_analysis', {}).get('problematic_pairs'):
                rule_analytics['formality_violations'] += 1
            
            # Enhanced explanation combining both systems
            enhanced_explanation = {
                **base_explanation,
                'fashion_rules': fashion_explanation,
                'combined_score': combined_score,
                'rule_compliance': 'high' if fashion_score > 0.7 else 'medium' if fashion_score > 0.4 else 'low'
            }
            
            enhanced_combos.append({
                'score': combined_score,
                'parts': parts,
                'explanation': enhanced_explanation
            })
        
        # Require footwear if user supplied shoes; drop outerwear when it's hot
        if cat_lists.get("footwear"):
            enhanced_combos = [c for c in enhanced_combos
                            if any(p.get("slot") == "footwear" for p in c["parts"])]
        if (weather_context or {}).get("hint") == "hot":
            enhanced_combos = [c for c in enhanced_combos
                            if not any(p.get("slot") == "outerwear" for p in c["parts"])]

        enhanced_combos.sort(key=lambda x: x['score'], reverse=True)
        final_combos = enhanced_combos if topk <= 0 else enhanced_combos[:topk]

        # Prepare analytics
        rule_analytics['final_combinations'] = len(final_combos)
        rule_analytics['avg_fashion_score'] = np.mean(rule_analytics['fashion_scores']) if rule_analytics['fashion_scores'] else 0
        
        return final_combos, rule_analytics
    
    def _generate_base_combinations(
        self,
        cat_lists: Dict[str, List],
        feats_map: Dict[str, torch.Tensor],
        names_map: Dict[str, List[str]],
        weather_context: Optional[Dict] = None
    ) -> List[Tuple[float, List[Dict], Dict]]:
        """Generate base combinations using existing visual similarity logic"""
        
        def _avg_pairwise_cos(indices):
            if len(indices) < 2:
                return 0.8
            
            pairs = []
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    bi, ii = indices[i]
                    bj, ij = indices[j]
                    Fi, Fj = feats_map[bi][ii].unsqueeze(0), feats_map[bj][ij].unsqueeze(0)
                    cos_sim = float((Fi @ Fj.T).detach().cpu().numpy().item())
                    pairs.append(cos_sim)
            
            return min(1.0, (np.mean(pairs) + 1.0) * 0.6)
        
        def _has(bucket): 
            return feats_map.get(bucket, torch.empty(0)).numel() > 0
        
        combos = []
        
        # Generate base combinations (tops + bottoms)
        bases = []
        if _has("tops") and _has("bottoms"):
            tops_count = feats_map["tops"].shape[0]
            bottoms_count = feats_map["bottoms"].shape[0]
            
            for it in range(tops_count):
                for ib in range(bottoms_count):
                    base_parts = [("tops", it), ("bottoms", ib)]
                    compat = _avg_pairwise_cos(base_parts)
                    bases.append((compat, base_parts))
        
        # For each base, create outfit variations
        for base_compat, base in bases:
            parts = [
                {"slot": b, "idx": i, "name": names_map[b][i] if i < len(names_map[b]) else f"{b}-{i}"}
                for (b, i) in base
            ]
            
            # Apply basic weather scoring
            weather_score = self._basic_weather_component(parts, weather_context)
            final_score = 0.7 * base_compat + 0.3 * weather_score
            
            base_explanation = {
                "visual_compatibility": base_compat,
                "weather_score": weather_score,
                "base_score": final_score
            }
            
            combos.append((final_score, parts, base_explanation))
            
            # Add variations with outerwear and footwear if available
            if _has("outerwear"):
                outerwear_count = feats_map["outerwear"].shape[0]
                for io in range(outerwear_count):
                    extended_parts = parts + [{"slot": "outerwear", "idx": io, 
                                             "name": names_map["outerwear"][io] if io < len(names_map["outerwear"]) else f"outerwear-{io}"}]
                    extended_base = base + [("outerwear", io)]
                    extended_compat = _avg_pairwise_cos(extended_base)
                    extended_weather = self._basic_weather_component(extended_parts, weather_context)
                    extended_score = 0.7 * extended_compat + 0.3 * extended_weather
                    
                    combos.append((extended_score, extended_parts, {
                        "visual_compatibility": extended_compat,
                        "weather_score": extended_weather,
                        "base_score": extended_score
                    }))
            
            if _has("footwear"):
                footwear_count = feats_map["footwear"].shape[0]
                for if_ in range(footwear_count):
                    extended_parts = parts + [{"slot": "footwear", "idx": if_, 
                                             "name": names_map["footwear"][if_] if if_ < len(names_map["footwear"]) else f"footwear-{if_}"}]
                    extended_base = base + [("footwear", if_)]
                    extended_compat = _avg_pairwise_cos(extended_base)
                    extended_weather = self._basic_weather_component(extended_parts, weather_context)
                    extended_score = 0.7 * extended_compat + 0.3 * extended_weather
                    
                    combos.append((extended_score, extended_parts, {
                        "visual_compatibility": extended_compat,
                        "weather_score": extended_weather,
                        "base_score": extended_score
                    }))
        
        return combos
    
    def _basic_weather_component(self, parts: List[Dict], weather_context: Optional[Dict]) -> float:
        """Basic weather compatibility"""
        if not weather_context:
            return 1.0
            
        hint = weather_context.get('hint', 'mild')
        if hint == 'mild':
            return 1.0
            
        text = " ".join([f"{p.get('slot','')} {p.get('name','')}".lower() for p in parts])
        
        if hint == "hot":
            penalty = 0.3 if any(w in text for w in ["jacket","coat","sweater","hoodie"]) else 0.0
            return max(0.0, 1.0 - penalty)
        elif hint == "cold":
            penalty = 0.4 if any(w in text for w in ["shorts","tank","sleeveless"]) else 0.0
            bonus = 0.2 if any(w in text for w in ["jacket","coat","sweater"]) else 0.0
            return max(0.0, min(1.0, 1.0 - penalty + bonus))
        elif hint == "rain":
            penalty = 0.3 if any(w in text for w in ["suede","canvas"]) else 0.0
            return max(0.0, 1.0 - penalty)
        
        return 1.0