from __future__ import annotations
from typing import Dict, List

# Basic, rule-based fashion compatibility knowledge.
# Keys are garment names in lower case with underscores; values list items
# that can typically be worn with the key item.
COMPATIBILITY_RULES: Dict[str, List[str]] = {
    "off_shoulder_top": ["cargo_pants", "pocket_pants", "shorts"],
    "t_shirt": ["jeans", "shorts", "cargo_pants"],
    "hoodie": ["jeans", "cargo_pants", "joggers"],
    "blouse": ["skirt", "trousers", "jeans"],
}


def get_compatible_items(item: str) -> List[str]:
    """Return items that pair well with ``item``.

    The rules are lightweight and can be extended by modifying
    :data:`COMPATIBILITY_RULES`.
    """
    return COMPATIBILITY_RULES.get(item, [])


def are_compatible(item_a: str, item_b: str) -> bool:
    """Check whether two items can be worn together.

    The check is symmetric: if either item lists the other as
    compatible, the pair is considered a match.
    """
    if item_b in COMPATIBILITY_RULES.get(item_a, []):
        return True
    if item_a in COMPATIBILITY_RULES.get(item_b, []):
        return True
    return False
