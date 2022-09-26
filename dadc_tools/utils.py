from typing import List


def get_unique_order_preserving(seq: List, apply_custom_fn=lambda x: x, apply_filter_fn=lambda x: x) -> List:
    """Return all the unique values in a list while preserving order, unlike list(set(seq))"""
    seen = set()
    seen_add = seen.add
    return [apply_custom_fn(x) for x in seq if not (apply_filter_fn(apply_custom_fn(x)) in seen or seen_add(apply_filter_fn(apply_custom_fn(x))))]
