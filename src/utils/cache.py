from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_response(prompt):
    pass
