try:
    from duckduckgo_search import DDGS
except ImportError as e:
    raise ImportError(
        "The 'duckduckgo_search' package is required for DDGS. "
        "Install it with: pip install duckduckgo-search"
    ) from e

