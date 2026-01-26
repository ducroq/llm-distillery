"""
Text preprocessing utilities for inference pipelines.

Provides preprocessing functions that match training-time transformations.
"""


def extract_head_tail(
    text: str,
    tokenizer,
    head_tokens: int = 256,
    tail_tokens: int = 256,
    separator: str = " [...] "
) -> str:
    """
    Extract first N + last M tokens from text.

    This preprocessing technique captures both the beginning (context, setup)
    and end (conclusions, outcomes) of articles, which often contain the most
    informative content for classification.

    Args:
        text: Input text to process
        tokenizer: HuggingFace tokenizer (must have encode/decode methods)
        head_tokens: Number of tokens to keep from the beginning
        tail_tokens: Number of tokens to keep from the end
        separator: String to insert between head and tail sections

    Returns:
        Processed text with head + separator + tail, or original text if
        shorter than head_tokens + tail_tokens

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        >>> text = "A very long article..." * 1000
        >>> result = extract_head_tail(text, tokenizer)
        >>> # Returns first 256 tokens + " [...] " + last 256 tokens
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Short enough - return as-is
    if len(tokens) <= head_tokens + tail_tokens:
        return text

    # Extract head and tail
    head = tokenizer.decode(tokens[:head_tokens], skip_special_tokens=True)
    tail = tokenizer.decode(tokens[-tail_tokens:], skip_special_tokens=True)

    return head + separator + tail
