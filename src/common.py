#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for the genre classifier and its gradient attributions.

Both `finetune.py` and `attributions_ft.py` import from this module so that the
story span, tokenization, and span-mean pooling are computed identically across
training and attribution.
"""

import numpy as np

GENRES = [
    "Horror",
    "Detective fiction",
    "Romance",
    "Science fiction",
    "Thriller",
]

LETTERS = ["A", "B", "C", "D", "E"]

# Story sentinel for locating the story's character span inside the rendered
# chat prompt
STORY_SENTINEL = "\ue000__GENRE_STORY_SENTINEL__\ue001"

SYSTEM_PROMPT = (
    "You are a literary classification assistant. "
    "Read the story and determine its genre."
)

USER_TEMPLATE = "Story:\n{story}"


def build_chat_prompt(tokenizer, story):
    """Render the story-only prompt and return the story's character span.

    Parameters
    ----------
    tokenizer : transformers.AutoTokenizer
        Fast tokenizer with a chat template
    story : str
        Story text

    Returns
    -------
    tuple[str, int, int]
        The rendered prompt, story_start_char, story_end_char

    Raises
    ------
    ValueError
        If the story text already contains the sentinel
    """
    if STORY_SENTINEL in story:
        raise ValueError("Story text contains STORY_SENTINEL, can't extract")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(story=STORY_SENTINEL),
        },
    ]

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    story_start = rendered.index(STORY_SENTINEL)
    story_end = story_start + len(story)

    prompt = rendered.replace(STORY_SENTINEL, story, 1)

    return prompt, story_start, story_end


def tokenize_with_story_mask(tokenizer, story, device=None):
    """Tokenize the story-only prompt and build a story-span token mask.

    Parameters
    ----------
    tokenizer : transformers.AutoTokenizer
        Fast tokenizer
    story : str
        Story text
    device : torch.device or str or None
        Optional device for the returned tensors

    Returns
    -------
    tuple[str, dict, list[tuple[int, int]], np.ndarray]
        The prompt, tokenized inputs, token offsets, and a boolean story mask
        (True where a token overlaps the story span)

    Raises
    ------
    ValueError
        If the tokenizer is not a fast tokenizer
    """
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for offset mappings")

    prompt, story_start, story_end = build_chat_prompt(tokenizer, story)

    inputs = tokenizer(
        prompt, return_tensors="pt", return_offsets_mapping=True
    )
    offsets = inputs.pop("offset_mapping")[0].tolist()

    story_token_mask = np.array(
        [
            bool(end > story_start and start < story_end and end > start)
            for start, end in offsets
        ],
        dtype=bool,
    )

    if story_token_mask.sum() == 0:
        raise ValueError("Story span mask is empty; check the chat template")

    if device is not None:
        inputs = inputs.to(device)

    return prompt, inputs, offsets, story_token_mask


def span_mean_pool(hidden_states, mask):
    """Mean-pool hidden states over the masked (story) token positions.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Hidden states, shape (batch, seq_len, d_model) or (seq_len, d_model)
    mask : torch.Tensor
        Boolean mask over sequence positions, shape (seq_len,)

    Returns
    -------
    torch.Tensor
        Pooled vector, shape (batch, d_model) or (d_model,)
    """
    if hidden_states.dim() == 3:
        # (batch, seq, d) -> select over seq
        m = mask.to(hidden_states.dtype)[None, :, None]
        summed = (hidden_states * m).sum(dim=1)
        counts = m.sum(dim=1).clamp_min(1.0)

        return summed / counts

    m = mask.to(hidden_states.dtype)[:, None]
    summed = (hidden_states * m).sum(dim=0)
    counts = m.sum(dim=0).clamp_min(1.0)

    return summed / counts
