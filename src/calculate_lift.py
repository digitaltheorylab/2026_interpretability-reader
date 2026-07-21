#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate genre lift scores from token attribution data.

This script aggregates token-level gradient attributions into word-level "genre
lift" scores. Lift measures how much more influential a word is in one genre
compared to its average influence across other genres.

Usage:
    python calculate_lift.py <attributions.parquet> <gnere-lift.parquet>
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

RE_PATTERN = re.compile(r"(?u)\b[a-zA-Z]{3,}\b")

CRUFT = set([
    "didn",
    "don",
    "doesn",
    "isn",
    "aren",
    "wasn",
    "weren",
    "hasn",
    "haven",
    "hadn",
    "won",
    "wouldn",
    "couldn",
    "shouldn",
    "can",
    "mustn",
    "needn",
    "shan",
    "mightn",
])

ENGLISH_STOP_WORDS = ENGLISH_STOP_WORDS | CRUFT

STORY_WORD_AGG = {
    "story_grad_sum": ("grad_mean", "sum"),
    "story_grad_mean": ("grad_mean", "mean"),
    "story_n": ("grad_mean", "size"),
}

GENRE_WORD_AGG = {
    "grad_mean_across_stories": ("story_grad_mean", "mean"),
    "grad_sum_across_stories": ("story_grad_sum", "sum"),
    "n_stories": ("story_id", "size"),
    "n_occurrences": ("story_n", "sum"),
}

def aggregate_tokens(
    text, offsets, gradients, pattern, mask=None, stop_words=None
):
    """Aggregate and filter subword tokens.

    Uses a regular expression match to find word-like tokens in story text and
    filters those tokens via a stop word list.

    Parameters
    ----------
    text : str
        Story text
    offsets : np.ndarray
        Token offsets from the LLM tokenizer, shape (n_token, 2)
    gradients : np.ndarray
        Token gradients, shape (n_token,)
    pattern : re.Pattern
        Regular expression pattern for finding tokens
    mask : np.ndarray or None
        Story mask for ignoring prompt tokens, shape (n_token,)
    stop_words : iterable or None
        Stop word list

    Returns
    -------
    list[dict]
        Aggregated words with metadata:
        - word: combined token string
        - span_start: start of word span
        - span_end: end of word span
        - token_indices: token indices in word
        - subwords: individual tokens in token string
        - n_tokens: number of tokens in word
        - grad_sum: sum of token gradients
        - grad_mean: mean of token gradients
    """
    if mask is None:
        mask = np.ones(len(offsets), dtype=bool)

    if stop_words is None:
        stop_words = set()

    # Stack offsets so offsets[:, 0] == start, offsets[:, 1] == end
    offsets = np.vstack(offsets)
    assert len(mask) == len(offsets), "Mask length doesn't match offset length"

    output = []

    for match in pattern.finditer(text):
        word = match.group(0)
        span_start, span_end = match.span()

        # Ignore stopwords
        if word.lower() in stop_words:
            continue

        # Overlap if: token_start < span_end and token_end > span_start
        (overlapping,) = np.where(
            mask
            & (offsets[:, 0] < span_end)
            & (offsets[:, 1] > span_start)
            & (offsets[:, 1] > offsets[:, 0])
        )

        if len(overlapping) == 0:
            continue

        # Build our word from tokens
        subwords = []

        for i in overlapping:
            tok_start, tok_end = offsets[i]

            clipped_start = max(tok_start, span_start)
            clipped_end = min(tok_end, span_end)

            subwords.append(text[clipped_start:clipped_end])

        # Get token gradients
        grads = gradients[overlapping]

        # Construct metadata
        row = {
            "word": word,
            "span_start": span_start,
            "span_end": span_end,
            "token_indices": overlapping.tolist(),
            "subwords": subwords,
            "n_tokens": len(overlapping),
            "grad_sum": grads.sum(),
            "grad_mean": grads.mean(),
        }

        output.append(row)

    return output


def aggregate(row):
    """Helper function for using ``aggregate_tokens()`` on DataFrame rows.

    Parameters
    ----------
    row : pd.Series
        DataFrame row

    Returns
    -------
    pd.DataFrame
        Aggregated token DataFrame
    """
    agg = aggregate_tokens(
        text=row["prompt"].lower(),
        offsets=row["token_offsets"],
        gradients=row["token_grads"],
        pattern=RE_PATTERN,
        mask=row["story_token_mask"],
        stop_words=ENGLISH_STOP_WORDS,
    )

    # Convert to a DataFrame and assign some metadata
    agg = pd.DataFrame(agg)
    agg["story_id"] = row["story_id"]
    agg["genre"] = row["true_genre"]

    return agg

def distinctive_words(df, min_stories=10, top_n=25):
    """Get distinctive words per genre.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated word DataFrame for all stories in the dataset
    min_stories : int
        Minimum number of stories a word must appear in within a genre
    top_n : int
        Number of distinctive words to return per genre

    Returns
    -------
    pd.DataFrame
        Distinctive words per genre, ranked by genre_lift
    """
    # One row per story/genre/word
    story_word = (
        df
        .groupby(["story_id", "genre", "word"], as_index=False)
        .agg(**STORY_WORD_AGG)
    )

    # One row per genre/word
    genre_word = (
        story_word
        .groupby(["genre", "word"], as_index=False)
        .agg(**GENRE_WORD_AGG)
    )

    # Group words, then count the number of word occurences and unique words
    word_group = genre_word.groupby("word")
    word_sum = word_group["grad_mean_across_stories"].transform("sum")
    word_cnt = word_group["grad_mean_across_stories"].transform("count")

    # Subtract this genre's own value before averaging
    genre_word["global_grad_mean"] = (
        (word_sum - genre_word["grad_mean_across_stories"]) / (word_cnt - 1)
    )

    # Calculate lift
    genre_word["genre_lift"] = (
        genre_word["grad_mean_across_stories"] - genre_word["global_grad_mean"]
    )

    # Filter to words above min_stories and return the top lifted words
    return (
        genre_word
        .query("n_stories >= @min_stories")
        .sort_values(["genre", "genre_lift"], ascending=[True, False])
        .groupby("genre", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def main(args):
    """Run the script."""
    df = pd.read_parquet(args.input)
    agg = df.apply(aggregate, axis=1)
    agg = pd.concat(agg.tolist(), ignore_index=True)

    lift = distinctive_words(
        agg, min_stories=args.min_stories, top_n=args.top_n
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lift.to_parquet(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate gradient lift from token attributions"
    )
    parser.add_argument("input", type=Path, help="Token attributions parquet")
    parser.add_argument("output", type=Path, help="Output parquet")
    parser.add_argument(
        "-m",
        "--min-stories",
        type=int,
        default=10,
        help="Minimum number of stories a word must appear in",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=250,
        help="Number of distinctive words to return per genre",
    )
    args = parser.parse_args()
    main(args)
