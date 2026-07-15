#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate short, genre-conditioned stories with an instruct-tuned model. Genres
also include a null baseline for comparison.

Usage:
    python generate_stories.py <output.parquet> -m <model> -n 1000 -b 32
"""

import argparse
from pathlib import Path
from string import Template

import pandas as pd
from torch.utils.data import Dataset
from transformers import GenerationConfig, pipeline, set_seed

SYSTEM_PROMPT = (
    "You are a creative-writing assistant. "
    "Write only the requested story text. Never add commentary."
)

PROMPT = Template(
    "Write a complete$genre short story of no more than 1,000 words. "
    "The story must have a clear ending. "
    "Do not include a title, preface, notes, or commentary.\n\nStory:\n"
)

GENRES = {
    "Horror": "horror",
    "Detective fiction": "detective fiction",
    "Romance": "romance",
    "Science fiction": "science fiction",
    "Thriller": "thriller",
    "NULL": None,
}

GENERATION_CONFIG = GenerationConfig(
    do_sample=True,
    max_new_tokens=2000,
    early_stopping=True,
    temperature=0.6,
    top_p=0.95,
    top_k=50,
)


class PromptDataset(Dataset):
    """Prompt dataset for generation batching."""

    def __init__(self, prompt, n, tokenizer):
        """Initialize the dataset.

        Parameters
        ----------
        prompt : str
            Generation prompt
        n : int
            Number of prompts
        tokenizer : transformers.AutoTokenizer
            Model tokenizer
        """
        self.n = n

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        self.prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __len__(self):
        """Get the length of the dataset."""
        return self.n

    def __getitem__(self, i):
        """Get a prompt formatted for instruct-tuned models."""
        return self.prompt


def _extract_text(output):
    """Extract generated text from pipeline.

    Parameters
    ----------
    output : list[str] or dict[str]
        Generated text

    Returns
    -------
    str
        Generated text
    """
    if isinstance(output, list):
        output = output[0]

    text = output["generated_text"]

    if isinstance(text, list):
        text = text[-1]["content"]

    return text.strip()


def generate_stories(pipe, genre_label, n=1000, batch_size=16):
    """Generate ``n`` genere-conditioned stories.

    Parameters
    ----------
    pipe : transformers.pipeline
        Text pipe
    genre_label : str
        Label of genre to generate
    n : int
        Number of stories to generate
    batch_size : int
        Batch size for generation

    Returns
    -------
    list[str]
        Generated story texts
    """
    genre = GENRES[genre_label]
    prompt = PROMPT.substitute(genre=f" {genre}" if genre else "")
    dataset = PromptDataset(prompt, n, tokenizer=pipe.tokenizer)

    outputs = pipe(
        dataset,
        generation_config=GENERATION_CONFIG,
        return_full_text=False,
        batch_size=batch_size,
    )

    return [_extract_text(seq) for seq in outputs]


def main(args):
    """Run the script."""
    set_seed(args.seed)

    pipe = pipeline("text-generation", model=args.model, device_map="auto")
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    pipe.tokenizer.padding_side = "left"
    GENERATION_CONFIG.pad_token_id = pipe.tokenizer.pad_token_id
    GENERATION_CONFIG.eos_token_id = pipe.model.generation_config.eos_token_id

    output = []
    for genre_label in GENRES:
        stories = generate_stories(
            pipe,
            genre_label,
            n=args.n_stories,
            batch_size=args.batch_size,
        )
        df = pd.DataFrame({"genre": genre_label, "text": stories})
        output.append(df)

    output = pd.concat(output, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate short genre-conditioned stories"
    )
    parser.add_argument("output", type=Path, help="Output parquet file")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="allenai/Olmo-3-7B-Instruct-SFT",
        help="Model checkpoint",
    )
    parser.add_argument(
        "-n",
        "--n-stories",
        type=int,
        default=1000,
        help="Number of stories per genre",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32, help="Pipeline batch size"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=5167, help="Random seed"
    )
    args = parser.parse_args()
    main(args)
