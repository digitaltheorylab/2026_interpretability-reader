#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-run the MCQA prompts/stories and capture per-layer hidden states for
probing.

Usage:
    python extract_hiddens.py <input.parquet> <output.parquet> -m <model>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from classify_mcqa import capture_layer_last_token_states


@torch.no_grad()
def capture_hidden_states(input_ids, model, device):
    """Run the model on a prompt and return per-layer hidden states at the
    answer position.

    Parameters
    ----------
    input_ids : list[str]
        Input IDs
    model : transformers.AutoModelForCausalLM
        The LLM
    device : torch.device
        Model device

    Returns
    -------
    np.ndarray
        Hidden states, shape (n_layer + 1, d_model). The last row is the
        post-final-norm hidden state (what the unembedding actually sees)
    """
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    with capture_layer_last_token_states(model) as hidden_states_by_layer:
        model(input_ids=input_ids)

    # Stack per-layer: list of (1, d_model) -> (n_layer, d_model)
    per_layer = torch.stack([hs[0] for hs in hidden_states_by_layer], dim=0)

    # Add the post-final-norm state. This is what the unembedding sees
    final_norm = getattr(model.model, "norm", None)
    if final_norm is not None:
        final_hs = final_norm(hidden_states_by_layer[-1])
    else:
        final_hs = hidden_states_by_layer[-1]

    all_states = torch.cat([per_layer, final_hs[0].unsqueeze(0)], dim=0)

    return all_states.to(torch.float16).cpu().numpy()


def main(args):
    """Run the script."""
    df = pd.read_parquet(args.input)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()

    device = model.get_input_embeddings().weight.device

    # Initialize some buffers for storing our hidden states
    all_hidden_states = []
    story_ids = []

    for n, row in enumerate(df.itertuples(index=False), start=1):
        hs = capture_hidden_states(
            input_ids=row.input_ids, model=model, device=device
        )

        all_hidden_states.append(hs)
        story_ids.append(row.story_id)

        if n % args.log_at == 0:
            print(f"Processed {n}/{len(df)} stories")

    # Stack the hidden states into a single array, which we'll save alongside
    # the story ids
    all_hidden_states = np.stack(all_hidden_states, axis=0).astype(np.float16)
    story_ids = np.array(story_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, hidden_states=all_hidden_states, story_ids=story_ids
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get per-layer hidden states from MCQA outputs"
    )
    parser.add_argument(
        "input", type=Path, help="Parquet file produced by mcqa.py"
    )
    parser.add_argument("output", type=Path, help="Output .npz file")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="allenai/Olmo-3-7B-Instruct-SFT",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--log-at", type=int, default=25, help="How often to log progress"
    )
    args = parser.parse_args()
    main(args)
