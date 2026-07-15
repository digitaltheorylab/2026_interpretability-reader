#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradient token attributions for the fine-tuned genre head.

Instead of backpropagating an answer-letter margin from the MCQA logits (as in
``mcqa.py``), we backpropagate a genre margin from a linear head trained on
mean-pooled hidden states of stories (see ``finetune.py``).

For each story we:
  1. Embed the prompt tokens with gradients enabled
  2. Run the frozen backbone, requesting hidden states
  3. Mean-pool the chosen layer's hidden states over the story span
  4. Apply the affine genre head -> 5 genre logits
  5. Backprop the objective correct_logit - logsumexp(other_logits)
  6. Record per-token gradient norms

Usage:
    python attributions_ft.py <stories.parquet> <head.npz> <output.parquet> \
        -m <model>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import GENRES, span_mean_pool, tokenize_with_story_mask


def forward_hidden_states_with_grads(model, input_ids, attention_mask=None):
    """Forward pass over explicit input embeddings, returning hidden states.

    Parameters
    ----------
    model : transformers.AutoModelForCausalLM
        Frozen backbone
    input_ids : torch.Tensor
        Input IDs, shape (1, seq_len)
    attention_mask : torch.Tensor or None
        Attention mask

    Returns
    -------
    tuple[tuple[torch.Tensor], torch.Tensor]
        Per-layer hidden states (the model output tuple) and the input
        embeddings tensor
    """
    model.zero_grad(set_to_none=True)

    embed = model.get_input_embeddings()
    inputs_embeds = embed(input_ids).detach()
    inputs_embeds.requires_grad_(True)
    inputs_embeds.retain_grad()

    forward_kwargs = {
        "inputs_embeds": inputs_embeds,
        "use_cache": False,
        "output_hidden_states": True,
    }
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask

    outputs = model(**forward_kwargs)

    return outputs.hidden_states, inputs_embeds


def genre_margin_attribution(
    head_weight,
    head_bias,
    chosen_layer,
    layer_hidden_states,
    story_mask,
    inputs_embeds,
    true_genre,
    model=None,
):
    """Compute genre logits and per-token gradient norms for one story.

    Objective: ``correct_genre_logit - logsumexp(other_genre_logits)``.

    Parameters
    ----------
    head_weight : torch.Tensor
        Genre head weight, shape (n_genre, d_model)
    head_bias : torch.Tensor
        Genre head bias, shape (n_genre,)
    chosen_layer : int
        Index into the transformer layers (0-based, excluding the embedding
        output) whose hidden state feeds the head
    layer_hidden_states : tuple[torch.Tensor]
        Hidden states from the forward pass; index 0 is the embedding output,
        so the chosen transformer layer is at ``chosen_layer + 1``
    story_mask : torch.Tensor
        Boolean story-span mask, shape (seq_len,)
    inputs_embeds : torch.Tensor
        Input embeddings with grad, shape (1, seq_len, d_model)
    true_genre : str
        Correct genre label
    model : transformers.AutoModelForCausalLM or None
        If given, gradients are cleared before backward

    Returns
    -------
    tuple[np.ndarray, str]
        Per-token gradient norms (seq_len,) and the predicted genre
    """
    if model is not None:
        model.zero_grad(set_to_none=True)

    if inputs_embeds.grad is not None:
        inputs_embeds.grad = None

    # Drop the embedding output
    hs = layer_hidden_states[chosen_layer + 1][0]  # (seq_len, d_model)

    # Pool the remaining states and send them through the fine-tuned head
    pooled = span_mean_pool(hs, story_mask)  # (d_model,)
    logits = pooled @ head_weight.T + head_bias  # (n_genre,)

    correct_idx = GENRES.index(true_genre)
    correct_logit = logits[correct_idx]
    other_logits = torch.cat([logits[:correct_idx], logits[correct_idx + 1 :]])

    objective = correct_logit - torch.logsumexp(other_logits, dim=0)
    objective.backward()

    token_grad_norms = inputs_embeds.grad.detach().norm(dim=-1)[0]

    predicted_genre = GENRES[int(logits.detach().argmax())]

    return token_grad_norms.float().cpu().numpy(), predicted_genre


def process_story(
    story_id, text, true_genre, model, tokenizer, head, device
):
    """Process one story end-to-end.

    Parameters
    ----------
    story_id : int
        Story ID
    text : str
        Story text
    true_genre : str
        Correct genre
    model : transformers.AutoModelForCausalLM
        Frozen backbone
    tokenizer : transformers.AutoTokenizer
        Fast tokenizer
    head : dict
        Head artifact tensors: ``weight``, ``bias``, ``chosen_layer``
    device : torch.device
        Model device

    Returns
    -------
    dict
        Row matching the chapter-6 attribution schema
    """
    prompt, inputs, offsets, story_token_mask = tokenize_with_story_mask(
        tokenizer=tokenizer, story=text, device=device
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    story_mask = torch.as_tensor(
        story_token_mask, dtype=torch.bool, device=device
    )

    hidden_states, inputs_embeds = forward_hidden_states_with_grads(
        model=model, input_ids=input_ids, attention_mask=attention_mask
    )

    token_grads, predicted_genre = genre_margin_attribution(
        head_weight=head["weight"],
        head_bias=head["bias"],
        chosen_layer=head["chosen_layer"],
        layer_hidden_states=hidden_states,
        story_mask=story_mask,
        inputs_embeds=inputs_embeds,
        true_genre=true_genre,
        model=model,
    )

    input_id_list = input_ids[0].cpu().tolist()
    tokens = [tokenizer.decode(tid) for tid in input_id_list]

    return {
        "story_id": int(story_id),
        "true_genre": true_genre,
        "predicted_genre": predicted_genre,
        "prompt": prompt,
        "tokens": tokens,
        "token_offsets": offsets,
        "input_ids": input_id_list,
        "token_grads": token_grads.tolist(),
        "story_token_mask": story_token_mask.tolist(),
    }


def main(args):
    """Run the script."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Reproduce the same sampling as finetune.py so story_ids align
    df = pd.read_parquet(args.input)
    df = df[df["genre"] != "NULL"]
    df = df.groupby("genre").sample(n=args.n_per_genre, random_state=args.seed)
    df.reset_index(drop=True, inplace=True)

    # Ensure story_id is an integer
    df["story_id"] = df.index.astype(int)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()

    # No gradients needed here
    for p in model.parameters():
        p.requires_grad_(False)

    device = model.get_input_embeddings().weight.device

    # Load the trained head and move it onto the model device
    artifact = np.load(args.head, allow_pickle=False)
    classes = [str(g) for g in artifact["genre_classes"]]
    if classes != GENRES:
        raise ValueError(
            f"Head genre order {classes} != expected {GENRES}; "
            "retrain or align GENRES"
        )

    head = {
        "weight": torch.tensor(
            artifact["head_weight"], dtype=model.dtype, device=device
        ),
        "bias": torch.tensor(
            artifact["head_bias"], dtype=model.dtype, device=device
        ),
        "chosen_layer": int(artifact["chosen_layer"]),
    }
    print(f"Loaded head: layer {head['chosen_layer']}, classes {classes}")
    print(f"Attributing {len(df)} stories through {args.model}...")

    results = []
    for n, row in enumerate(df.itertuples(index=False), start=1):
        results.append(
            process_story(
                story_id=row.story_id,
                text=row.text,
                true_genre=row.genre,
                model=model,
                tokenizer=tokenizer,
                head=head,
                device=device,
            )
        )
        if n % args.log_at == 0:
            print(f"Processed {n}/{len(df)} stories")

    out = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    acc = (out["predicted_genre"] == out["true_genre"]).mean()
    print(f"Finished. Genre accuracy across attributed stories: {acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradient token attributions for the fine-tuned genre head"
    )
    parser.add_argument("input", type=Path, help="Stories parquet")
    parser.add_argument("head", type=Path, help="Head .npz from finetune.py")
    parser.add_argument("output", type=Path, help="Output parquet")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="allenai/Olmo-3-7B-Instruct-SFT",
        help="Model checkpoint",
    )
    parser.add_argument(
        "-n",
        "--n-per-genre",
        type=int,
        default=100,
        help="Number of stories per genre (must match finetune.py)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=5167, help="Random seed"
    )
    parser.add_argument(
        "--log-at", type=int, default=25, help="How often to log progress"
    )
    args = parser.parse_args()
    main(args)
