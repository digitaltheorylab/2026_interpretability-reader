#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a linear genre head on a frozen model backbone.

The head is a 5-way logistic regression classifier. It learns from the
mean-pooled hidden states of story spans.

This script:
  1. Caches mean-hidden states for each story
  2. Trains a cross-validated linear head at every layer
  3. Selects the layer where genre is most decodable (the "probe-best" layer)
  4. Refits the head on a train split at that layer
  5. Saves the head weights, chosen layer, classes, and the train/test split

Outputs (one .npz):
  - head_weight          (n_genre, d_model)
  - head_bias            (n_genre,)
  - chosen_layer         int
  - genre_classes        (n_genre,) str
  - story_ids            (n_story,) int
  - pooled_states        (n_story, n_layer, d_model) float16  (optional cache)
  - cv_accuracy_by_layer (n_layer,) float
  - train_idx, test_idx

Usage:
    python train_probe.py <stories.parquet> <output.npz> -m <model>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import GENRES, span_mean_pool, tokenize_with_story_mask


@torch.no_grad()
def capture_pooled_states(model, inputs, story_token_mask, device):
    """Capture per-layer, story-span-mean-pooled hidden states for one story.

    Parameters
    ----------
    model : transformers.AutoModelForCausalLM
        Frozen backbone
    inputs : dict
        Tokenized inputs (already on device)
    story_token_mask : np.ndarray
        Boolean story-span mask, shape (seq_len,)
    device : torch.device
        Model device

    Returns
    -------
    np.ndarray
        Pooled hidden states, shape (n_layer, d_model), float16
    """
    mask = torch.as_tensor(story_token_mask, dtype=torch.bool, device=device)

    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # Exclude the first layer, which is the embedding output
    hs = outputs.hidden_states[1:]

    pooled = []
    for layer_hs in hs:
        vec = span_mean_pool(layer_hs[0], mask)  # (d_model,)
        pooled.append(vec.float().cpu())

    return torch.stack(pooled, dim=0).to(torch.float16).numpy()


def probe_layer_cv(X, y, n_splits=5, seed=5167):
    """Cross-validated linear-probe accuracy at one layer.

    Parameters
    ----------
    X : np.ndarray
        Pooled hidden states, shape (n_story, d_model)
    y : np.ndarray
        Genre labels, shape (n_story,)
    n_splits : int
        CV folds
    seed : int
        Random seed

    Returns
    -------
    float
        Mean CV accuracy
    """
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(l1_ratio=0, C=1.0, max_iter=2000),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    return scores.mean()


def fit_head(X_train, y_train, classes):
    """Fit a linear head so weights apply directly to raw pooled hidden states
    at attribution time.

    This head is a single affine map (W, B), which means attribution can
    backpropagate through ``hs @ W.T + b`` without any extra preprocessing.

    Parameters
    ----------
    X_train : np.ndarray
        Pooled hidden states, shape (n_train, d_model)
    y_train : np.ndarray
        Genre labels, shape (n_train,)
    classes : list[str]
        Genre class order

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Head weight (n_genre, d_model) and bias (n_genre,), ordered by
        ``classes``
    """
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(X_train, y_train)

    # Reorder rows to match the genre order
    order = [list(clf.classes_).index(g) for g in classes]
    W = clf.coef_[order].astype(np.float32)
    b = clf.intercept_[order].astype(np.float32)

    return W, b


def main(args):
    """Run the script."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the stories and drop the NULL genre, then sample
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

    # No gradients needed
    for p in model.parameters():
        p.requires_grad_(False)

    device = model.get_input_embeddings().weight.device
    print(
        f"Caching pooled states for {len(df)} stories through {args.model}..."
    )

    pooled_all = []
    story_ids = []
    genres = []

    # Step through the stories and obtain the pooled hidden states
    for n, row in enumerate(df.itertuples(index=False), start=1):
        _, inputs, _, story_token_mask = tokenize_with_story_mask(
            tokenizer=tokenizer, story=row.text, device=device
        )
        pooled = capture_pooled_states(model, inputs, story_token_mask, device)
        pooled_all.append(pooled)
        story_ids.append(int(row.story_id))
        genres.append(row.genre)

        if n % args.log_at == 0:
            print(f"Cached {n}/{len(df)} stories")

    # Stack the hidden states, shape (n_story, n_layer, d_model)
    pooled_all = np.stack(pooled_all, axis=0)
    story_ids = np.array(story_ids)
    y = np.array(genres)

    n_layer = pooled_all.shape[1]
    print(f"Pooled states: {pooled_all.shape}. Probing {n_layer} layers...")

    # Probe every layer to find where genre is most decodable
    cv_acc = np.zeros(n_layer, dtype=np.float32)
    for layer in range(n_layer):
        X = pooled_all[:, layer, :].astype(np.float32)
        cv_acc[layer] = probe_layer_cv(X, y, seed=args.seed)
        print(f"  layer {layer:2d}: CV acc = {cv_acc[layer]:.3f}")

    argmax_layer = int(cv_acc.argmax())

    if args.layer is not None:
        if not 0 <= args.layer < n_layer:
            raise ValueError(
                f"--layer {args.layer} out of range [0, {n_layer - 1}]"
            )
        chosen_layer = int(args.layer)
        print(
            f"Forced layer: {chosen_layer} (CV acc = {cv_acc[chosen_layer]:.3f})"
            f" | argmax would be {argmax_layer} "
            f"(CV acc = {cv_acc[argmax_layer]:.3f})"
        )
    else:
        chosen_layer = argmax_layer
        print(
            f"Chosen layer: {chosen_layer} (CV acc = {cv_acc[chosen_layer]:.3f})"
        )

    # Do a train/test split for a held-out
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    X_layer = pooled_all[:, chosen_layer, :].astype(np.float32)
    W, b = fit_head(X_layer[train_idx], y[train_idx], GENRES)

    # How well does the head perform on the held-out set?
    test_logits = X_layer[test_idx] @ W.T + b
    test_pred = np.array(GENRES)[test_logits.argmax(axis=1)]
    test_acc = (test_pred == y[test_idx]).mean()
    print(f"Held-out head accuracy at layer {chosen_layer}: {test_acc:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "head_weight": W,
        "head_bias": b,
        "chosen_layer": np.int64(chosen_layer),
        "argmax_layer": np.int64(argmax_layer),
        "genre_classes": np.array(GENRES),
        "story_ids": story_ids,
        "cv_accuracy_by_layer": cv_acc,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "held_out_accuracy": np.float32(test_acc),
    }
    if args.save_states:
        save_kwargs["pooled_states"] = pooled_all.astype(np.float16)

    np.savez_compressed(args.output, **save_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a linear genre head on a frozen backbone"
    )
    parser.add_argument("input", type=Path, help="Stories parquet")
    parser.add_argument("output", type=Path, help="Output .npz artifact")
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
        help="Number of stories per genre",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Held-out fraction"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Force the head's layer instead of selecting the most decodable one",
    )
    parser.add_argument(
        "--save-states",
        action="store_true",
        help="Also save pooled hidden states in the artifact",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=5167, help="Random seed"
    )
    parser.add_argument(
        "--log-at", type=int, default=25, help="How often to log progress"
    )
    args = parser.parse_args()
    main(args)
