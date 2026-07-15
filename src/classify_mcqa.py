#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multiple-choice genre classification data with logit lens and gradient
attribution.

Usage:
    python classify_mcqa.py <input.parquet> <output.parquet> -m <model>
"""

import argparse
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import GENRES, LETTERS, STORY_SENTINEL

# Task-specific classifier system prompt. Unlike the creative writing persona
# in ``storygen.py``, this frames the model as a genre classifier 
SYSTEM_PROMPT = (
    "You are a literary classification assistant. "
    "Read the story and the answer options, then respond with the single "
    "letter of the option that best identifies the story's genre."
)

# The user-turn body. The answer letter is predicted on the assistant-turn
# opener, which is supplied by the chat template's .add_generation_prompt()
USER_TEMPLATE = Template(
    "Read the following story and identify its genre.\n\n====\n\n"
    "Story:\n$story\n\n====\n\n"
    "$options"
)

@dataclass
class MCQAResult:
    # Metadata about the story
    story_id: int
    true_genre: str
    letter_to_genre: dict[str, str]
    correct_letter: str
    predicted_letter: str

    # Prompt + story text and tokens
    prompt: str
    tokens: list[str]
    token_offsets: list[tuple[int, int]]
    input_ids: list[int]

    # Model internals
    logits_by_layer: list[list[float]]
    token_grads: list[float]
    story_token_mask: list[bool]


def _get_module_device(module):
    """Return device for a module.

    Gets devices for device_map="auto", which may put model components on
    different devices.

    Parameters
    ----------
    module : torch.nn.Module
        The module

    Returns
    -------
    torch.device
        The module device
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _get_letter_tokens(tokenizer):
    """Helper to get token IDs for ``LETTERS``.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        LLM tokenizer

    Returns
    -------
    dict[str, int]
        Mapping of letter to token IDs

    Raises
    ------
    ValueError
        If a letter->token transformation yields more than one token
    """
    letter_tokens = {}
    for letter in LETTERS:
        text = f" {letter}"
        tid = tokenizer.encode(text, add_special_tokens=False)
        if len(tid) != 1:
            raise ValueError(f"' {letter}' is not a single token")

        letter_tokens[letter] = tid[0]

    return letter_tokens


def _shuffle_letters():
    """Shuffle genre/letter mapping.

    Shuffling this mapping guards against any tacit relationships a model may
    have formed between a letter and genre.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Mappings for letter/genre and genre/letter
    """
    shuffled = random.sample(GENRES, len(GENRES))

    letter_to_genre = dict(zip(LETTERS, shuffled))
    genre_to_letter = {
        genre: letter for letter, genre in letter_to_genre.items()
    }

    return letter_to_genre, genre_to_letter


def _build_chat_mcqa_prompt(story, letter_to_genre, tokenizer):
    """Build the chat-templated MCQA prompt and return the story's char span.

    The story is inserted as a sentinel, the full MCQA user turn is rendered
    through the chat template, then the sentinel's span is located and the real
    story is splicted back in.

    Parameters
    ----------
    story : str
        Story text
    letter_to_genre : dict[str, str]
        Letter to genre mapping
    tokenizer : transformers.AutoTokenizer
        Fast tokenizer with a chat template

    Returns
    -------
    tuple[str, int, int]
        Prompt, story_start_char, story_end_char

    Raises
    ------
    ValueError
        If STORY_SENTINEL is in the story
    """
    if STORY_SENTINEL in story:
        raise ValueError("Story text contains STORY_SENTINEL, can't extract")

    options = "\n".join(
        f"{letter}) {letter_to_genre[letter]}" for letter in LETTERS
    )

    user_content = USER_TEMPLATE.substitute(
        story=STORY_SENTINEL, options=options
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Find where the story will start/end in the story's character sequence
    story_start = rendered.index(STORY_SENTINEL)
    story_end = story_start + len(story)

    # Splice the story into the prompt
    prompt = rendered.replace(STORY_SENTINEL, story, 1)

    return prompt, story_start, story_end


def tokenize_prompt(tokenizer, story, letter_to_genre, device):
    """Build/tokenize the chat MCQA prompt and return a story-span token mask.

    Parameters
    ----------
    tokenizer : transformers.AutoTokenizer
        LLM tokenizer
    story : str
        Story text
    letter_to_genre : dict[str, str]
        Letter to genre mapping
    device : torch.device
        Model device

    Returns
    -------
    tuple[str, dict, list[tuple[int, int]], np.ndarray]
        The prompt, tokenized inputs, offsets, and story token mask

    Raises
    ------
    ValueError
        If the tokenizer isn't fast or the story mask is empty
    """
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for offset mappings")

    prompt, story_start, story_end = _build_chat_mcqa_prompt(
        story=story, letter_to_genre=letter_to_genre, tokenizer=tokenizer
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", return_offsets_mapping=True
    )

    # Get the offsets to construct our mask
    offsets = inputs.pop("offset_mapping")[0].tolist()

    story_token_mask = []
    for start, end in offsets:
        # Special tokens have offset (0, 0), so exclude zero-width spans
        overlaps_story = end > story_start and start < story_end and end > start
        story_token_mask.append(bool(overlaps_story))

    story_token_mask = np.array(story_token_mask, dtype=bool)

    if story_token_mask.sum() == 0:
        raise ValueError("Story span mask is empty; check the chat template")

    inputs = inputs.to(device)

    return prompt, inputs, offsets, story_token_mask


@contextmanager
def capture_layer_last_token_states(model):
    """Capture the final-token hidden state from each transformer layer.

    Parameters
    ----------
    model : transformers.AutoModelForCausalLM
        The LLM

    Yields
    ------
    list[torch.Tensor]
        One tensor per transformer layer, shape (batch_size, d_model)
    """
    hidden_states_by_layer = []
    hooks = []

    def hook_fn(module, inputs, output):
        """Capture the hidden states' last position."""
        hs = output[0] if isinstance(output, tuple) else output
        hidden_states_by_layer.append(hs[:, -1, :].detach())

    try:
        # Register each hook in the model
        for layer in model.model.layers:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        yield hidden_states_by_layer

    finally:
        # When the forward pass is totally done, remove the hooks
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def gather_layer_logits(
    letter_tokens,
    hidden_states_by_layer,
    outputs,
    unembedding,
    final_norm=None,
):
    """Get answer-letter logits from each captured layer plus the model's final
    logits.

    Parameters
    ----------
    letter_tokens : dict[str, int]
        Mapping from answer letters to token IDs
    hidden_states_by_layer : list[torch.Tensor]
        Hidden states captured from each transformer layer
    outputs : transformers.utils.ModelOutput
        HuggingFace model outputs from the forward pass
    unembedding : torch.Tensor
        The unembedding layer, usually model.lm_head.weight
    final_norm : torch.nn.Module or None
        Optional final norm

    Returns
    -------
    np.ndarray
        Layer logits, shape (n_layer + 1, len(LETTERS)). The last row is the
        actual final model logits
    """
    unembedding_device = unembedding.device
    final_norm_device = (
        _get_module_device(final_norm) if final_norm is not None else None
    )

    logits_by_layer = []
    letter_ids = torch.tensor(
        [letter_tokens[letter] for letter in LETTERS], dtype=torch.long
    )

    for hs in hidden_states_by_layer:
        # Normalize, if need
        if final_norm is not None:
            hs = hs.to(final_norm_device)
            hs = final_norm(hs)

        hs = hs.to(unembedding_device)

        # Shape: (batch, d_model) @ (d_model, vocab_size)
        layer_logits = hs @ unembedding.T

        letter_logits = layer_logits[0, letter_ids.to(layer_logits.device)]
        logits_by_layer.append(letter_logits.detach().cpu())

    # Get the final model logits
    final_logits = outputs.logits[0, -1, letter_ids.to(outputs.logits.device)]
    logits_by_layer.append(final_logits.detach().cpu())

    return torch.stack(logits_by_layer, dim=0).float().numpy()


def forward_with_embedding_grads(model, input_ids, attention_mask=None):
    """Run a forward pass using explicit input embeddings.

    This allows us to compute gradients of the answer objective with respect to
    each input token embedding. See ``get_token_gradients()``.

    Parameters
    ----------
    model : transformers.AutoModelForCausalLM
        The LLM
    input_ids : torch.Tensor
        Input IDs for the model
    attention_mask : torch.Tensor or None
        Attention mask

    Returns
    -------
    tuple[transformers.utils.ModelOutput, torch.Tensor]
        Model outputs and input embeddings
    """
    model.zero_grad(set_to_none=True)

    input_embeddings = model.get_input_embeddings()

    # Detach from the embedding table so gradients are collected on the
    # per-example input embeddings, not primarily on embedding weights
    inputs_embeds = input_embeddings(input_ids).detach()
    inputs_embeds.requires_grad_(True)
    inputs_embeds.retain_grad()

    forward_kwargs = {"inputs_embeds": inputs_embeds, "use_cache": False}
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask

    outputs = model(**forward_kwargs)

    return outputs, inputs_embeds


def get_token_gradients(
    correct_letter, letter_tokens, outputs, inputs_embeds, model=None
):
    """Compute token-level gradient norms.

    Objective: correct_answer_logit - logsumexp(other_answer_logits).

    Parameters
    ----------
    correct_letter : str
        Correct answer letter
    letter_tokens : dict[str, int]
        Mapping from answer letter to token ID
    outputs : transformers.utils.ModelOutput
        Model outputs
    inputs_embeds : torch.Tensor
        Input embeddings
    model : transformers.AutoModelForCausalLM or None
        (Optional) LLM. If provided, gradients are cleared before backward

    Returns
    -------
    np.ndarray
        Token gradients, shape (seq_len,). Each value is the gradient norm for
        one input token
    """
    if model is not None:
        model.zero_grad(set_to_none=True)

    if inputs_embeds.grad is not None:
        inputs_embeds.grad = None

    logits = outputs.logits[0, -1, :]

    # Get the logit for the correct answer, then the others
    correct_token_id = letter_tokens[correct_letter]
    correct_logit = logits[correct_token_id]

    other_logits = torch.stack(
        [
            logits[letter_tokens[letter]]
            for letter in LETTERS
            if letter != correct_letter
        ]
    )

    # Set up the objective and to a backward pass
    objective = correct_logit - torch.logsumexp(other_logits, dim=0)
    objective.backward()

    token_grad_norms = inputs_embeds.grad.detach().norm(dim=-1)[0]

    return token_grad_norms.float().cpu().numpy()


def process_story(
    story_id, text, true_genre, model, tokenizer, letter_tokens, device
):
    """Process one story through the chat-templated MCQA path.

    This function:
    1. Shuffles the answer-letter mapping
    2. Builds the chat-templated prompt
    3. Runs the model while capturing layer states
    4. Computes logit-lens answer logits
    5. Predicts the answer letter
    6. Computes token gradient attribution

    Parameters
    ----------
    story_id : int
        Story ID
    text : str
        Story text
    true_genre : str
        True genre
    model : transformers.AutoModelForCausalLM
        The LLM
    tokenizer : transformers.AutoTokenizer
        LLM tokenizer
    letter_tokens : dict[str, int]
        Token IDs for letters
    device : torch.device or str
        Model device

    Returns
    -------
    dict[str, int | str | np.ndarray]
        Story data
    """
    letter_to_genre, genre_to_letter = _shuffle_letters()
    correct_letter = genre_to_letter[true_genre]

    prompt, inputs, offsets, story_token_mask = tokenize_prompt(
        tokenizer=tokenizer,
        story=text,
        letter_to_genre=letter_to_genre,
        device=device,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    unembedding = model.lm_head.weight
    final_norm = getattr(model.model, "norm", None)

    with capture_layer_last_token_states(model) as hidden_states_by_layer:
        outputs, inputs_embeds = forward_with_embedding_grads(
            model=model, input_ids=input_ids, attention_mask=attention_mask
        )

    logits_by_layer = gather_layer_logits(
        letter_tokens=letter_tokens,
        hidden_states_by_layer=hidden_states_by_layer,
        outputs=outputs,
        unembedding=unembedding,
        final_norm=final_norm,
    )

    predicted_idx = int(logits_by_layer[-1].argmax())
    predicted_letter = LETTERS[predicted_idx]

    token_grads = get_token_gradients(
        correct_letter=correct_letter,
        letter_tokens=letter_tokens,
        outputs=outputs,
        inputs_embeds=inputs_embeds,
        model=model,
    )

    input_ids = input_ids[0].cpu().tolist()
    tokens = [tokenizer.decode(tid) for tid in input_ids]

    return MCQAResult(
        story_id=story_id,
        true_genre=true_genre,
        letter_to_genre=letter_to_genre,
        correct_letter=correct_letter,
        predicted_letter=predicted_letter,
        prompt=prompt,
        tokens=tokens,
        token_offsets=offsets,
        input_ids=input_ids,
        logits_by_layer=logits_by_layer.tolist(),
        token_grads=token_grads.tolist(),
        story_token_mask=story_token_mask.tolist(),
    )


def main(args):
    """Run the script."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the stories and drop the NULL genre, then sample
    df = pd.read_parquet(args.input)
    df = df[df["genre"] != "NULL"]
    df = df.groupby("genre").sample(n=args.n_per_genre, random_state=args.seed)
    df.reset_index(drop=True, inplace=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()
    print(f"Running {len(df)} stories through {args.model}...")

    device = model.get_input_embeddings().weight.device
    letter_tokens = _get_letter_tokens(tokenizer)

    results = []

    for n, row in enumerate(df.itertuples(index=True), start=1):
        result = process_story(
            story_id=int(row.Index),
            text=row.text,
            true_genre=row.genre,
            model=model,
            tokenizer=tokenizer,
            letter_tokens=letter_tokens,
            device=device,
        )

        results.append(asdict(result))

        if n % args.log_at == 0:
            print(f"Processed {n}/{len(df)} stories")

    output = pd.DataFrame(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"Saved chat-templated MCQA results to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple-choice genre classification data with logit "
            "lens and gradient attribution."
        )
    )
    parser.add_argument("input", type=Path, help="Input stories parquet")
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
        "--n-per-genre",
        type=int,
        default=100,
        help="Number of stories per genre",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=5167, help="Random seed"
    )
    parser.add_argument(
        "--log-at", type=int, default=25, help="How often to log progress"
    )
    args = parser.parse_args()
    main(args)
