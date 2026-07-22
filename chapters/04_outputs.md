---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Model Outputs

```{code-cell}
:tags: [remove-input]
from utils.plotting import FACET_ASPECT, FACET_HEIGHT, set_plot_theme

set_plot_theme()
```

```{code-cell}
:tags: [remove-output]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
```

This chapter discusses text generation and sampling methods. We'll look at the
raw scores from models, transform them into probability distributions, and
consider how to generate text on the basis of those distributions.

Below, we load GPT-2 and its tokenizer:

```{code-cell}
:tags: [remove-output]
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

We'll then run three more lines of code:

1. `torch.manual_seed()` helps ensure that later sampling steps produce
   consistent output 
1. `torch.set_grad_enabled(False)` allows us to call GPT-2 without wrapping
   each call in `torch.no_grad()`
1. `model.eval()` puts the model in evaluation mode

```{code-cell}
:tags: [remove-output]
torch.manual_seed(5167)
torch.set_grad_enabled(False)
model.eval()
```

Now, let's write out some text, tokenize it, and send it to the model.

```{code-cell}
text = "The work of art in the age of mechanical"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

## Next Token Prediction 

GPT-2 outputs **logits**: raw, unnormalized scores for each token in its
vocabulary. These scores indicate how strongly the model favors each token as a
possible next token, before the scores are converted into a probability
distribution.

```{code-cell}
logits = outputs.logits
print("Shape of logits:", list(logits.shape))
```

Take a look at the shape of these logits. For every input sequence, the model
assigns a vector of logits to each token position. The number of logit vectors
matches the length of the input sequence, and the length of each logit vector
matches the model's vocabulary size. If that weren't the case, the following
assertions would fail.

```{code-cell}
assert inputs["input_ids"].size(1) == logits.size(1), "Unmatched size"
assert model.config.vocab_size == logits.size(2), "Unmatched size"
```

```{note}
Where exactly do logits come from? After GPT-2 processes the input tokens, it
produces a contextual hidden vector for each hidden position. A final linear
layer, often called the **language modeling head** or informally the
**unembedding layer**, projects each hidden vector back into a vocabulary
space. That projection produces one raw score, or logit, for every token in the
vocabulary.

For GPT-2, this output layer is tied to the input embedding matrix: The same
learned weights used to map token IDs into vectors are also used to map hidden
vectors back to token scores. This is a common technique in model design.
```

So far, though, we do not have a newly generated token. Instead, we have
next-token information for every position in our input sequence. To predict the
token that should follow the full input sequence, we take the logit vector
corresponding to the final input token.

```{code-cell}
logits = logits[0, -1, :]
```

### Softmax

To express these logits as probabilities, we run them through **softmax**:

$$
\operatorname{softmax}(\mathbf{z})_i =
\frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
$$

Where:

- $e$ is the base of the natural logarithm
- $z_i$ is the logit (raw score) for token $i$
- $\sum_{j=1}^n e^{z_j}$ is the sum of the exponentials of all logits, which
  serves as a normalization factor
- $n$ is the total number of possible next tokens, usually the model's
  vocabulary size

Each softmax output falls between $0$ and $1$:

$$
0 \leq \operatorname{softmax}(\mathbf{z})_i \leq 1
$$

And all outputs sum to $1$:

$$
\sum_{i=1}^n \operatorname{softmax}(\mathbf{z})_i = 1
$$

Here's a toy example:

```{code-cell}
z = torch.tensor([1.25, -0.3, 0.87])
z -= z.max()  # In practice, we subtract max for numerical stability

exp = torch.exp(z)
sigma_z = exp / exp.sum()

sigma_z
```

Each of these scores is now a probability. Their sum will equal $1$.

```{code-cell}
print("Sum of sigma:", sigma_z.sum().item())
```

While we could wrap the above in a function, PyTorch can also do it for us.
We'll use the library's implementation of softmax from here on out.

```{code-cell}
probs = F.softmax(logits, dim=-1)
```

### The next token

With probabilities in hand, we can identify the most likely token using
**argmax**. This selects the highest value from `probs`.

```{code-cell}
idx = torch.argmax(probs).item()
token = tokenizer.decode(idx)
print("Most likely token:", repr(token))
```

We can also sort the probabilities and index them to get $k$ highest values.
`torch.sort()` returns values sorted in ascending/descending order along with a
tensor of indices that correspond to the original input.

```{code-cell}
k = 25
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
```

Index both tensors to get the top $k$ values.

```{code-cell}
sorted_probs = sorted_probs[:k]
sorted_indices = sorted_indices[:k]
```

Now we use the indices to get our tokens:

```{code-cell}
topk_tokens = []
for idx in sorted_indices:
    token = tokenizer.decode(idx.item())
    topk_tokens.append(repr(token))
```

Let's put these into a [pandas][pd] **DataFrame**. This is a two-dimensional
data structure, similar in nature to a table in a spreadsheet. DataFrames come
with a huge amount of functionality for data analysis, but in this chapter,
we'll use them primarily for formatting and inspecting outputs.

[pd]: https://pandas.pydata.org/

The easiest way to initialize a DataFrame is with a dictionary. Each key-value
pair becomes a column with a set of rows.

```{code-cell}
topk_df = pd.DataFrame(
    {
        "token": topk_tokens,
        "probability": sorted_probs.detach().cpu().numpy()
    }
)

topk_df
```

DataFrames play nicely with seaborn---in fact, the latter is built with them in
mind. Assign the column name `"token"` to `x` and `"probability"` to `y` to get
a rank plot of the top tokens.

```{code-cell}
fig, ax = plt.subplots()
sns.barplot(topk_df, x="token", y="probability", ax=ax)
ax.set(title="Most Probable Tokens", xlabel="Token", ylabel="Probability")
ax.tick_params(axis="x", rotation=90)
plt.show()
```

Let's also look at the overall probability distribution.

```{code-cell}
fig, ax = plt.subplots()
sns.histplot(
    probs.detach().cpu().numpy(),
    stat="density",
    element="step",
    log_scale=True,
    ax=ax
)
ax.set(title="Token Probability Distribution", xlabel="Probability (log)")
plt.show()
```

## Sampling Methods

So far, we've only inspected token probabilities. While these scores tell us
what's likely to come next, we must decide what token to select. This section
covers several methods of **sampling** from the token probabilities.

### Greedy decoding

**Greedy decoding** isn't really sampling at all. It takes the most likely
token every time, just as we did above. The function below implements this
logic.

```{code-cell}
def greedy_decoding(probs):
    """Perform greedy decoding.

    Parameters
    ----------
    probs : torch.Tensor, shape (vocab_size,)
        Token probabilities

    Returns
    -------
    int
        Selected token ID
    """
    idx = torch.argmax(probs)

    return idx.item()
```

Now run:

```{code-cell}
idx = greedy_decoding(probs)
token = tokenizer.decode(idx)
print(repr(text + token))
```

No matter how many times we run it, we get the same result:

```{code-cell}
for _ in range(5):
    idx = greedy_decoding(probs)
    token = tokenizer.decode(idx)
    print(repr(text + token))
```

### Multinomial sampling

**Multinomial sampling** treats the probabilities as exactly that: a
probability distribution to draw from. Instead of always taking the top token,
it samples in proportion to each token's probability. A token with probability
0.6 is chosen roughly 60% of the time, whereas one with 0.05 is chosen rarely.

```{code-cell}
def multinomial_sampling(probs):
    """Perform multinomial sampling.

    Parameters
    ----------
    probs : torch.Tensor, shape (vocab_size,)
        Token probabilities

    Returns
    -------
    int
        Sampled token ID
    """
    idx = torch.multinomial(probs, num_samples=1)

    return idx.item()
```

Whereas greedy decoding always produced the same result, multinomial sampling
does the opposite:

```{code-cell}
for _ in range(5):
    idx = multinomial_sampling(probs)
    token = tokenizer.decode(idx)
    print(repr(text + token))
```

### Top-k sampling

**Top-k sampling** limits the sampling pool to only the top $k$ most probable
tokens. This makes outputs more diverse than greedy decoding, but it guards
against the possibility of extremely rare tokens getting chosen. That can
always happen with standard multinomial sampling.

The function below implements a top-k filter for the probabilities.

```{code-cell}
def topk_filtering(probs, k=50):
    """Filter a probability distribution to only the top-k tokens.

    Parameters
    ----------
    probs : torch.Tensor, shape (vocab_size,)
        Token probabilities
    k : int
        Top token pool size

    Returns
    -------
    torch.Tensor, shape (vocab_size,)
        Filtered probabilities

    Raises
    ------
    ValueError
        If k is less than 0
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    k = min(k, probs.numel())
    values, indices = torch.topk(probs, k=k)

    # Initialize vector of 0 probabilities and assign values from probs to 
    # indices that correspond to the top-k values
    filtered = torch.zeros_like(probs)
    filtered[indices] = probs[indices]

    # Normalize probabilities
    filtered = filtered / filtered.sum()

    return filtered
```

First we filter our probabilities. Then we sample from the filtered
distribution.

```{code-cell}
topk_probs = topk_filtering(probs)

for _ in range(5):
    idx = multinomial_sampling(topk_probs)
    token = tokenizer.decode(idx)
    print(repr(text + token))
```

### Top-p sampling

**Top-p**, or **nucleus sampling**, limits the candidate pool by the cumulative
probability of tokens. Instead of fixing top tokens to a hard number, like with
top-k sampling, it dynamically selects the smallest set of tokens whose
probabilities sum to at least $p$. This set is called the **nucleus**.

Consider how this sampling method adapts to the model's underlying certainty.
When the model is very certain, the number of tokens that sum to $p$ will be
quite small because the model assigns high probabilities to only a few tokens.
As the model gets more uncertain, this number expands. This is because the
model assigns each token a lower probability, which in turn requires more
tokens to fulfill the $p$ constraint.

```{code-cell}
def topp_filtering(probs, p=0.9):
    """Filter a probability distribution using top-p/nucleus filtering.

    Parameters
    ----------
    probs : torch.Tensor, shape (vocab_size,)
        Token probabilities
    p : float
        Cumulative probability

    Returns
    -------
    torch.Tensor, shape (vocab_size,)
        Filtered probabilities

    Raises
    ------
    ValueError
        If p is less than or equal to 0, or greather than 1
    """
    if not 0 < p <= 1:
        raise ValueError("p must be in the interval (0, 1]")

    values, indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(values, dim=0)

    # Index of the first token where cumulative probability >= p. Add 1 to keep
    # tokens up to and including the cutoff
    p = torch.tensor(p, device=probs.device, dtype=probs.dtype)
    cutoff = torch.searchsorted(cumulative, p).item() + 1
    candidate_indices = indices[:cutoff]

    # Normalize probabilities
    filtered = torch.zeros_like(probs)
    filtered[candidate_indices] = probs[candidate_indices]

    filtered = filtered / filtered.sum()

    return filtered
```

As before, we filter our probabilities, then run.

```{code-cell}
topp_probs = topp_filtering(probs)

for _ in range(5):
    idx = multinomial_sampling(topp_probs)
    token = tokenizer.decode(idx)
    print(repr(text + token))
```

### Temperature

Temperature scaling reshapes the underlying probability distribution before any
token is selected. It's applied directly to model logits $z$:

$$
\frac{z}{t}
$$

Where $t$ is a scalar:

$$
0 \lt t \lt \infty
$$

Lower temperatures ($t \lt 1$) make model outputs more deterministic (closer to
greedy decoding) by sharpening the probability distribution, while higher
temperatures ($t \gt 1$) make model outputs more random by flattening the
distribution.

The intuition for temperature is as follows. Softmax only cares about
_differences_ between logits:

$$
\operatorname{softmax}\left(\frac{z}{t}\right)
$$

- Dividing by small $t$ magnifies the gaps
- Dividing by large $t$ shrinks them, making tokens more equal
- $t = 1$ keeps the gaps unchanged

We implement temperature scaling below.

```{code-cell}
def temperature_scaling(logits, temperature=1.0):
    """Perform temperature scaling.

    Parameters
    ----------
    logits : torch.Tensor
        Output logits, shape (vocab_size,)
    temperature : float
        Scaling factor. Higher values make the distribution flatter; lower
        values make it sharper

    Returns
    -------
    torch.Tensor, shape (vocab_size,)
        Probabilities after temperature scaling

    Raises
    ------
    ValueError
        If temperature is below 0
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)

    return probs
```

Let's scale our logits and use multinomial sampling.

```{code-cell}
temp_probs = temperature_scaling(logits, temperature=1.5)

for _ in range(5):
    idx = multinomial_sampling(temp_probs)
    token = tokenizer.decode(idx)
    print(repr(text + token))
```

We can see the effects of temperature scaling by inspecting a few different
values for $t$. Below, we get probability values for the top-500 tokens under
different temperature settings. 

```{code-cell}
temperatures = [0.5, 1.0, 1.5]
top_n = 500 

rows = []
for temperature in temperatures:
    temp_probs = temperature_scaling(logits, temperature=temperature)
    sorted_probs, _ = torch.sort(temp_probs, descending=True)
    top_probs = sorted_probs[:top_n]
    
    for rank, prob in enumerate(top_probs, start=1):
        rows.append(
            {
                "temperature": "temp=" + str(temperature),
                "rank": rank,
                "prob": prob.item()
            }
        )
```

Let's format as a DataFrame.

```{note}
Here we initialize a Dataframe with a list of dictionaries. As long as all the
dictionaries have the same keys, pandas will assign their values to the right
columns.
```

```{code-cell}
temp_df = pd.DataFrame(rows)
```

And plot:

```{code-cell}
fig, ax = plt.subplots()
sns.lineplot(temp_df, x="rank", y="prob", hue="temperature", ax=ax)
ax.set_yscale("log")
ax.set(
    title="Token Probabilities by Temperature",
    xlabel="Token rank",
    ylabel="Probability",
)
ax.legend(title="Temperature")
plt.show()
```

### Combining methods

Sampling methods may be combined, and indeed they usually are. The function
below reshapes token probabilities using all the methods we've discussed. It
scales by temperature, performs greedy decoding if asked, and then runs
top-k/-p (in that order!).

```{code-cell}
def get_sampling_distribution(
    logits, greedy=False, temperature=1.0, k=None, p=None
):
    """Get a sampling distribution.

    Parameters
    ----------
    logits : torch.Tensor
        Output logits, shape (vocab_size,)
    greedy : bool
        Whether to use greedy decoding
    temperature : float
        Temperature scaling factor
    k : int or None
        Top-k filtering pool size
    p : float or None
        Top-p filtering cumulative probability

    Returns
    -------
    torch.Tensor
        Token probabilities, shape (vocab_size,)
    """
    probs = temperature_scaling(logits, temperature=temperature)

    if greedy:
        idx = greedy_decoding(probs)
        probs = torch.zeros_like(probs)
        probs[idx] = 1.0

        return probs

    if k is not None:
        probs = topk_filtering(probs, k=k)

    if p is not None:
        probs = topp_filtering(probs, p=p)

    return probs
```

Let's run this function across several configurations. We'll store these
configurations in a dictionary of dictionaries. The keys in the outer
dictionary are string representations of the parameters set by the inner
dictionaries.

```{code-cell}
configs = {
    "greedy": dict(greedy=True),
    "multinomial": dict(),
    "temp=0.7": dict(temperature=0.7),
    "temp=1.3": dict(temperature=1.3),
    "top-k=50": dict(k=50),
    "top-p=0.9": dict(p=0.90),
    "top-k=50, top-p=0.9": dict(k=50, p=0.9),
    "temp=0.7, top-p=0.9": dict(temperature=0.7, p=0.9),
    "temp=1.3, top-p=0.9": dict(temperature=1.3, p=0.9),
}
```

```{tip}
You can create dictionaries using `dict()` instead of the curly brace `{ }`
syntax. Separate keys and values with `=`.
```

Now, for each configuration, we call `get_sampling_distribution()` by sending
it the logits and arguments for `greedy`, `temperature`, `k`, and `p`. Use
`.items()` to iterate through the keys and values of a dictionary
simultaneously.

```{code-cell}
for name, config in configs.items():
    sampling_probs = get_sampling_distribution(logits, **config)
    idx = multinomial_sampling(sampling_probs)
    token = tokenizer.decode(idx)

    print("Config:", name)
    print("  Text:", repr(text + token), flush=True)
```

## Text Generation

We'll get a better sense of the differences between sampling methods if we
generate more than one token. To do that, we'll need to write a **generation
loop**, which dynamically constructs an input sequence for the model using
newly sampled tokens.

### The generation loop

The following code block implements this logic. It mirrors what we did above
with `get_sampling_distribution()`, except it adds a generation step with an
inner for-loop. For each configuration, it:

1. Tokenize `text` and copy its `input_ids` to a new variable, `seq`
1. Send `seq` to the model and get the final logit vector from `seq_outputs`
1. Sample from `seq_logits` to get the next token ID, `idx`
1. Append `idx` to `seq` and repeat the process with the newly extended
   sequence

```{code-cell}
new_tokens = 10

for name, config in configs.items():
    inputs = tokenizer(text, return_tensors="pt")
    seq = inputs["input_ids"]

    for _ in range(new_tokens):
        seq_outputs = model(input_ids=seq)

        seq_logits = seq_outputs.logits[0, -1, :]
        sampling_probs = get_sampling_distribution(seq_logits, **config)
        idx = multinomial_sampling(sampling_probs)

        # Append the sampled token ID, shape (1, seq_len)
        next_token = torch.tensor([[idx]], device=seq.device)
        seq = torch.cat([seq, next_token], dim=1) 

    generated = tokenizer.decode(seq[0])

    print("Config:", name)
    print("  Text:", repr(generated), flush=True)
```

This is the core generation logic we saw back in Chapter 1 when we used
`pipeline`.

### Early stopping

In theory, text generation would continue indefinitely if we didn't set a value
for `new_tokens`. But there's another popular stopping condition, which makes
use of a model's special "end-of-sequence" (EOS) token:

```{code-cell}
tokenizer.eos_token
```

In training, this token marks the end of a sequence, such as a document or
another chunk of text defined by the model builders. Since this token appears
in training data, the model learns how to use it just like any other token.
This also means that our sampling method can select this token during
generation.

We can thus leverage EOS tokens to implement **early stopping**: when our
sampling method picks this token, we `break` our loop and stop generation.
Let's see how this works with the following settings:

```{code-cell}
text = "News at 5."
max_new_tokens = 100
eos_token_id = tokenizer.eos_token_id
```

Now we run the loop as before but add a check before extending our sequences.
And, instead of printing the results, we'll store them for a DataFrame.

```{code-cell}
eos_results = []
for name, config in configs.items():
    inputs = tokenizer(text, return_tensors="pt")
    seq = inputs["input_ids"]

    stopped_early = False

    for _ in range(max_new_tokens):
        seq_outputs = model(input_ids=seq)

        seq_logits = seq_outputs.logits[0, -1, :]
        sampling_probs = get_sampling_distribution(seq_logits, **config)
        idx = multinomial_sampling(sampling_probs)

        # Is the sampled token our EOS token? If so, stop
        if idx == eos_token_id:
            stopped_early = True
            break

        # Append the sampled token ID, shape (1, seq_len)
        next_token = torch.tensor([[idx]], device=seq.device)
        seq = torch.cat([seq, next_token], dim=1) 

    generated = tokenizer.decode(seq[0])

    eos_results.append(
        {
            "config": name,
            "stopped_early": stopped_early,
            "text": generated, 
        }
    )
```

How did the different configurations behave under early stopping?

```{code-cell}
eos_df = pd.DataFrame(eos_results)
eos_df
```

## Comparing Sampling Methods

The effects of different sampling methods are often apparent in output text.
For example, greedy decoding often produces an excess of repetitions, and high
temperature generates erratic sequences (`"News at 5. 5. A giant salt rabbit
aggregates..."`). But between these extremes, it can be difficult to discern
the effect sampling methods have on output. In this section, we'll look at a
few empirical ways to compare them.

### Distribution statistics

First, let's look at how sampling methods shape the underlying token
probability distribution. For each sampling configuration, we'll derive the
following statistics from the output of `get_sampling_distribution()`.

| Statistic         | Description                                            | Answers                                   |
|-------------------|--------------------------------------------------------|-------------------------------------------|
| `support_size`    | Number of candidate tokens that could be sampled       | How many tokens survive filtering?        |
| `entropy`         | A measure of uncertainty                               | How spread out is the distribution?       |
| `effective_vocab` | Exponentiation of entropy                              | How many equally likely tokens would produce the same uncertainty? |
| `max_prob`        | Maximum probability among candidate tokens             | How confident is the top choice?          |
| `expected_rank`   | Probability-weighted average rank of the sampled token | On average, what rank token do we sample? |

```{code-cell}
def distribution_stats(probs):
    """Derive statistics from a probability distribution.

    Parameters
    ----------
    probs : torch.Tensor
        Token probabilities, shape (vocab_size,)

    Returns
    -------
    dict
        Statistics for support size, entropy, effective vocab, max probability,
        and expected rank
    """
    nonzero = probs > 0
    p_nonzero = probs[nonzero]

    entropy = -(p_nonzero * torch.log(p_nonzero)).sum()
    effective_vocab = torch.exp(entropy)

    sorted_probs, _ = torch.sort(probs, descending=True)
    ranks = torch.arange(1, probs.numel() + 1, device=probs.device)
    expected_rank = (sorted_probs * ranks).sum()

    return {
        "support_size": nonzero.sum().item(),
        "entropy": entropy.item(),
        "effective_vocab": effective_vocab.item(),
        "max_prob": probs.max().item(),
        "expected_rank": expected_rank.item(),
    }
```

Let's get statistics for each item in `configs`.

```{code-cell}
dist_results = []
for name, kwargs in configs.items():
    probs = get_sampling_distribution(logits, **kwargs)
    stats = distribution_stats(probs)

    # Add a dictionary for each statistic
    for metric, value in stats.items():
        dist_results.append(
            {
                "method": name,
                "metric": metric,
                "value": value,
            }
        )
```

We reformat into a DataFrame and plot.

```{code-cell}
dist_df = pd.DataFrame(dist_results)

g = sns.catplot(
    data=dist_df,
    x="method",
    y="value",
    col="metric",
    col_wrap=2,
    kind="bar",
    sharey=False,
    height=FACET_HEIGHT,
    aspect=FACET_ASPECT,
)

titles = {
    "support_size": "Support size (log tokens)",
    "entropy": "Entropy (nats)",
    "effective_vocab": "Effective vocab (tokens)",
    "max_prob": "Max probability (0-1)",
    "expected_rank": "Expected rank (position)",
}

g.set_titles("{col_name}")
for ax in g.axes.flat:
    metric = ax.get_title()
    if metric == "support_size":
        ax.set_yscale("log")  # Use log scale since support_size can be huge

    ax.set_title(titles.get(metric, metric))

g.set_axis_labels("", "Value")
g.tick_params(axis="x", rotation=90)
g.fig.suptitle("Distribution Statistics for Different Sampling Methods", y=1.05)
plt.show()
```

Some observations:

- Greedy decoding is an outlier, sitting at the extreme of every metric: a
  single candidate token, zero entropy, and a maximum probability of 1.0
- Multinomial sampling and temperature scaling keep the entire vocabulary but
  the latter reshapes token probabilities, sometimes drastically. We see that
  reshaping reflected in entropy, effective vocab, and expected rank
- When temperature is high, the distribution flattens and the model reaches
  deep into its vocabulary, with expected rank sometimes climbing into the
  thousands
- Filtering methods (top-k and top-p) cut the support size dramatically while
  keeping entropy low, concentrating probability on a handful of tokens. That
  said, a generous top-p can retain meaningful uncertainty, while a tight top-k
  sharply concentrates probability

### Perplexity

**Perplexity** measures how well a model predicts a sequence of text. It
captures how "surprised" the model is by each token: low perplexity means the
model found the text predictable, while high perplexity means it was caught off
guard.

Perplexity is the exponentiated average negative log-likelihood of a sequence.
For a sequence of $N$ tokens, perplexity is often written as:

$$
\operatorname{PPL} = \exp 
\left( -\frac{1}{N} \sum_{i=1}^{N} 
\log p(x_i \mid x_{<i})
\right)
$$

Where:

- $p(x_i \mid x_{<i})$ is the probability the model assigns to token $x_i$
  given all preceding tokens
- $N$ is the number of tokens in the sequence

In practice, causal language models usually compute this loss by predicting
each token from the tokens before it, so the first token is often ignored in
the average.

The measure itself is similar to entropy. In fact, perplexity is simply the
exponentiated cross-entropy. Where effective vocabulary answered "How many
equally likely tokens match this uncertainty?" perplexity asks the same
question across a whole _sequence_: On average, how many tokens was the model
choosing between at each step?

We can get the cross-entropy of a sequence by furnishing a model with labels:
token IDs for the sequence.

```{code-cell}
:tags: [remove-output]
text = "The work of art in the age of mechanical reproduction"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```

This value is stored under `.loss`. Use `torch.exp()` to compute perplexity.

```{code-cell}
loss = outputs.loss
print("Cross-entropy:", loss.item())
print("Perplexity:", torch.exp(loss).item())
```

Remember, the above value tells us how surprised a model was on average. If we
calculate perplexity on our generated sequences stored in `eos_df`, we'll get
another view of the effect of sampling methods: a view that identifies model
surprise. Below, we iterate through every row in the `"text"` column of
`eos_df` by indexing the DataFrame, much like indexing a dictionary by key.

```{code-cell}
perp_results = []
for seq in eos_df["text"]:
    inputs = tokenizer(seq, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])

    perp = torch.exp(outputs.loss).item()
    perp_results.append(perp)
```

Now, we **assign** a new column to `eos_df` using the list `perp_results`:

```{code-cell}
eos_df["perplexity"] = perp_results
```

Use `.sort_values()` to sort the rows by a column: 

```{code-cell}
eos_df.sort_values("perplexity", ascending=False)
```

Note that low perplexity doesn't necessarily mean "better text." Here, the
model is unsurprised by its own output when the sampling configuration is
itself conservative. For this kind of analysis, you can think of perplexity as
a measure of how far a sampling configuration moves a model away from its
baseline expectations. Indeed, the sequence with the lowest perplexity reads
almost like a direct scrape from the web.

```{code-cell}
lowest_perp = eos_df.nsmallest(1, "perplexity")
print(lowest_perp["text"].item())
```

On the other end, the sequence with the highest perplexity is totally chaotic.

```{code-cell}
highest_perp = eos_df.nlargest(1, "perplexity")
print(highest_perp["text"].item())
```
