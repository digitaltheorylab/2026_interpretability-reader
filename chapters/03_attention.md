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

# Attention and Contextual Embeddings

```{code-cell}
:tags: [remove-input]
from utils.plotting import SQUARE_FIGSIZE, set_plot_theme

set_plot_theme()
```

```{code-cell}
:tags: [remove-output]
import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
```

In the last chapter, we saw how LLMs use embedding vectors to represent tokens.
So far, these vectors have been **static**: We used the same vector for all
instances of a token, no matter the context. But Transformer-based language
models are powerful because they represent contextual information. That is,
they use **dynamic** embeddings that reflect specific sequences of tokens. This
chapter walks through that process.

As before, we load GPT-2 and its tokenizer. Note the small change, though: We
initialize with `attn_implementation="eager"` to tell the model to store some
key information we'll use later on.

```{code-cell}
:tags: [remove-output]
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, attn_implementation="eager"
)
model.eval()
```

Let's also get our embedding layer:

```{code-cell}
embedding_layer = model.get_input_embeddings()
emb = embedding_layer.weight.detach().cpu().to(torch.float64)
```

...and encode some text.

```{code-cell}
text = "Then I tried to find some way of embracing my mother's ghost."
encoded = tokenizer.encode(text)
seq_emb = emb[encoded]
```

One last thing: We will initialize some configuration settings for our plots
using a Python **dictionary**. Dictionaries store unique elements, but they
associate those elements with a particular value. These can be individual
values, like numbers, or containers, like lists. Every element in a dictionary
is a **key-value** pair. This makes dictionaries powerful data structures for
associating values in data with metadata of one kind or another.

In our case, we create a dictionary of plotting arguments, which we'll send to
seaborn. This reduces code duplication. To make this dictionary, use curly
brackets `{ }` and colons `:` to separate the key-value pairs:

```{code-cell}
plot_config = {
    "cmap": "Blues",
    "robust": True,
    "annot": True,
    "fmt": ".1f",
}
```

Dictionaries, like lists, can be indexed. But instead of using the element
position as an index, you use a dictionary key in combination with square
brackets `[ ]`. When you index a dictionary in this way, you return the value
associated with that key:

```{code-cell}
print("Value associated with 'cmap':", plot_config["cmap"])
```

We'll discuss iteration with dictionaries later on. First, it's time to talk
context.

## (Simplified) Attention

Internally, LLMs build contextual representations through **attention**. In the
classic attention layer, each token's representation is updated by comparing it
to the representations of other tokens in the sequence. The basic operation is
called **scaled dot-product attention**:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = 
\operatorname{softmax}\left(
\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} 
\right)\mathbf{V}
$$

Where:

- $\mathbf{Q}$ = queries
- $\mathbf{K}$ = keys
- $\mathbf{V}$ = values
- $\mathbf{Q}\mathbf{K}^\top$ gives the raw attention scores via dot products
  between the queries and keys
- $\sqrt{d_k}$ is the scaling factor, where $d_k$ is the dimensionality of
  $\mathbf{K}$
- $\operatorname{softmax}(\ldots)$ turns each row of scores into attention
  weights
- Multiplying those weights by $\mathbf{V}$ produces weighted combinations of
  the value vectors

### Implementing attention

We will implement this by writing our own function. So far we have relied on
external functions, but we can also write our own. Before doing so, let's
review the vocabulary associated with functions:

- The placeholder variables are **parameters**
- **Arguments** are the values assigned to parameters during a call
- To **call** a function means using it to compute something
- The **body** is the code inside a function
- A function's **scope** is the local context in which it runs code
- The **return value** is the output of a function

In Python, a function begins with the `def` keyword, followed by:

- The name of the function, which follows the conventions of variables
- Parameters surrounded by parentheses `( )` and separated by commas `,`
- A colon `:`

There is no practical limit to the number of parameters. Code in the body of
the function should be indented according to the same conventions for loops and
conditionals (four spaces). To return a result from the function, use the
`return` keyword.

It's also a good idea to document your function so you know what the inputs and
outputs mean. You do this with **docstrings**, a special string at the start of
a function that begins/ends with three quotes `"""`. Docstrings should have:

- A one-sentence description of what the function does
- Descriptions for each parameter, with optional data type annotations
- Description of the return value

```{note} Docstring conventions
There are several different ways to write docstrings. The function below uses
conventions from the the [NumPy style guide][np].

[np]: https://numpydoc.readthedocs.io/en/latest/format.html
```

Let's implement!

```{code-cell}
def attention(Q, K, V, mask=None):
    """Calculate scaled dot-product attention.

    Parameters
    ----------
    Q : torch.Tensor
        Query matrix, shape (n_token, n_dim)
    K : torch.Tensor
        Key matrix, shape (n_token, n_dim)
    V : torch.Tensor
        Value matrix, shape (n_token, n_dim)
    mask : torch.Tensor or None
        A triangular masking matrix

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Word embeddings weighted by attention and the attention weights
    """
    # 1. Compare every query with every key using dot(Q, K)
    scores = Q @ K.T

    # 2. Scale the scores by K's dimension size so they do not get too large
    dim_k = K.shape[1]
    scores = scores / math.sqrt(dim_k)

    # 3. Optionally mask scores to the right
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 4. Normalize scores into attention weights via softmax 
    weights = F.softmax(scores, dim=-1)

    # 5. Compute outputs as weighted sums of values using dot(weights, V)
    output = weights @ V

    return output, weights
```

Since we wrote a docstring, we can use `help()` to read our function's
documentation.

```{code-cell}
help(attention)
```

Now, let's make our `Q`, `K`, and `V`, matrices. Below, they're all the same
thing: copies of the sequence embeddings. Once we've assigned those embeddings
to each variable, we can run `attention()`:

```{code-cell}
Q = K = V = seq_emb
output, weights = attention(Q, K, V)
```

Observe the shapes of inputs to and outputs from attention. Each of the input
matrices is the same shape as our embeddings: $(t, d)$, where $t$ is the number
of tokens in the input and $d$ is the vector dimension. So, too, is `output`:
It represents a new version of our embeddings, which has had attention applied
to it. But `weights` is different: It's a square matrix with the shape $(t,
t)$.

```{code-cell}
print("Q/K/V shape:", list(Q.shape))
print("Attention embedding shape:", list(output.shape))
print("Attention weights shape:", list(weights.shape))
```

Attention weights tell you how much a given token attends to other tokens in
the input sequence (including itself). We'll construct a heatmap below to
visualize these relationships.

First, though, let's make some labels. These are our input tokens. We'll decode
the token IDs with the tokenizer and add them to `plot_config` as new keys. Our
heatmap will reference them that way.

```{code-cell}
labels = []
for token in encoded:
    decoded = tokenizer.decode(token)
    decoded = repr(decoded)
    labels.append(decoded)

plot_config["xticklabels"] = labels
plot_config["yticklabels"] = labels
```

Now we plot.

```{code-cell}
fig, ax = plt.subplots()
sns.heatmap(weights, ax=ax, **plot_config)
ax.set(title="Attention Weights")
plt.show()
```

Some models, like BERT, allow for these kinds of representations. But see how
the upper triangle of the heatmap has values? Those represent attention weights
between tokens that come _after_ a given token. This doesn't make sense for
**autoregressive** models like GPT-2, which move from left to right. Such
models shouldn't have access to future tokens since the whole point of text
generation is that a model itself learns what comes next!

We can control for this by constructing an **attention mask**. This sets future
token scores to `-inf` before softmax (step 3 in our function). After softmax,
those positions become attention weight `0`. In effect, the attention mask
removes a model's ability to look into the future.

```{code-cell}
N = len(encoded)
mask = torch.tril(torch.ones(N, N, device=seq_emb.device))
print(mask)
```

Let's rerun attention with `mask`.

```{code-cell}
output, weights = attention(Q, K, V, mask=mask)

fig, ax = plt.subplots()
sns.heatmap(weights, **plot_config, ax=ax)
ax.set(title="Attention Weights with Masking")
plt.show()
```

See how the upper triangle never gets weights?

### Using query and key projections 

In reality, no model uses the raw embeddings for attention, as we've done
above. They usually put embeddings through a **linear transform** layer before
running attention:

```{code-cell}
linear_layer = nn.Linear(in_features=768, out_features=768, bias=True)
linear_layer
```

This layer is trained like other components of a model, though. That means we
can't run our embeddings through `linear_layer` and replicate what GPT-2 does
(`linear_layer`'s weights and biases are randomized before being trained). But
we can do something else instead: fit a PCA to the model's embeddings and
project our sequence embeddings through that. This simulates the work of the
linear layer---albeit in a very rough way. Later on in the chapter we will use
the actual trained layer(s) GPT-2 uses to run attention.

As with `attention()`, we implement our own embeddings projector as a function.
The code below mostly replicates the dimensionality reduction logic in the last
chapter, so you'll likely recognize it.

```{code-cell}
def project_embeddings(emb, seq_emb, n_components=64):
    """Project embeddings into a smaller PCA space.

    Parameters
    ----------
    emb : torch.Tensor
        The full embedding matrix, shape (vocab_size, n_dim)
    seq_emb : torch.Tensor
        The sequence embeddings, shape (n_token, n_dim)
    n_components : int
        Number of PCA dimensions to keep

    Returns
    -------
    torch.Tensor
        PCA-projected sequence embeddings, shape (n_token, n_components)
    """
    # Learn PCA directions from the full vocabulary
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized")
    pca.fit(emb.numpy())

    # Project `seq_emb` into the PCA space
    projected = pca.transform(seq_emb.numpy())

    # Convert back to a PyTorch tensor
    projected = torch.tensor(projected, dtype=seq_emb.dtype, device=emb.device)

    return projected
```

Now, we project `seq_emb` onto the PCA space learned from `emb`:

```{code-cell}
seq_proj = project_embeddings(emb, seq_emb, n_components=64)
```

Assign `seq_proj` to `Q` and `K`. But note that `V` is still a clone of
`seq_emb`. This is because it's the output matrix that we modify with
attention.

```{code-cell}
Q = K = seq_proj
V = seq_emb
output, weights = attention(Q, K, V, mask=mask)

print("Q/K shape:", list(Q.shape))
print("V shape:", list(V.shape))
print("Attention embedding shape:", list(output.shape))
print("Attention weights shape:", list(weights.shape))
```

As before, we plot with a heatmap. Note one other piece: `zero_mask`, which
tells seaborn to refrain from plotting values in the upper triangle (they're
zeroed-out anyway).

```{code-cell}
zero_mask = (weights == 0).numpy()

fig, ax = plt.subplots()
sns.heatmap(weights, mask=zero_mask, ax=ax, **plot_config)
ax.set(title="Attention Weights with PCA-Projected Queries and Keys")
plt.show()
```

### Attention Variants

There are actually several different kinds of attention. We overview the main
versions below.

**Self-attention** means that each token in an input sequence is compared with
every other token. This enables the model to capture relationships across the
entire input sequence. (We saw this above---it's the first version of attention
we ran.)

```{code-cell}
Q = K = V = seq_emb
output, weights = attention(Q, K, V)
```

**Causal self-attention** adds a constraint: tokens can only attend to prior
tokens rather than being allowed to look ahead. (We've also seen this!)

```{code-cell}
Q = K = V = seq_emb
mask = torch.tril(torch.ones(N, N, device=seq_emb.device))
output, weights = attention(Q, K, V, mask=mask)
```

**Cross-attention** uses one set of embeddings for queries and another for keys
and values. Encoder/decoder models will use this for translation tasks, where
queries are one language and keys/values another. Similarly, certain image/text
models use cross-attention to project image and text embeddings into a shared
vector space.

We simulate the external embeddings (query matrix) with random ones below:

```{code-cell}
Q = torch.rand_like(seq_emb, device=seq_emb.device)
K = V = seq_emb
output, weights = attention(Q, K, V)
```

**Multi-head attention** involves using multiple attention mechanisms, or
**heads**, in parallel, which are then concatenated together when they are
passed elsewhere in the network. During training, each head learns to focus on
different kinds of relationships in input sequences.

This is what a head split looks like:

```{code-cell}
n_head = 12
n_token, n_dim = seq_emb.shape
head_dim = n_dim // n_head
heads = seq_emb.view(n_token, n_head, head_dim).transpose(0, 1)

print("Shape of each head:", list(heads[0].shape))
```

From here, you perform attention for each one and concatenate them all:

```{code-cell}
head_outputs = []
for idx in range(n_head):
    current_head = heads[idx]
    Q = K = V = current_head

    output, weights = attention(Q, K, V)
    head_outputs.append(output)

multihead_embeddings = torch.cat(head_outputs, dim=-1)
```

Note that PyTorch has functionality for this kind of operation (and many other
attention variants). We can construct an multi-head attention layer:

```{code-cell}
multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12)
multihead_attn
```

And send our embeddings through it, as above:

```{code-cell}
Q = K = V = seq_emb.to(torch.float32)
output, weights = multihead_attn(Q, K, V)
```

## (Actual) Attention

Now that you have a sense of how attention works in principle, let's see it in
action with the model's trained representations. First, let's tokenize our
text. We'll use a slightly different method this time so that the tokenizer
returns tensors.

```{code-cell}
inputs = tokenizer(text, return_tensors="pt").to(model.device)
inputs
```

Well, technically it returns dictionaries of tensors. Here are the input IDs:

```{code-cell}
inputs["input_ids"]
```

And here's the attention mask:

```{code-cell}
inputs["attention_mask"]
```

````{note}
Note the shape of our inputs:

```{code-cell}
print("Shape of input IDs:", list(inputs["input_ids"].shape))
```

The tokenizer has added a **batch dimension**, which represents the number of
sequences in the input. Right now, that's only `1`, but models can encode
several sequences in parallel.
````

With our text tokenized, we can send the inputs to the model. Because we are
only inspecting the model rather than training it, we wrap the call in
`torch.no_grad()`. This tells PyTorch not to store the extra information the
model would need to update its weights during training.

```{code-cell}
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_attentions=True,
    )
```

Since `inputs` is a dictionary, we can also use some special notation to assign
arguments automatically. This is called **unpacking**. It works with lists and
dictionaries alike (as well as other data structures in Python). Below, we
unpack the contents of `inputs` with the double star operator `**`. This
automatically routes our two keys, `"input_ids"` and `"attention_mask"` to
their proper parameters in `model(...)`.

```{code-cell}
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
```

```{note}
Sequences like lists are unpacked with a single star `*`.
```

### Extracting attention weights

Attention weights are stored under `.attentions`. The shape of each weight
tensor is as follows: batch size, number of heads, number of tokens, number of
tokens.

```{code-cell}
print("Number of weight tensor:", len(outputs.attentions))
print("Shape of a weight tensor:", list(outputs.attentions[0].shape))
```

Let's look at the attention relationships across our sequence. Unlike above,
though, we actually have a dozen separate tensors to choose from: one for every
layer in the model. We'll look at each of these, but first we need to do some
formatting. The code block below removes the batch dimension from each tensor,
takes the average across attention heads, and converts the result to a NumPy
array.

```{code-cell}
attentions = []
for tensor in outputs.attentions:
    tensor = tensor.squeeze(0)
    weights = tensor.mean(dim=0)
    weights = weights.detach().cpu().numpy()

    attentions.append(weights)

print("Shape of an attention matrix:", attentions[0].shape)
```

We now build a big heatmap. The `enumerate()` function returns an index
position in addition to the weights stored under `attentions`. We use this to
tell Matplotlib where it should place the layer's heatmap by indexing an `axes`
array.

```{code-cell}
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
axes = axes.flatten()

for layer_idx, weights in enumerate(attentions):
    ax = axes[layer_idx]
    sns.heatmap(
        weights,
        mask=zero_mask,
        ax=ax,
        cmap="Blues",
        robust=True,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Layer " + str(layer_idx))

plt.suptitle("Layerwise Attention Weights")
plt.show()
```

Layer `2` seems to feature a particularly strong set of attention weights
between `" my"` and `" mother's ghost."` Let's look at it in detail.

```{code-cell}
fig, ax = plt.subplots()
sns.heatmap(attentions[2], mask=zero_mask, ax=ax, **plot_config)
ax.set(title="Attention Weights for Layer 2")
plt.show()
```

Remember that we averaged over the attention heads. But perhaps one or two of
these heads is particularly sensitive to this dependency relation. Let's look.
First, we extract all the heads at layer `2`.

```{code-cell}
layer_2_attn = outputs.attentions[2].squeeze(0)
layer_2_heads = []

for head in layer_2_attn:
    weights = head.detach().cpu().numpy()
    layer_2_heads.append(weights)
```

Then plot:

```{code-cell}
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
axes = axes.flatten()

for head_idx, head in enumerate(layer_2_heads):
    ax = axes[head_idx]
    sns.heatmap(
        head,
        mask=zero_mask,
        ax=ax,
        cmap="Blues",
        robust=True,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Head " + str(head_idx))

plt.suptitle("Attention Heads at Layer 2")
plt.show()
```

Heads `3`, `4`, and `9` assign relatively high attention weights among tokens
in the phrase `" my mother's ghost."` Heads `5` and `8` show stronger weights
around `" embracing"` and the following noun phrase, while head `7` spreads
attention across much of the phrase.

Following these descriptive observations, you might imagine how you could
construct a more systematic analysis. Such an analysis would ask: Are these
heads at this layer mostly responsible for representing possessive
relationships in the model?

### Token-to-token relationships

Let's look at some more granular relationships instead of the all-to-all
comparisons of our heatmap. For example, we can ask: for each token, which
previous token does it attend to most?

The function bellow takes one attention matrix and prints the strongest
token-to-token link for each token.

```{code-cell}
def strongest_previous_token(weights, labels, exclude_self=True, ignore=None):
    """Print the strongest previous-token attention link for each token.

    Parameters
    ----------
    weights : np.ndarray
        Attention matrix, shape (n_token, n_token)
    labels : list[str]
        Token labels
    exclude_self : bool
        Whether to ignore a token's attention to itself
    ignore : list[int] or None
        Token positions that should not be selected as attended-to tokens
    """
    if ignore is None:
        ignore = []

    N = len(labels)

    for query_idx in range(N):
        row = weights[query_idx].copy()

        # Ignore future tokens
        for key_idx in range(query_idx + 1, N):
            row[key_idx] = -1

        # Optionally ignore the token itself
        if exclude_self:
            row[query_idx] = -1

        # Optionally ignore selected key positions
        for key_idx in ignore:
            row[key_idx] = -1

        # The first token has no previous token if we exclude self-attention
        if row.max() < 0:
            continue

        key_idx = row.argmax()
        weight = row[key_idx]

        query_token = labels[query_idx]
        key_token = labels[key_idx]

        print(
            query_token.ljust(15), "-->", key_token.ljust(15), round(weight, 3)
        )
```

We can run this on the averaged heads of layer `2`:

```{code-cell}
strongest_previous_token(attentions[2], labels)
```

...or look at a particular head in that layer:

```{code-cell}
strongest_previous_token(layer_2_heads[3], labels)
```

Note, though, that "Then" dominates across our layers when we look at the
averaged heads. 

```{code-cell}
for layer_idx in [0, 11]:
    print("Layer", layer_idx)
    strongest_previous_token(attentions[layer_idx], labels)
```

This doesn't necessarily mean that `"Then"` is the most meaningful word in the
sentence. In a causal language model like GPT-2, the first token is special
because every later token is allowed to attend to it. First tokens tend to act
as **attention sinks**, which absorb much of the attention weight allocated by
the model. Averaging across heads and reporting only the single strongest link
makes this phenomenon even more visible.

But this is why we have an optional `ignore` parameter in our function. Let's
ignore `"Then"` and look at the results.

```{code-cell}
for layer_idx in [0, 7, 11]:
    print("Layer", layer_idx)
    strongest_previous_token(attentions[layer_idx], labels, ignore=[0])
```

## Contextual Embeddings

Attention weights can tell you a lot about how token vectors change as the
model works. But we can also trace these changes by extracting the model's
**hidden states**: the layer-wise output of every Transformer block. Every
hidden state contains one representation for each token in the sequence at a
particular depth in the model.

For GPT-2, this means we can inspect:

- The input representation before any Transformer block runs
- The representation after the first, second, third, etc. Transformer block(s)
- The model's final representation

In other words, hidden states let us trace how a token's representation changes
as it becomes contextualized.

To extract this information, set `output_hidden_states=True` in the model call:

```{code-cell}
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
```

Hidden states are stored in `outputs.hidden_states`.

```{code-cell}
print("Number of hidden-state tensors:", len(outputs.hidden_states))
print("Shape of first hidden-state:", list(outputs.hidden_states[0].shape))
```

GPT-2 has 12 Transformer blocks, but we get 13 hidden-state tensors. The first
one is the input representation. The remaining 12 are the outputs of the
Transformer blocks.

Each tensor has shape:

$$
(\text{batch size}, \text{number of tokens}, \text{hidden dimension})
$$

Since we only gave the model one sequence, the batch size is `1`. Below, we'll
squeeze out the batch dimension, move our hidden states to the CPU, and stack
them into a single matrix:

```{code-cell}
hidden_states = torch.stack(outputs.hidden_states).squeeze(1).detach().cpu()
print("Hidden states shape:", list(hidden_states.shape))
```

Future chapters will use these hidden states to understand internal mechanisms
in models. Below, we will simply look at how token vectors change between them.

### Projecting hidden states into a shared space

Hidden states live in a 768-dimensional space, so we can't plot them directly.
As we did in the last chapter, we use PCA to project them down to two
dimensions.

First: we clip off the final hidden state. GPT-2 applies a final normalization
step after the last Transformer block, which can shift representations sharply
in PCA space. Below, we only want to focus on block-by-block updates.

```{code-cell}
trajectory_states = hidden_states[:-1]
print("Trajectory states shape:", list(trajectory_states.shape))
```

To fit a PCA, we must flatten from `(n_state, n_token, n_dim)` to `(n_state *
n_token, n_dim)`. Then we convert to a NumPy array.

```{code-cell}
n_state, n_token, n_dim = trajectory_states.shape
flat_hidden_states = trajectory_states.reshape(n_state * n_token, n_dim)
flat_hidden_states = flat_hidden_states.to(torch.float64).numpy()
```

Now we **standardize** the dimensions of the array by removing the mean and
scaling the data to unit variance. This prevents the plot from being dominated
by a small number of dimensions with large values.

```{code-cell}
scaler = StandardScaler()
scaled = scaler.fit_transform(flat_hidden_states)
```

With the arrays scaled, we fit a new PCA:

```{code-cell}
pca = PCA(n_components=2, random_state=0)
hidden_2d = pca.fit_transform(scaled)
```

Finally, we reshape back to `(n_state, n_token, 2)` so that we have plot-ready
arrays for each token.

```{code-cell}
hidden_2d = hidden_2d.reshape(n_state, n_token, 2)
print("2d hidden states shape:", hidden_2d.shape)
```

### Plotting trajectories

Now that our hidden states are in a shared space, we can plot token
trajectories. We select those for `" embracing my mother's ghost"` and plot.

```{code-cell}
selected = [8, 9, 10, 11]

fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
for token_idx in selected:
    token_path = hidden_2d[:, token_idx, :]

    x = token_path[:, 0]
    y = token_path[:, 1]

    ax.plot(x, y, marker="o", label=labels[token_idx])

    # Label the first, middle, and final points by hidden-state index
    for state_idx in [0, n_state // 2, n_state - 1]:
        ax.text(
            x[state_idx],
            y[state_idx],
            str(state_idx),
            ha="center",
            va="center",
            fontsize=8,
        )

    # Label the final point with the token
    ax.text(x[-1], y[-1], labels[token_idx], ha="left", va="bottom")

ax.set(
    title="Layer-by-layer Token Representation Trajectories",
    xlabel="PC 1",
    ylabel="PC 2",
)
plt.show()
```

Each line follows one token through the model.

### Measuring trajectories

Let's measure this movement for these and all other tokens. We'll use
**Euclidean distance** to measure how far the token's final hidden state is
from its initial embedding.

For two points in a two-dimensional space:

$$
(x_1, y_1)
\quad\text{and}\quad
(x_2, y_2)
$$

The Euclidean distance between them is:

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

Unlike cosine similarity, this is just an ordinary straight line distance
between two points.

Below, we take each token's start and end position and, in a for-loop, run the
formula above on those pairs.

```{code-cell}
start = hidden_2d[0]
end = hidden_2d[-1]

distances = []
for token_idx in range(n_token):
    x1, y1 = start[token_idx]
    x2, y2 = end[token_idx]

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distances.append(distance)
```

Now we plot. Tokens with larger bars are those that move farther in the PCA
projection. This suggests that their representations changed more as the model
processed the sentence. But remember: PCA compresses 768 dimensions down to 2,
so the plot and distances are rough representations at best.

```{code-cell}
fig, ax = plt.subplots()
sns.barplot(x=labels, y=distances, ax=ax)
ax.set(
    title="How Far Did Each Token Move?",
    xlabel="Token",
    ylabel="Distance in PCA space",
)
ax.tick_params(axis="x", rotation=90)
plt.show()
```
