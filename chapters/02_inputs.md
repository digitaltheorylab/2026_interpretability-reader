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

# Model Inputs

```{code-cell}
:tags: [remove-input]
from utils.plotting import SQUARE_FIGSIZE, set_plot_theme

set_plot_theme()
```

In this chapter, you'll learn about the core inputs that go into
Transformer-based language models. Before we begin, we need to perform several
imports from various libraries:


```{code-cell}
:tags: [remove-output]
from heapq import nlargest

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
```

```{note}
In future chapters, you'll see these import blocks at the top of text with no
other commentary.
```

With everything imported, we can load a model. But unlike in the last chapter,
we won't use `pipeline`. Instead, we will initialize GPT-2 directly.

```{code-cell}
:tags: [remove-output]
checkpoint = "gpt2"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

A second preparatory step: we use `.eval()` to put the model in evaluation
mode. This mode disables certain settings that models use for training.

```{code-cell}
model.eval()
```

## The Embedding Matrix

As we saw in the previous chapter, the first component of a model is an
**embedding layer**. It contains learned representations for every item in the
model's vocabulary: long sequences of numbers, or **vectors**, that the model
has adjusted during training. Think of the embedding layer as a dictionary of
sorts. When we send the model a word, it looks up the corresponding vector
representation in this layer.

We can obtain the embedding layer from the model by calling the
`.get_input_embeddings()` method:

```{code-cell}
embedding_layer = model.get_input_embeddings()
```

Let's try to send it some text. First, we'll initialize a `text` variable and
assign a string to it.

```{code-cell}
text = "This is a sequence of text"
```

Now, we'll attempt to send that string through `embedding_layer`. Note the
`try`/`except` logic: If this fails---that's a hint!---we'll catch the error.

```{code-cell}
try:
    embedding_layer(text)
except TypeError as e:
    print(e)
```

The message tells us that we are sending the wrong kind of data to
`embedding_layer`. It expects a **tensor**: a structured collection of numbers
used by libraries in machine learning (here, that library is [PyTorch][torch]).
More specifically, an embedding layer expects a tensor containing integers. But
text is a string, so what do these numbers represent?

[torch]: https://pytorch.org

We'll answer this question in the next section. But first, let's extract the
underlying **matrix** of embeddings from `embedding_layer`. This contains all
of the model's embedding vectors. Its **shape** corresponds to the vocabulary
size of the model and the length, or **dimension**, of each embedding vector.

```{code-cell}
emb = embedding_layer.weight.detach().numpy().astype(np.float64)
print("Embedding matrix shape:", emb.shape)
```

Our next task will be to transform the contents of `text` so that they align to
this matrix.

## Tokenization

That transformation entails **tokenizing** a string: segmenting the string into
individual units that match the model's vocabulary. These units are called
**tokens**. A **tokenizer** performs this segmentation. Every modern LLM
therefore has an associated tokenizer.

We load the one for GPT-2 below:

```{code-cell}
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

### Tokenization basics

Let's try to tokenize some text.

```{code-cell}
encoded = tokenizer.encode("Hello")
```

The tokenizer returns integers in a **list**, a general-purpose,
one-dimensional container for storing data.

```{code-cell}
type(encoded)
```

We'll discuss the contents of this list in a while, but first let's overview
some basic properties of lists. Lists are among the most common data structures
in Python. They make very little assumptions about the kind of data they store,
and they store this data in an ordered manner. That is, lists have a first
element, a second element, and so on up until the full length of the list.

In our case, that length is `1`:

```{code-cell}
len(encoded)
```

...and the single element this list contains is an integer:

```{code-cell}
encoded
```

You access this element by **indexing** the list. The square brackets `[ ]` are
Python's index operator. Use them in conjunction with the **index position** of
the element(s) you want to select. The index position is a number that
corresponds to the location of an element in the list.

```{code-cell}
encoded[0]
```

Why `0`? Python uses **zero-based indexing**. That means the positions of
elements are counted from `0`, not `1`.

Let's see how this works on a new string. Below, we use the tokenizer to
segment a multi-word string:

```{code-cell}
encoded = tokenizer.encode("Hello world!")
```

We now have a list of three tokens:

```{code-cell}
encoded
```

If we index the list with `1`, we'll extract the second element of the list:

```{code-cell}
encoded[1]
```

You can use `.decode()` to convert the contents of `encoded` back into string
representations all at once:

```{code-cell}
tokenizer.decode(encoded)
```

...or index a particular element in the list and decode that alone:

```{code-cell}
tokenizer.decode(encoded[1])
```

```{note} See the whitespace?
Many tokenizers prefix words with a whitespace `' '` character to mark word
boundaries. This also helps them reconstruct the input text exactly.
```

Note that lists can also be indexed from the end using negative indices. For
example, `-1` refers to the last element of the list:

```{code-cell}
encoded[-1]
```

The following two indices return the same element from `encoded`:

```{code-cell}
encoded[2] == encoded[-1]
```

### Subword tokenization

You'll likely have noticed by now that tokenizers encode not only words but
punctuation and other kinds of text in a string. So far, the examples have been
straightforward: each word and punctuation mark has corresponded to one element
in the output. But this isn't always the case. Consider the following
segmentation, which uses two tokens to represent a single word:

```{code-cell}
encoded = tokenizer.encode("programmatic")
encoded
```

Why is this? The answer has to do with the highly general nature of LLMs. These
models are trained on huge datasets, which means they must represent millions
of different pieces of text. But model vocabularies would quickly become
enormous if they contained a separate token for every unique word. This would
be inefficient, especially because many words are extremely rare.

In older word-level tokenization schemes, rare or unseen words were either a)
ignored, or b) mapped to a special "unknown" token. But LLMs need to handle
open-ended text, including rare words, new words, names, typos, code,
punctuation, and strings they may never have seen during training. They do this
by using pieces of words, or **subwords**, as tokens.

This allows a tokenizer to represent many words as combinations of smaller
pieces. At the same time, it adds a layer of potential indirection between text
and model input. While there are six words in this sequence:

```
large language models use subword tokenization
```

...tokenizing that sequence gives:

```{code-cell}
encoded = tokenizer.encode("large language models use subword tokenization")
print("Number of tokens:", len(encoded))
```

Instead of manually indexing the list, let's use a **for-loop** to step through
each element of `encoded` and decode it. A for-loop begins with the `for`
keyword, followed by:

- A **loop variable**, which represents one element in a list at a time
- The `in` keyword
- The list, or other **iterable**, we want to iterate over
- A colon `:`

Code in the body of the loop must be indented. 

Below, we iterate through all 8 tokens and decode each one. We then use
`repr()` in conjunction with `print()` to display the token and its decoded
string.

```{code-cell}
for number in encoded:
    decoded = tokenizer.decode(number)
    decoded = repr(decoded)
    print(number, "\t-->\t", decoded)
```

For-loops may be nested. In the following, we make our own list of sentences
with `[ ]`, encode each of them, and decode their contents in the same way.

```{code-cell}
sentences = ["This is a sentence", "And here is a second sentence"]

for sentence in sentences:
    print("Tokenizing sentence:", sentence)
    encoded = tokenizer.encode(sentence)

    for number in encoded:
        decoded = tokenizer.decode(number)
        decoded = repr(decoded)
        print(number, "\t-->\t", decoded)

    print()
```

See how the **outer** loop advances only after the **inner** loop is finished?

Alternatively, `.encode()` accepts lists of strings. The output of the
expression below is a list of lists:

```{code-cell}
encoded_batch = tokenizer.encode(sentences)
print(encoded_batch)
```

### Token IDs

As we've seen, the output of `.encode()` is a list (or lists) of numbers, which
don't seem to share a relationship with the input text. Of what significance is
the mapping between `0` and `"!"`?

The answer is, "Very little"---at least in terms of semantics. There is no
essential relationship between `0` and `"!"`. In principle, the number `0`
could be have been assigned to any token in the vocabulary. But for this
tokenizer, `0` stands for one specific token: `"!"`. That is, `0` is a **token
ID**, a unique integer that represents one token in the model's vocabulary.
Think of it like a label that the model uses to associate input text with its
internal representations.

This means that two instances of the same token type receive the same ID:

```{code-cell}
tokenizer.encode(["Token", "Token"])
```

...which differentiates these tokens from others:

```{code-cell}
tokenizer.encode(["Token", "Token", "token"])
```

Recall that our embeddings matrix contains vector representations for every
token in a model vocabulary. Usually, the number of rows in this matrix equal
the model's vocabulary size:

```{code-cell}
emb.shape[0] == model.config.vocab_size
```

These matching numbers give us a clue about the meaning of token IDs. A matrix
can also be indexed with the same logic we've been using for lists. Below, we
index the first row of our embeddings matrix:

```{code-cell}
vector = emb[0]
print("Vector length:", len(vector))
```

But what token does this vector correspond to? That's the role of the token ID:
it's actually an index position into the embeddings matrix. Every token ID
selects one row from this matrix, giving the model an initial vector
representation for the token.

```{code-cell}
encoded = tokenizer.encode("Hello world!")
for number in encoded:
    vector = emb[number]
    print("Vector length", len(vector), "for token ID", number)
```

We can index the matrix all at once by using a list of IDs:

```{code-cell}
encoded = tokenizer.encode("Hello world!")
emb_seq = emb[encoded]
print("Sequence embeddings matrix shape:", emb_seq.shape)
```

## Vector Basics

Just as with token IDs, a model's unique embedding vectors are all different.
We can use a function from [NumPy][np], a numerical computing library, to check
for unique vectors. Setting `axis=0` in `np.unique()` will tell NumPy to
perform compare **row-wise** comparisons across a two-dimensional **array**.

[np]: https://numpy.org/

```{code-cell}
num_unique = np.unique(emb_seq, axis=0)
print("Number of unique vectors:", len(num_unique))
```

Note, though, that instances of the same token get the same word embedding:

```{code-cell}
A = emb[0]
B = emb[0]
print("Arrays are the same:", np.array_equal(A, B))
```

Or, with a for-loop using `range()`, which generates a sequence of `N` numbers:

```{code-cell}
N = A.size
found_mismatch = False

for vector_index in range(N):
    a = A[vector_index]
    b = B[vector_index]

    if a != b:
        found_mismatch = True

if found_mismatch:
    print("Value mismatch found")
else:
    print("All values are the same")
```

The model will ultimately change these representations based on contextual
information in the input text, but that's a topic for the next chapter. For
now, let's discuss a few properties of vectors.

### Vector components

A vector has a **magnitude** and a **direction**.

**Magnitude**

- Description: The length of a vector from its origin to its endpoint. This is
  calculated as the square root of the sum of squares of its values
- Notation: $\|\mathbf{A}\| = \sqrt{a_1^2 + a_2^2 + \ldots + a_n^2}$
- Result: Single value (scalar)

```{code-cell}
magnitude = 0
for value in A:
    square = np.square(value)
    magnitude += square

magnitude = np.sqrt(magnitude)
print("Magnitude:", round(magnitude, 2))
```

Equivalently, using NumPy:

```{code-cell}
magnitude = np.linalg.norm(A)
print("Magnitude:", round(magnitude, 2))
```

A vector with length, or magnitude 1, is a **unit vector**. Unit vectors are
often used to represent direction without the original vector's magnitude.

**Direction**

- Description: The orientation of a vector in space, represented by a unit
  vector pointing in the same direction
- Notation: $\mathbf{\hat{A}} = \frac{\mathbf{A}}{\|\mathbf{A}\|}$
- Result: Vector with $n$ values and magnitude 1 

```{code-cell}
direction = A / magnitude
mag_direction = np.linalg.norm(direction)
print("Values in direction vector:", len(direction))
print("Magnitude of direction vector:", round(mag_direction, 2))
```

Let's plot two vectors to show their magnitude and direction. First, we get the
embeddings. Note that we insert a space before `"model"` to get the token
used for `"model"` when it appears after a whitespace, as it usually would
inside a sentence, rather than the token used at the start of a sequence.

```{code-cell}
encoded = tokenizer.encode(" model vector")
plot_emb = emb[encoded]
A = plot_emb[0]
B = plot_emb[1]
```

One problem immediately presents itself: we can only plot in two dimensions,
but our vectors have 768 dimensions. Later on we'll learn a trick for
summarizing those dimensions in a 2D plot, but for now, we'll index our vectors
to get the first two dimensions.

You can index multiple elements from a list or array using the colon `:` inside
`[ ]`. The following takes elements up to _but not including_ index position
`2`:

```{code-cell}
A_2d = A[0:2]
print("Number of values in A_2d:", len(A_2d))
```

When you index from the start of a list or array, you don't need the `0`:

```{code-cell}
B_2d = B[:2]
print("B_2d value count matches A_2d:", len(A_2d) == len(B_2d))
```

From here, we use a visualization library, [Matplotlib][mpl], to show the first
two dimensions of our vectors. Observe how direction and magnitude differ
between each vector. Not only do they point in different directions: they are
also different lengths. 

[mpl]: https://matplotlib.org

```{code-cell}
vectors = [A_2d, B_2d]
labels = ["model", "vector"]
colors = ["#0072B2", "#E69F00"]

limit = 0.25
head_width = limit * 0.05

fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
for vector, label, color in zip(vectors, labels, colors):
    x = vector[0]
    y = vector[1]
    ax.arrow(0, 0, x, y, head_width=head_width, color=color)
    ax.text(x, y, label, color=color)

ax.axhline(0, color="black")
ax.axvline(0, color="black")
ax.set(
    xlim=[-limit, limit],
    ylim=[-limit, limit],
    aspect="equal",
    title="Two-Dimensional Plot for 'model' and 'vector'",
    xlabel="Dimension 1",
    ylabel="Dimension 2",
)
plt.show()
```

### Vector operations

We turn now to basic operations you can perform on vectors. Nearly all these
operations produce vectors of length $n$; we'll construct these ourselves by
initializing empty lists and appending values to them using `.append()`.

```{code-cell}
N = A.size
```

**Summation**

- Description: Element-wise sums
- Notation: $\mathbf{A} + \mathbf{B} = (a_1 + b_1, a_2 + b_2, \ldots, a_n +
  b_n)$
- Result: Vector of length $n$

```{code-cell}
loop_result = []
for idx in range(N):
    x = A[idx] + B[idx]
    loop_result.append(x)
```

Equivalently, with NumPy:

```{code-cell}
np_result = A + B
print("Loop and NumPy match:", np.allclose(loop_result, np_result))
```

**Subtraction**

- Description: Element-wise differences
- Notation: $\mathbf{A} - \mathbf{B} = (a_1 - b_1, a_2 - b_2, \ldots, a_n -
  b_n)$
- Result: Vector of length $n$

```{code-cell}
loop_result = []
for idx in range(N):
    x = A[idx] - B[idx]
    loop_result.append(x)
```

Equivalently, with NumPy:

```{code-cell}
np_result = A - B
print("Loop and NumPy match:", np.allclose(loop_result, np_result))
```

**Multiplication, element-wise**

- Description: Element-wise products
- Notation: $\mathbf{A} \odot \mathbf{B} = (a_1 b_1, a_2 b_2, \ldots, a_n b_n)$
- Result: Vector of length $n$

```{code-cell}
loop_result = []
for idx in range(N):
    x = A[idx] * B[idx]
    loop_result.append(x)
```

Equivalently, with NumPy:

```{code-cell}
np_result = A * B
print("Loop and NumPy match:", np.allclose(loop_result, np_result))
```

**Multiplication, dot product**

- Description: Sum of element-wise products
- Notation: $\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^n a_i b_i$
- Result: Single value (scalar)

```{code-cell}
loop_result = 0
for idx in range(N):
    x = A[idx] * B[idx]
    loop_result += x
```

Equivalently, with NumPy:

```{code-cell}
np_result = A @ B
print("Loop and NumPy match:", np.allclose(loop_result, np_result))
```

Here is the actual dot product:

```{code-cell}
print("Dot product:", round(loop_result, 2))
```

The dot product is one of the most important operations in modern machine
learning. It measures the extent to which two vectors point in the same
direction:

- If the dot product is **positive**, the angle between the vectors is less
  than 90 degrees, and they point in the same direction
- If it is **negative**, the angle is greater than 90 degrees, and they point
  away from one another
- If it is **zero**, the vectors are perpendicular

Note that the dot product is also influenced by the vectors' magnitudes, not
just their directions. Two vectors that point in exactly the same direction
will have a larger dot product if they are longer. To measure direction alone,
we normalize by magnitude. This operation is known as **cosine similarity**.

## Cosine Similarity

We express cosine similarity as follows:

$$
\operatorname{sim}_{\cos}(\mathbf{A}, \mathbf{B})
= \cos\theta =
\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
$$

Where:

- $\mathbf{A} \cdot \mathbf{B}$ is the dot product of two vectors
- $\|\mathbf{A}\|$ and $\|\mathbf{B}\|$ are the magnitudes (norms) of each
  vector
- $\theta$ is the angle between the two vectors

Cosine similarity ranges between $[-1, 1]$, where:

- $1$: Same orientation; perfect similarity
- $0$: Orthogonal vectors; vectors have nothing in common
- $-1$: Opposite orientation; vectors are the opposite of one another

We calculate this value like so:

```{code-cell}
dot_product = A @ B
norm_by = np.linalg.norm(A) * np.linalg.norm(B)
cos_sim = dot_product / norm_by
print("Cosine similarity between 'model' and 'vector':", round(cos_sim, 2))
```

Somewhat similar, but marginally so.

The machine learning library, [scikit-learn][sklearn], provides a function for
calculating the cosine similarity of two vectors. It does this in a
**pairwise** manner, meaning it computes the similarity between every pair of
vectors in the input. For a matrix with $n$ rows, this produces an $n \times n$
matrix where entry $(i, j)$ is the cosine similarity between row $i$ and row
$j$.

[sklearn]: https://scikit-learn.org/stable

```{code-cell}
cos_sim = cosine_similarity(plot_emb)
np.round(cos_sim, 2)
```

Diagonal entries are always `1.0`, since each vector is perfectly similar to
itself. The off-diagonal entries give the similarities between different
vectors. Note, too, that the matrix is **symmetric**: entry $(0, 1)$ equals
entry $(1, 0)$.

### Ranking embeddings

The `cosine_similarity()` function accepts two arguments, `X` and `Y`. When
both are provided, it computes similarities between rows of `X` and rows of
`Y`. Let's use this functionality to calculate the cosine similarity between
`"language"` and all other embeddings.

First, we get our **query** vector. Unlike before, we won't extract the actual
index position from `encoded`. Instead, we'll index our embeddings with a list,
which will return a 2D array with one row.

```{code-cell}
encoded = tokenizer.encode(" language")
query = emb[encoded]
```

Now we calculate our similarities:

```{code-cell}
similarities = cosine_similarity(query, emb)
print("Similarities shape:", similarities.shape)
```

In other words: we have a single vector of cosine similarity scores between
`"language"` and all tokens in the model vocabulary.

What are the top 25 most similar tokens? First, we pair each similarity score
with its index. Then, we use `nlargest()` to find the pairs with the highest
scores. Finally, we print the results with `.ljust()` to control the spacing
between each token and its cosine similarity score.

```{code-cell}
top_k = 25
scores = similarities[0]
N = len(scores)

scored = []
for idx in range(N):
    sublist = [scores[idx], idx]
    scored.append(sublist)

top = nlargest(top_k, scored)

for sublist in top:
    score = sublist[0]
    idx = sublist[1]

    token = tokenizer.decode(idx)
    print(repr(token).ljust(15), round(score, 3))
```

## Vector Space

Cosine similarity is powerful because it enables comparisons between vectors
regardless of their magnitudes---in most cases, we only care about direction.
This means we can meaningfully compare tokens in a shared **vector space**,
where each token's position reflects its meaning relative to others.

### Plotting similarity scores

To build some intuitions about this space, let's select a subset of words from
our embeddings matrix.

```{code-cell}
words = " bank money loan river stream water bark tree dog cat"
encoded = tokenizer.encode(words)
batch_emb = emb[encoded]
print("Batch embeddings shape:", batch_emb.shape)
```

As before, we can calculate cosine similarity between these vectors.

```{code-cell}
similarities = cosine_similarity(batch_emb)
```

We can display these similarity scores with a **heatmap** using another
visualization library, [seaborn][sns].

[sns]: https://seaborn.pydata.org

```{code-cell}
labels = []
for token in encoded:
    decoded = tokenizer.decode(token)
    labels.append(decoded)

fig, ax = plt.subplots()
sns.heatmap(
    similarities,
    cmap="RdBu_r",
    robust=True,
    annot=True,
    fmt=".2f",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
)
ax.set(title="Cosine Similarity Scores")
plt.show()
```

### Dimensionality reduction

Even more effective, though, would be a plot that shows these vectors in a
shared space. But there's a hitch: as we saw already, the number of values (or
**dimensions**) of each vector exceeds what we can visualize. While we could
again plot just the first two dimensions of each vector, the result would be a
very poor representation of all the semantic information encoded across all 768
dimensions. So what to do?

Below, we use **principal component analysis**, or PCA, to reduce the
**dimensionality** of our vectors so we can plot them. PCA identifies new axes,
called **principal components**, along which the data varies most. By keeping
only the top few components and discarding the rest, we can project
high-dimensional vectors down to a few dimensions while preserving as much of
the variation as possible.

As with cosine similarity, the scikit-learn library can perform PCA for us.
Below, we initialize a `PCA` object with two components and **fit** it to our
vectors. This is the step where `PCA` identifies the new components.

```{code-cell}
pca = PCA(n_components=2)
pca.fit(batch_emb)
```

Now, we transform our embeddings with our fitted `PCA` object. This will reduce
each of our 10 tokens' vectors to 2D summaries.

```{code-cell}
emb_2d = pca.transform(batch_emb)
print("Transformed embeddings shape:", emb_2d.shape)
```

From here, we plot. Each point has two coordinates: an x-coordinate and a
y-coordinate. The two numbers in each row of `emb_2d` correspond to those
coordinates. In the code block below, the`:` means "take every row." So:

- `[:, 0]` means "take every row, and get column `0`"
- `[:, 1]` means "take every row, and get column `1`"

```{code-cell}
X = emb_2d[:, 0]
Y = emb_2d[:, 1]
```

### Visualizing embeddings

With our coordinate data in hand, we can turn to seaborn. Observe how, in the
plot below, the shading in the heatmap is mirrored by the orientation of points
in our scatter plot. In the latter, the closer two points are, the more similar
they are in meaning.

```{code-cell}
n_labels = len(labels)

fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
sns.scatterplot(x=X, y=Y, alpha=0.8, ax=ax)
ax.set(title="Two-Dimensional Vector Space", xlabel="PC 1", ylabel="PC 2")
for idx in range(n_labels):
    ax.text(X[idx], Y[idx], labels[idx])

plt.show()
```

Note that the "space" in this plot isn't a pre-existing container that vectors
sit inside. The space is defined by the vectors themselves---their positions
and the relationships between them. This means that the structure of this space
emerges entirely from the vectors' values. If we changed our vectors, say by
swapping one token for another in the model vocabulary, we'd change the space.

This is an important point for the next chapter. When an LLM processes text, it
doesn't move tokens around inside a fixed space. Instead, it produces new
vectors, which in turn define a new space. As we'll see, attention is one such
mechanism for producing these new, context-aware vectors.
