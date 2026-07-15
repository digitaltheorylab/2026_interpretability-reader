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

# Setup and Overview

## Python Basics

We will be writing all of our code in Python. It's an extremely popular,
general purpose programming language with big communities of support. For our
purposes, most notable among these communities are the ones that support
machine learning and natural language processing: most people use Python to
write this kind of code.

### Expressions

To write Python code in a notebook, click on a cell and type out an
**expression**. Expressions are combinations of values, variables, operators,
and functions, which Python **interprets** and then **evaluates**.

Here's a simple expression:

```python
2 + 2
```

Press `Shift` + `Enter` to run it:

```{code-cell}
2 + 2
```

Try subtraction:

```{code-cell}
7 - 1
```

You can write any arithmetic equation in Python using these and other
**operators**: symbols that represent operations for arithmetic, comparison,
and logical evaluation.

| Operator | Meaning                     |
|----------|-----------------------------|
| `+`      | Addition                    |
| `-`      | Subtraction                 |
| `*`      | Multiplication              |
| `/`      | Division                    |
| `%`      | Remainder division (modulo) |
| `**`     | Exponentiation              |

Use parentheses `( )` to create more complicated expressions. Python will
evaluate them in the standard order of operations: parentheses, exponentiation,
multiplication, division, addition, and finally subtraction (PEMDAS).

```{code-cell}
(2 + 2) * (7 - 1)
```

**Comments** are ignored by Python. You mark them with `#`. If we put this
character before our expression above, the following code block won't evaluate
the math:

```{code-cell}
# (2 + 2) * (7 - 1)
```

Typically, you'll use comments to remark on something in the code, or to write
some inline documentation.

```{code-cell}
# Python evaluates expressions using the standard PEMDAS logic
(2 + 2) * (7 - 1)
```

### Variables

Below we calculate the area of a triangle using the formula:

$$
A = \frac{1}{2}bh
$$

Where $b$ is the base and $h$ is the height.

We can express this formula in code, using $b=2.5$ and $h=4$:

```{code-cell}
1 / 2 * 2.5 * 4
```

But without the context above, it's difficult to determine what `2.5` and `4`
stand for in this expression. Enter **variables**. Variables are identifiers
that store values in code. Create one by writing out the name of the variable
and using the assignment operator `=` to link the variable with an expression:

```{code-cell}
b = 2.5
h = 4
1 / 2 * b * h
```

Usually, the more explicit you are with your variable names, the better:

```{code-cell}
base = 2.5
height = 4
1 / 2 * base * height
```

Variables can be any combination of letters, numbers, or underscores `_`. They
can't start with a number, however:

```{code-cell}
:tags: [raises-exception]
4height = 4
```

````{attention}
The code above **raises** a `SyntaxError`, one of several different [kinds of
errors][exceptions] in Python. We'll encounter more in the next few chapters,
and chances are good that you'll write buggy code when you're first learning
Python. The most important thing to do when your code raises an error is to
_read the error message_.

In our `SyntaxError`, for example:

```py
SyntaxError: invalid decimal literal
```

The message is telling us that the number `4` is an invalid fixed, or
"literal," value in a variable.

[exceptions]: https://docs.python.org/3/library/exceptions.html
````

The other constraint with variable names: operators are disallowed. Other than
that, you're free to write variables as you please. Below, we use the values
assigned to `base` and `height` in an expression and store the result of that
expression in a new variable, `area`:

```{code-cell}
area = 1 / 2 * base * height
```

Use `print()` to show what value is stored in a variable:

```{code-cell}
print(area)
```

You can also **reassign** values to variables:

```{code-cell}
base = 5
area = (1/2) * base * height
print(area)
```

### Strings

Python uses different **data types** to store certain kinds of values. We've
already used numeric types, but it's also possible to create strings and
logical values.

To create a string, use either single `'` or double `"` quotes:

```{code-cell}
"I am a string"
```

The quotes must match, or you'll get an error:

```{code-cell}
:tags: [raises-exception]
"I am a string'
```

If you want to use quotations inside your string, you need to use a different
kind of quotation mark to enclose it:

```{code-cell}
"How do you say 'Hello' in German?"
```

Alternatively, **escape** the string with `\`. This tells Python to treat the
following symbol as if it were just a string, not a special character in the
language itself:

```{code-cell}
'I\'m a string'
```

### Numbers

There are two kinds of numeric types in Python: **integers** and **floats**.
Integers are just whole numbers:

```{code-cell}
4
```

...whereas floats represent decimals:

```{code-cell}
5.1
```

Use `type()` to determine what kind of data type you're using (this works for
all data types):

```{code-cell}
type(5.1)
```

When you perform arithmetic in Python, the language automatically determines
whether the result should be represented as an integer or a float. We use
`type()` to show the data type of a few results below.

```{code-cell}
print("Integers:", type(5 + 5))
print("Floats:", type(2.1 / 7.9))
print("Float from integers:", type(10 / 1))
print("Integers and floats:", type(5 * 5.0))
```

Both numeric types may be either positive or negative. Use `-` to create a
negative number:

```{code-cell}
-4
```

Expressions may also produce negative numbers:

```{code-cell}
8 - 12
```

### Comparisons

You'll often need to compare values when you write code. You'll do this with
**comparison** operators.

| Operator | Meaning                  |
|----------|--------------------------|
| `<`      | Less than                |
| `>`      | Greater than             |
| `<=`     | Less than or equal to    |
| `>=`     | Greater than or equal to |
| `==`     | Equal to                 |
| `!=`     | Not equal to             |

```{note}
"Equal to" uses two equal symbols to distinguish it from the assignment
operator `=`.
```

Comparisons return `True` or `False`. These are **Boolean** data types. Here
are a few examples:

```{code-cell}
-6 < 0
```

```{code-cell}
5 + 5 != 9
```

```{code-cell}
1.4 >= 1.6
```

Comparisons will often work across types:

```{code-cell}
"1" == 1
```

Booleans are also assigned their own **keywords**:

```{code-cell}
True
```

They may be compared like any other data type:

```{code-cell}
False == True
```

### Conditionals

Often, you'll use Booleans in combination with **conditional expressions** to
check the state of your code and perform a **branching** operation. We manage
these operations with special keywords.

Below, we use `if` to determine whether a comparison is true. If our code meets
this condition, we use `print()` to print `True`.

```{code-cell}
x = 5
y = 3
if x > y:
    print(True)
```

Importantly, if a comparison does not meet a condition, the code does nothing.

```{code-cell}
y = 10
if x > y:
    print(True)
```

We would need to handle this second case ourselves using `else`:

```{code-cell}
if x > y:
    print(True)
else:
    print(False)
```

Note that the code blocks above use indentation. In Python, indentation is
meaningful. The language uses indentation blocks to separate certain portions
of its operations that are only relevant in particular contexts.

```{code-cell}
if x > y:
    print(True)
else:
    if y < 100:
        print("We entered a new context")
    else:
        print(False)
```

```{note}
Most Python programmers represent indentation with four spaces. But the Python
interpreter is flexible about indentation: You can use any number of spaces or
tab `<TAB>` characters to indent a code block. But you must be consistent: All
code must have the same indentation.
```

The `elif` keyword allows us to rewrite this logic without additional nesting:

```{code-cell}
if x > y:
    print(True)
elif y < 100:
    print("Second check worked")
else:
    print(False)
```

Alternatively, you can combine conditional checks. Below, we rewrite the
expression above using `or`:

```{code-cell}
if y > x or y < 100:
    print(True)
else:
    print(False)
```

Here is a table of keywords for Boolean operations:

| Keyword | Meaning  |
|---------|----------|
| `and`   | And      |
| `or`    | Or       |
| `not`   | Not      |
| `is`    | Identity |
| `in`    | In       |


## Functions, Modules, and Packages

From here, you could construct whatever code you'd like. But writing out the
logic for special equations or common mathematical operations, as well as for
certain general use patterns like print statements, would be a lot of work.
Worse, you'd have to do this every time you wanted to write a new piece of
code.

This is why **functions** exist. Functions are pieces of reusable code that
offer access to all sorts of features in Python and its external packages.
Functions are like little machines that accept inputs and (usually) produce
some kind of output. In the context of programming languages, we call the
inputs to a function its **arguments** and its outputs **return values**. When
you run a function, you **call** it.

### Calling functions

Calling a function involves writing out its name followed by parentheses; put
any arguments to the function inside those parentheses.

```{code-cell}
n = 4.813
round(n)
```

Functions often accept more than one argument. For example, `round()` has two:

1. `number`: the number to round
1. `ndigits`: decimal places to keep

Separate arguments with a comma `,`.

```{code-cell}
round(n, 1)
```

The arguments you supply to Python are assigned to a function's **parameters**.
These are function-specific variables that exist as long as the function runs.
Some parameters have **default arguments**, so you don't need to supply them
when you call the function. That was the case when we first used `round()`. Its
second parameter defaults to `0`.

Normally, parameters are assigned by their position: the first argument goes to
the first parameter, the second to the second, etc. But you can override these
positions by writing out the parameter name to which you want to assign an
argument.

The following three calls to `round()` are all the same:

```{code-cell}
round(n, 1) == round(n, ndigits=1) == round(ndigits=1, number=n)
```

### Importing modules

Every function we've used so far is is a **built-in** function. You'll have
access to these functions anytime you use Python. To see a full list of these
built-ins, refer to [Python's built-in documentation page][builtin].

[builtin]: https://docs.python.org/3/library/functions.html

Python's **Standard Library** contains many more functions than this, but they
aren't automatically available. Instead, they are stored in external
**modules**: files that define functions, classes, and variables.

To import a module, use the `import` keyword:

```{code-cell}
import math
```

Now we can access functions from `math` using the dot `.` notation. Below, we
calculate the square root with `sqrt()`.

```{code-cell}
math.sqrt(4)
```

To see all functions in a module, you can visit the documentation for the
[Python Standard Library][stdlib]. Or you can use `help()`:

[stdlib]: https://docs.python.org/3/library/index.html

```{code-cell}
:tags: [skip-execution]
help(math)
```

Likewise, use `help()` to see a function's arguments:

```{code-cell}
help(math.sqrt)
```

Note the lack of parentheses above. You aren't calling the function. Note, too,
the consistent use of the dot `.` when referencing this function. We can't use
it separately:

```{code-cell}
:tags: [raises-exception]
sqrt(4)
```

But if you find yourself using `sqrt()` frequently, and if you don't need other
functions from `math`, you can import only this function using `from...import`.

```{code-cell}
from math import sqrt

sqrt(4)
```

Finally, you can **alias** a module using `as`. This provides shorthand
notation for specifying modules and their contents without having to write the
full name of the module every time you use it.

```{code-cell}
import random as rand

rand.randint(1, 10)
```

### Importing packages

**Packages** contain multiple modules. The Python Standard Library contains
both, and this distinction doesn't really matter when you're using a basic
installation of Python.

It matters a little more when you start installing external code, which usually
comes as packages. That said, our workshop won't really touch on Python package
installation (it can actually be a mess to install packages!). Instead, all the
external packages you'll need have been installed into our coding environment
ahead of time.

Just as with Python's own modules and packages, you can import installed
packages using `import`:

```{code-cell}
import numpy

numpy.pi
```

````{note}
If you're curious, you can check which packages have been installed separately
like so:

```{code-cell}
import sys

print("math installed separately:", "math" not in sys.stdlib_module_names)
print("numpy installed separately:", "numpy" not in sys.stdlib_module_names)
```
````


## Text Generation Basics

The following chapters make heavy use out of one particular package:
[`transformers`][transformers]. This package provides a unified framework for
downloading, running, and training pretrained models from the [Hugging Face
Hub][hub]. We'll have a chance to learn about the details of this framework
later on. For now, let's just use `transformers` to generate a bit of text.

[transformers]: https://pypi.org/project/transformers
[hub]: https://huggingface.co/models

### Loading a model

First, we import `pipeline` from the package:

```{code-cell}
:tags: [remove-output]
from transformers import pipeline
```

Now, we load GPT-2 by **initializing** the `pipeline` with two arguments and
assign it to a variable, `generator`:

```{code-cell}
:tags: [remove-output]
generator = pipeline("text-generation", model="gpt2")
```

### Generating output

With the `pipeline` initialized, `transformers` makes basic text generation
trivial. Simply call it with a string:

```{code-cell}
:tags: [remove-output]
inp = "It was the best of times, it was"
output = generator(inp)
```

Let's look at the result.

```{code-cell}
print(output)
```

This worked! But right now, the generated string is packaged up in a particular
**data structure**. Specifically, the output of `generator` is a **list** that
stores a **dictionary**. We'll talk about both of these data structures in
future chapters. For now, just use these square brackets `[ ]` to pluck out the
string:

```{code-cell}
text = output[0]["generated_text"]
print(text)
```

### Generation configuration

`transformers` supports dozens of different generation configurations. Import
`GenerationConfig` to access them.

```{code-cell}
from transformers import GenerationConfig
```

As we did above, use `help()` to see these options:

```{code-cell}
:tags: [skip-execution]
help(GenerationConfig)
```

Below, we set `max_new_tokens=1`. This directs `pipeline` to emit a single new
token.

```{code-cell}
:tags: [remove-output]
generation_config = GenerationConfig(max_new_tokens=1)
output = generator(inp, generation_config=generation_config)
```

Let's look:

```{code-cell}
text = output[0]["generated_text"]
print(text)
```

Here, we get 10 new tokens and use sampling with `do_sample=True`. We'll also
set `temperature=1.5`. More about this parameter later, but in short: A higher
value lets models reach for less probable tokens (it defaults to `1.0`).
Finally, we set `num_return_sequences=5` to generate 5 different sequences.

```{code-cell}
:tags: [remove-output]
generation_config = GenerationConfig(
    max_new_tokens=10, do_sample=True, temperature=1.5, num_return_sequences=5
)
output = generator(inp, generation_config=generation_config)
```

As before, let's inspect the output, but this time, we'll look at the whole
thing:

```{code-cell}
print(output)
```

See how each different sequence has been packaged up between its own set of
curly braces `{ }`? The pipeline generated five different outputs, and they're
all different.

## Looking Under the Hood

But how does all this work? That's the question we'll explore in the next three
chapters before moving on to interpretability techniques. Over the course of
these first chapters you'll learn how generation works from start to finish.
Think of them as an extended walk-through of the network diagram below:

```{code-cell}
print(generator.model)
```

