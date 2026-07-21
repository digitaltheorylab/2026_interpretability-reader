# Introduction to Interpretability for Language Models

+ **Instructor:** Tyler Shoemaker
+ **Dates:** 07/20/2026--07/30/2026
+ **Meeting time:** MTWR, 10am-1pm EST

In this two-week crash course, participants will gain a baseline understanding
of the vocabulary, concepts, and methods involved in current interpretability
work. Our first week overviews the basic components of generative language
models like GPT-2. The second week ferries participants through a variety of
approaches that researchers use to explain model behavior, ending with a day on
multimodality. Throughout, our focus will be on the _model_ as an object in its
own right. Instead of using a model to analyze some external dataset, our
hands-on sessions select various facets of the model itself (embeddings,
outputs, etc.) and build explanations on that basis.

## Syllabus

| Day | Date     | Topic                              |
|-----|----------|------------------------------------|
| 1   | M (7/20) | Setup and overview                 |
| 2   | T (7/21) | Tokenization and embeddings        |
| 3   | W (7/22) | Embeddings and attention           |
| 4   | R (7/23) | Sampling and generation            |
| 5   | M (7/27) | Text as data                       |
| 6   | T (7/28) | Attribution and feature importance |
| 7   | W (7/29) | Meta-modeling                      |
| 8   | R (7/30) | Open day: participants choose      |

## Making the Reader Data

Scripts for making the reader data are under `src/`. To do this all in one go,
run:

```sh
bash scripts/build-data.sh
```

Alternatively, run individual scripts:

| Task                                          | Script                | Chapter(s) |
|-----------------------------------------------|-----------------------|------------|
| Generate genre-conditioned short stories      | `generate_stories.py` | 5          |
| Run MCQA genre classification                 | `classify_mcqa.py`    | 6, 7       |
| Get hidden states from MCQA                   | `extract_hiddens.py`  | 6, 7       |
| Fine-tune a linear genre classifier           | `train_probe.py`      | 7          |
| Obtain token attributions from the classifier | `attribute_tokens.py` | 7          |
| Calculate genre lift from token attributions  | `calculate_lift.py`   | 7          |

Utilities for scripts are stored in `common.py`.
