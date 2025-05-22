# AI DIAL RAG EVAL

[![PyPI version](https://img.shields.io/pypi/v/aidial-rag-eval.svg)](https://pypi.org/project/aidial-rag-eval/)

## Overview

Library designed for RAG (Retrieval-Augmented Generation) evaluation, where retrieval and generation metrics are calculated.

## Usage

Install the library using [pip](https://pip.pypa.org/en/stable/getting-started):

```sh
pip install aidial-rag-eval
```

### Example

The example of how to get retrieval metrics along with answer inference based on the context.

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from aidial_rag_eval import create_rag_eval_metrics_report
from aidial_rag_eval.metric_binds import CONTEXT_TO_ANSWER_INFERENCE,\
    ANSWER_TO_GROUND_TRUTH_INFERENCE, GROUND_TRUTH_TO_ANSWER_INFERENCE

llm = ChatOpenAI(model="gpt-4o")

df_ground_truth = pd.DataFrame([
    {
        "question": "What is the diameter of the Earth and the name of the biggest ocean?",
        "documents": ["earth.pdf"],
        "facts": ["The diameter of the Earth is approximately 12,742 kilometers.", "The biggest ocean on Earth is the Pacific Ocean."],
        "answer": "The Earth's diameter measures about 12,742 kilometers, and the Pacific Ocean is the largest ocean on our planet."
    },])
df_answer = pd.DataFrame([
    {
        "question": "What is the diameter of the Earth and the name of the biggest ocean?",
        "documents": ["earth.pdf"],
        "context":  [
            "The Earth, our home planet, is the third planet from the sun. It's the only planet known to have an atmosphere containing free oxygen and oceans of liquid water on its surface. The diameter of the Earth is approximately 12,742 kilometers.",
            "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions, extending from the Arctic Ocean in the north to the Southern Ocean in the south."
        ],
        "answer": "The Earth has a diameter of approximately 12,742 kilometers."
    },
])

df_metrics = create_rag_eval_metrics_report(
    df_ground_truth,
    df_answer,
    llm=llm,
    metric_binds=[
        CONTEXT_TO_ANSWER_INFERENCE,
        ANSWER_TO_GROUND_TRUTH_INFERENCE,
        GROUND_TRUTH_TO_ANSWER_INFERENCE,
    ],
)
print(df_metrics[["facts_ranks", "recall", 'precision', 'mrr', 'f1', 'ctx_ans_inference', 'ans_gt_inference', 'gt_ans_inference']])
```

It is expected to see the following results:

| recall | precision | mrr | f1  | ctx_ans_inference | ans_gt_inference | gt_ans_inference |
| ------ | --------- | --- | --- | ----------------- | ---------------- | ---------------- |
| 0.5    | 0.5       | 0.5 | 0.5 | 1.0               | 0.5              | 1.0              |

In this table:

- "recall" of 0.5 indicates that only 1 out of 2 ground truth facts were found in the context.
- "precision" of 0.5 reflects that just 1 context chunk out of 2 includes any ground truth facts.
- The prefix of the inference metrics signifies the premise and hypothesis in the following format: *premise*_*hypothesis*_inference.
  - "ctx" refers to 'context'
  - "ans" refers to 'answer'
  - "gt" refers to 'ground truth answer'
- "ctx_ans_inference" and "ans_gt_inference" values of 1.0 mean our answer can be derived directly from the context and the ground truth answer, respectively.
- "gt_ans_inference" of 0.5, denotes that the ground truth answer can only be partially inferred from our answer.

## Developer environment

This project uses [Python>=3.9](https://www.python.org/downloads/) and [Poetry>=1.8.5](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests locally for available python versions:

```sh
make test
```

Run unit tests for the specific python version:

```sh
make test PYTHON=3.11
```

The generation evaluation requires an access to the LLM. The generation evaluation tests (located in `tests/llm_tests` directory) use cached LLM responses by default. To run the tests with real LLM responses, you need add `--llm-mode=real` argument to the test command:

```sh
make test PYTHON=3.11 ARGS="--llm-mode=real"
```

The test run with real LLM responses requires the following environment variables to be set:

| Variable     | Description                      |
| ------------ | -------------------------------- |
| DIAL_URL     | The URL of the DIAL server.      |
| DIAL_API_KEY | The API key for the DIAL server. |

Copy `.env.example` to `.env` and customize it for your environment.

## Clean

To remove the virtual environment and build artifacts run:

```sh
make clean
```

## Build

To build the package run:

```sh
make build
```
