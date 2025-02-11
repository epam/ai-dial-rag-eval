# aidial-rag-eval

## Project Description

`aidial-rag-eval` is a library designed for RAG (Retrieval-Augmented Generation) evaluation, where retrieval and generation metrics are calculated.

## Installation

To install the library, use the following pip command:

```bash
pip install aidial-rag-eval
```

## Usage

### Example

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from aidial_rag_eval import create_rag_eval_metrics_report
from aidial_rag_eval.metric_binds import CONTEXT_TO_ANSWER_INFERENCE

azure_llm = ChatOpenAI(...)

df_ground_truth = pd.DataFrame(
    {
        "question": ["Who am I?"],
        "answer": ["I am clever."], 
        "facts": [["I am smart."]],
        "documents": [["doc_name"]],
    }
)

df_answer = pd.DataFrame(
    {
        "question": ["Who am I?"],
        "answer": ["I am able to think."], 
        "context": [["I am smart."]],
        "documents": [["doc_name"]],
    }
)

df_metrics = create_rag_eval_metrics_report(
    df_ground_truth,
    df_answer,
    llm=azure_llm,
    metric_binds=[
        CONTEXT_TO_ANSWER_INFERENCE,
    ],
)
```

## License


## Contact


---