---
marp: true
---
# Goal of the Pipeline

The goal of the pipeline is to ensure that the model generates responses that are coherent, factually accurate, and contextually appropriate to the context and ground truth answer.

---

# Vocabulary
- **Context**: Chunks of text retrieved from the document.
- **Question**: A request for which we are expecting an answer.
- **Ground Truth Answer**: A predefined answer that we expect to be given.
- **Answer**: The answer which we are evaluating.
- **Hypothesis**: A sentence that we want to entail.
- **Premise**: A set of sentences from which we want to entail the hypothesis.
- **Fact**: A substring of the hypothesis.
- **Answer Refusal**: Explicit state indicating missing/insufficient information or refusal to answer.

# Inputs:
- Context
- Question
- Ground truth answer
- Answer

# Outputs:
- Estimate of how the answer is entailed by the context
- Estimate of how the answer is entailed by the ground truth answer
- Estimate of how the ground truth answer is entailed by the answer
- Whether the answer is a refusal
- Whether the ground truth answer is a refusal

# Pipeline Overview
The pipeline is divided into two parts: sentence conversion and NLI evaluation. Almost all steps can be tuned. The current hyperparameters are assumed to be set.

## Pipeline Steps
### Sentence Conversion Stage
1. **Tokenization**:
   - The context, question, ground truth answer, and answer are split into sentences using `nltk.sent_tokenizer`.
   - For each sentence: If the sentence length exceeds 500 characters, it is split by `"\n\n"`, than by `"\n"`. If the length is still more than 500 characters, it is split into chunks of size 500.
   - For each sentence: If the sentence length is less than 20 characters, it is appended to the following sentences(or previous if it is the last sentence).
2. **Pronoun Replacement**:
   - For all sentences in the ground truth answer and answer, replace all pronouns with their corresponding nouns using a Language Model. Up to 16 sentences are processed in one prompt.
   - Do it in 32 parallel threads.

### Evaluating Stage
1. **NLI (Natural Language Inference)**:
   - Create 3 pairs of (premise, list of hypotheses):
     1. (context, answer)
     2. (question last sentence + ground truth answer, answer)
     3. (question last sentence + answer, ground truth answer)
   - For each pair: For each sentence (hypothesis) from the list of hypotheses, attempt to entail it from the premise in terms of NLI.
   - Use LLM to process, placing each hypotheses in the prompt with the premise and optional document name.
   - Ask the LLM to:
     - Split each hypothesis into facts.
     - Explain if the facts entail from the premise.
     - Tag the facts.
   - Do it in 32 parallel threads.

   - The final estimate of each hypothesis is the mean estimate of its facts.
   - The final estimate for each pair is the mean estimate of its hypotheses.

2. **Answer Refusal**:
   - Take the first 2 sentences from the answer and from the ground truth answer.
   - Create a batch of size 8 to give to the LLM.
   - Ask the LLM to tag each answer and each ground truth answer in a batch as either an answer refusal or not.
   - Do it in 32 parallel threads.

---

# Scope, Known Limitations, Challenges
## Scope: The answer and ground truth answer are plain texts.

The main weak part of the algorithm is the text split on hypothesis.
If one part of the text doesn't make sense without another part and they are split into different sentences, then we can fail entailment. Examples: List, Table

# Current results

### LLM Models:

- gemini-flash1: gemini-flash.001
- gemini-flash2: gemini-flash.002
- gpt4o-mini: gpt4o-mini

### Prompt:

- same: nli and answer refusal tasks in the same prompt
- separated: nli and answer refusal tasks in different prompt

### Concurrency:

- no concur: old algorithm
- concur %d: number of threads


|               | LLM Model     | Prompt     | Concurrency | Time transforming | Time getting NLI | Input tokens | Output tokens | Total tokens |
|---------------|---------------|------------|-------------|-------------------|------------------|--------------|---------------|--------------|
|               | gemini-flash1 | same       | no concur   | 01:45             | 10:14            | NAN          | NAN           | NAN          |
|               | gemini-flash1 | separated  | no concur   | 00:41             | 07:01            | NAN          | NAN           | NAN          |
|               | gemini-flash1 | separated  | concur 32   | 00:05.9123        | 00:27            | 227514       | 35973         | 263487       |
|               | gemini-flash2 | separated  | concur 32   | 00:07.0558        | 00:38            | 227552       | 34828         | 262380       |
|               | gpt4o-mini    | separated  | concur 32   | 00:10.5841        | 00:51            | 213086       | 35989         | 249075       |

### ROC AUC SCORE, manually labeled hypotheses vs experiments


|               | LLM Model     | Prompt     | Concurrency | Context to answer ROC AUC | Answer to ground truth answer ROC AUC   |  Ground truth answer to answer ROC AUC |
|---------------|---------------|------------|-------------|---------------------------|-----------------------------------------|----------------------------------------|
|               | gemini-flash1 | same       | no concur   | 0.9473684210526316        | 0.8596837944664033                      | 0.8538461538461539                     |
|               | gemini-flash1 | separated  | no concur   | 0.9473684210526316        | 0.969038208168643                       | 0.984615384615384                      |
|               | gemini-flash1 | separated  | concur 32   | 0.9473684210526316        | 0.9137022397891963                      | 0.923076923076923                      |
|               | gemini-flash2 | separated  | concur 32   | 0.9342105263157895        | 0.9538866930171278                      | 0.8747252747252748                     |
|               | gpt4o-mini    | separated  | concur 32   | 0.9605263157894737        | 0.7476943346508563                      | 0.9285714285714286                     |

---