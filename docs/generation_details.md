## Goal of the Pipeline

The pipeline aims to evaluate the coherence, factual accuracy, and contextual appropriateness of model-generated responses.

## Vocabulary

* **Question:** A request for an answer.
* **Document:** Original source containing the information.
* **Context:** Text excerpts retrieved from the document.
* **Ground Truth Answer:** The expected answer.
* **Answer:** The generated response.
* **Inference:** Determining if a hypothesis can be logically concluded from a premise.
* **Hypothesis:** A sentence to be evaluated for entailment.
* **Premise:** A set of sentences used to infer the hypothesis.
* **Statement:** A substring of the hypothesis.
* **Answer Refusal:** Explicitly indicating missing or insufficient information or a refusal to answer.

## Inputs

* Context
* Question
* Ground Truth Answer
* Answer

## Outputs

* Entailment score between the answer and the context.
* Entailment score between the answer and the ground truth answer.
* Entailment score between the ground truth answer and the answer.
* Answer refusal flag.
* Ground truth answer refusal flag.

## Pipeline Overview

The pipeline consists of two stages: sentence conversion and NLI evaluation.

### Sentence Conversion Stage

1. **Tokenization:**
   * Split the question, ground truth answer, and answer into sentences using:
     1. `nltk.sent_tokenizer`.
     2. For sentences exceeding 500 characters, split by `"\n\n"` and then by `"\n"`. If the length remains over 500 characters, split into 500-character chunks.
     3. Combine sentences shorter than 20 characters with the following sentence.
2. **Pronoun Replacement:**
   * Replace pronouns in the ground truth answer and answer with their corresponding nouns using a Language Model.

### Evaluating Stage

1. **NLI (Natural Language Inference):**
   * Create three pairs of (premise, hypothesis list):
     1. (context, answer)
     2. (the end of the question + ground truth answer, answer)
     3. (the end of the question + answer, ground truth answer)
   * For each pair, attempt to entail each hypothesis from the premise using NLI.
   * Process each hypothesis with the premise using an LLM.
   * Instruct the LLM to:
     * Split each hypothesis into statements.
     * Determine if the statements are entailed by the premise.
     * Tag the statements.
   * Calculate the final estimate for each hypothesis as the average estimate of its statements.
   * Calculate the final estimate for each pair as the average estimate of its hypotheses.

2. **Answer Refusal:**
   * Extract the first three sentences from the answer and ground truth answer.
   * Use an LLM to classify each answer and ground truth answer as either an answer refusal or not.

## Scope, Known Limitations

**Scope:** The answer and ground truth answer are plain text.

**Limitations:**

* The text splitting on hypotheses can lead to inaccurate entailment if parts of a sentence are separated. This is particularly problematic for lists and tables.
