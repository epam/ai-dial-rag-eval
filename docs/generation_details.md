## Goal of the Pipeline

The pipeline aims to evaluate the coherence, factual accuracy, and contextual appropriateness of model-generated responses.

## Vocabulary

* **Question:** A request for an answer.
* **Document:** Original source containing the information.
* **Context:** Text excerpts retrieved from the document.
* **Ground Truth Answer:** The expected answer.
* **Answer:** The generated response.
* **Inference:** Determining whether information expressed in a hypothesis is stated in a premise.
* **Hypothesis:** A text to be evaluated for entailment.
* **Hypothesis Segment:** A sentence-level substring of a hypothesis obtained by tokenization.
* **Premise:** A text from which the hypothesis is inferred.
* **Statement:** An atomic declarative claim extracted from a hypothesis segment.
* **Answer Refusal:** Explicitly indicating missing or insufficient information or a refusal to answer.
* **Error:** Information about a failure that occurred during processing of a pipeline stage.
* **NaN (Not a Number):** The value assigned to a metric when an error prevents its calculation.

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

### Error Handling

If any stage of the pipeline fails for a given input (e.g., decontextualization, statement extraction, or inference scoring), the error is captured and propagated. The resulting metric for that input is `NaN`. Error details are preserved in the output for diagnosis.

### Inference Pipeline

The inference pipeline evaluates whether a hypothesis can be inferred from a premise.
It consists of three sequential LLM stages.

#### Stage 1: Decontextualization

> **Note:** Requires the `en_core_web_sm` spaCy model (`python -m spacy download en_core_web_sm`).

* The hypothesis is split into hypothesis segments (sentences) using spaCy sentence tokenization:
  1. Split by spaCy sentencizer (`en_core_web_sm`).
  2. For segments exceeding 500 characters, split further by `"\n\n"`, then `"\n"`, then into 500-character chunks.
  3. Merge segments shorter than 20 characters with the following segment.
* Pronouns and context-dependent references in the segments are replaced with their explicit referents using an LLM, making each segment self-contained.

#### Stage 2: Statement Extraction

* Each hypothesis segment is broken down into atomic, declarative, self-contained statements using an LLM.
* Enumerations and lists are split into separate statements, each preserving the parent clause relationship.
* Segments that are already simple statements are returned unchanged.

#### Stage 3: Inference Scoring

* For each hypothesis segment, the extracted statements are evaluated against the premise using an LLM.
* The LLM tags each statement as:
  * **ENT** — entailed by the premise
  * **CONT** — contradicts the premise
  * **NEUT** — neutral (neither entailed nor contradicted)
* The inference score for a segment is the proportion of **ENT** tags among its statements.

#### Aggregation

Segment-level scores are aggregated into a hypothesis-level result:

* **inference**: Weighted average of segment scores (weight = number of statements per segment). `NaN` if any segment score is `NaN`.
* **inference_min**: Same weighted average with `NaN` segment scores replaced by `0.0`.
* **inference_max**: Same weighted average with `NaN` segment scores replaced by `1.0`.

When an error occurs before the inference stage (in decontextualization or statement extraction), all three values are set to `NaN`, `0.0`, and `1.0` respectively, since the number of statements per segment is unknown.

#### Evaluated Premise/Hypothesis Pairs

The inference pipeline is run for three pairs:

| Pair                   | Premise | Hypothesis |
|------------------------|---------|-----------|
| Context -> Answer      | Joined context chunks | Answer |
| Answer -> Ground Truth | Last segment of the question + Answer | Ground Truth Answer |
| Ground Truth -> Answer | Last segment of the question + Ground Truth Answer | Answer |

#### Final Score

`mean_inference` and `median_inference` are computed as the mean and median of all three inference scores per row.

### Answer Refusal Pipeline

* The first three segments of the answer (or ground truth answer) are extracted.
* An LLM classifies the text as either an answer refusal or an actual answer.
* Refusal criteria: the text states that the question is wrong, that information is missing or unavailable, or that answering is refused. Single words, numbers, signs, and links are not considered refusals.
* Output: `1.0` for refusal, `0.0` for an actual answer.

## Scope, Known Limitations

**Scope:** The answer and ground truth answer are plain text.

**Limitations:**

* Splitting a hypothesis into hypothesis segments can lead to inaccurate inference if a sentence is split across segment boundaries. This is particularly problematic for lists and tables.
