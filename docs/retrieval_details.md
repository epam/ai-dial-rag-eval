## Goal of the Pipeline

The pipeline aims to evaluate the relevance and accuracy of the retrieved document chunks in respect to a posed question.

## Vocabulary

* **Question:** A request for an answer.
* **Document:** Original source containing the information.
* **Context:** Chosen chunks of the document that are most relevant to the question.
* **Retrieval:** Retrieving the most fitting document chunks based on the model.
* **Ground Truth Fact:** Expected crucial information necessary to form a correct answer.
* **Chunk:** Individual parts the document is divided into.
* **Precision**: A metric showing how many of the retrieved chunks are useful for generation.
* **Recall:** A metric displaying the completeness of the retrieval results.

## Inputs

* Question
* Document
* Context
* Ground Truth Facts

## Outputs

* Fact ranks.
* Precision and Recall scores.
* Mean Reciprocal Rank, F1 Score.

### Metrics Evaluation

1. **Relating Chunks and Facts:**
   * Map the retrieved chunks with the respective ground truth facts using a Facts Matcher.
   * Generate two arrays: Context relevance and Facts ranks.
   * Context relevance represents how many ground truth facts a chunk contains.
   * Facts ranks displays the index of the chunk in the context where the fact was found (or `-1` if not found).
2. **Calculating Precision:**
   * The precision is the ratio of the retrieved chunks which include ground truth facts to total retrieved chunks.
3. **Calculating Recall:**
   * The recall is the ratio of ground truth facts found in chunks to the total number of ground truth facts.
4. **Additional Metrics:**
   * MRR (Mean Reciprocal Rank) - Mean inverse of the rank of the first correct chunk.
   * F1 Score - Harmonic mean of the precision and recal.

## Scope, Known Limitations

**Scope:** Chunks are parts of the original document. Facts are substrings of the chunks.