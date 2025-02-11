import pandas as pd

from aidial_rag_eval.types import AnswerColumns, GroundTruthColumns, MergedColumns

GT_KEY_COLUMNS = [GroundTruthColumns.DOCUMENTS, GroundTruthColumns.QUESTION]
A_KEY_COLUMNS = [AnswerColumns.DOCUMENTS, AnswerColumns.QUESTION]
MERGED_KEY_COLUMNS = [MergedColumns.DOCUMENTS, MergedColumns.QUESTION]


def merge_ground_truth_and_answers(
    ground_truth: pd.DataFrame, answers: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges ground_truth and collected answers dataframes.

    Parameters
    -----------
    ground_truth : pd.DataFrame
        contains the ground truth answers
        The structure of the ground truth
        is described in `aidial_rag_eval.types.GroundTruthAnswers`.

    answers : pd.DataFrame
        contains the collected answers
        The structure of the collected answers
        is described in `aidial_rag_eval.types.CollectedAnswers`.

    Returns
    ------------
    pd.DataFrame
        Returns merged ground_truth dataframe and answers dataframe.
        The structure of the df_merged
        is described in `aidial_rag_eval.types.MergedColumns`.
    """
    ground_truth_copy = ground_truth.copy()
    if GroundTruthColumns.ANSWER in ground_truth_copy.columns:
        ground_truth_copy = ground_truth_copy.rename(
            columns={GroundTruthColumns.ANSWER: MergedColumns.GROUND_TRUTH_ANSWER}
        )
    ground_truth_copy[GroundTruthColumns.DOCUMENTS] = ground_truth_copy[
        GroundTruthColumns.DOCUMENTS
    ].apply(frozenset)
    answers_copy = answers.copy()
    answers_copy[AnswerColumns.DOCUMENTS] = answers_copy[AnswerColumns.DOCUMENTS].apply(
        frozenset
    )
    ground_truth_copy = ground_truth_copy.rename(
        columns=dict(zip(GT_KEY_COLUMNS, MERGED_KEY_COLUMNS))
    )
    answers_copy = answers_copy.rename(
        columns=dict(zip(A_KEY_COLUMNS, MERGED_KEY_COLUMNS))
    )
    data = pd.merge(
        ground_truth_copy,
        answers_copy,
        on=MERGED_KEY_COLUMNS,
    )
    data[MergedColumns.DOCUMENTS] = answers_copy.loc[
        data.index, MergedColumns.DOCUMENTS
    ]
    return data
