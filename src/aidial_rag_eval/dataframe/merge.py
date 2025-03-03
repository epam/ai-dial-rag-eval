import pandas as pd

from aidial_rag_eval.types import (
    ANSWERS_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    MERGED_KEY_COLUMNS,
    AnswerColumns,
    GroundTruthColumns,
    MergedColumns,
)


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
    ground_truth_copy = ground_truth[
        ground_truth.columns.intersection(GROUND_TRUTH_COLUMNS)
    ].copy()
    assert isinstance(ground_truth_copy, pd.DataFrame)
    answers_copy = answers[answers.columns.intersection(ANSWERS_COLUMNS)].copy()
    assert isinstance(answers_copy, pd.DataFrame)
    ground_truth_copy = ground_truth_copy.rename(
        columns={
            GroundTruthColumns.ANSWER.value: MergedColumns.GROUND_TRUTH_ANSWER.value
        }
    )
    ground_truth_copy[GroundTruthColumns.DOCUMENTS] = ground_truth_copy[
        GroundTruthColumns.DOCUMENTS
    ].apply(frozenset)
    answers_copy[AnswerColumns.DOCUMENTS] = answers_copy[AnswerColumns.DOCUMENTS].apply(
        frozenset
    )
    data = pd.merge(
        ground_truth_copy,
        answers_copy,
        on=MERGED_KEY_COLUMNS,
    )
    data[MergedColumns.DOCUMENTS] = answers_copy.loc[
        data.index, MergedColumns.DOCUMENTS
    ]
    data[MergedColumns.DOCUMENTS] = data[MergedColumns.DOCUMENTS].apply(list)
    return data
