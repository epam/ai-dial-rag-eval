import pandas as pd
import streamlit as st
from stqdm import stqdm

from rag_eval.data import get_answers_datasets_list, read_dataset, write_ground_truth
from rag_eval.extract_facts import extract_facts_raw_mixtral

st.set_page_config(
    page_title="Ground truth",
    layout="wide",
)

stqdm.pandas()

if st.button("Refresh"):
    st.cache_data.clear()


answers_path = st.selectbox("Answers", get_answers_datasets_list())

answers_data = read_dataset(answers_path)

answers_data_selections = answers_data.copy()
answers_data_selections.insert(0, "Select", False)

answers_data_edited = st.data_editor(
    answers_data_selections,
    column_config={"Select": st.column_config.CheckboxColumn(required=True)},
    use_container_width=True,
    disabled=answers_data.columns,
)
answers_data_selected = answers_data_edited[answers_data_edited["Select"]]


if "facts" not in st.session_state:
    st.session_state.facts = pd.DataFrame()

if st.button("Extract facts", type="primary"):
    result = answers_data_selected.progress_apply(extract_facts_raw_mixtral, axis=1)
    st.session_state.facts = pd.DataFrame(
        {
            "facts": result.apply(lambda x: x[0]),
            "validation_result": result.apply(lambda x: x[1]),
        }
    )
    st.success("Facts extracted successfully")


if not st.session_state.facts.empty:

    ground_truth_preview = pd.DataFrame(
        {
            "question": answers_data_selected["question"],
            "ground_truth_answer": answers_data_selected["answer"],
            "ground_truth_facts": st.session_state.facts["facts"].apply(
                lambda x: "\n\n".join(x if x is not None else [])
            ),
            "validation_result": st.session_state.facts["validation_result"],
        }
    )

    st.dataframe(ground_truth_preview)

    ground_truth_data = pd.DataFrame(
        {
            "question": answers_data_selected["question"],
            "ground_truth_answer": answers_data_selected["answer"],
            "ground_truth_facts": st.session_state.facts["facts"],
            "validation_result": st.session_state.facts["validation_result"],
        }
    )

    ground_truth_data = ground_truth_data[
        ground_truth_data["validation_result"] == "valid"
    ]
    ground_truth_data.drop(columns=["validation_result"], inplace=True)

    suggested_name = answers_path.display_name.replace(
        ".parquet", "_ground_truth.parquet"
    )
    ground_truth_name = st.text_input("Ground truth name", value=suggested_name)

    overwrite = st.checkbox("Overwrite existing dataset")

    if st.button("Save valid facts", type="primary", disabled=ground_truth_data.empty):
        try:
            write_ground_truth(ground_truth_name, ground_truth_data, rewrite=overwrite)
            st.success(f"Ground truth saved successfully as `{ground_truth_name}`")
        except Exception as e:
            st.error(f"Failed to save ground truth: `{e}`")
