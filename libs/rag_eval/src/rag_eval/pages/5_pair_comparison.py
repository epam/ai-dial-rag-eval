import streamlit as st

from rag_eval.data import get_evaluation_results_datasets_list, read_dataset
from rag_eval.highlight import highlight_context_by_facts

st.set_page_config(
    page_title="Pairwise comparison",
    layout="wide",
)


st.markdown("# Pairwise comparison")

datasets_list = get_evaluation_results_datasets_list()

evaluation_result_1 = st.sidebar.selectbox("First evaluation", datasets_list, index=0)

evaluation_result_2 = st.sidebar.selectbox("Second evaluation", datasets_list, index=1)

with st.spinner("Loading evaluation results..."):
    evaluation_result_1_data = read_dataset(evaluation_result_1)
    evaluation_result_2_data = read_dataset(evaluation_result_2)

    merge_columns = ["question"]
    common_columns = ["ground_truth_facts"]
    comparison_columns = ["context", "Recall"]

    comparison_data = evaluation_result_1_data[
        merge_columns + common_columns + comparison_columns
    ].merge(
        evaluation_result_2_data[merge_columns + comparison_columns],
        on=merge_columns,
        suffixes=("_1", "_2"),
        validate="1:1",
    )

# comparison_data["context_1"] = comparison_data["context_1"].apply(
#    lambda x: "\n\n".join(x.tolist()))


for column in ["context_1", "context_2"]:
    comparison_data[column] = comparison_data.apply(
        lambda row: highlight_context_by_facts(
            row, context_column=column  # noqa: B023
        ),  # noqa: B023
        axis=1,
    )


show_as_highlighted_table = st.toggle("Highlighted table", value=False)
if show_as_highlighted_table:
    highlighted = comparison_data.style.set_properties(
        width="5%", **{"text-align": "left", "escape": False}
    )
    st.markdown(highlighted.to_html(), unsafe_allow_html=True)
else:
    st.dataframe(comparison_data, use_container_width=True)
