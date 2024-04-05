import streamlit as st
from data import (
    get_answers_datasets_list,
    get_ground_truth_datasets_list,
    read_answers,
    read_ground_truth,
    write_evaluation_results,
)
from evaluate import evaluation_report
from highlight import highlight_details

st.set_page_config(
    page_title="Evaluation",
    layout="wide",
)

st.markdown("# Evaluation")

ground_truth_path = st.sidebar.selectbox(
    "Ground truth", get_ground_truth_datasets_list()
)
answers_path = st.sidebar.selectbox("Answers", get_answers_datasets_list())


ground_truth_data = read_ground_truth(ground_truth_path)
with st.expander(f"Ground truth: {ground_truth_path}"):
    st.dataframe(ground_truth_data)


answers_data = read_answers(answers_path)
with st.expander(f"Answers: {answers_path}"):
    st.dataframe(answers_data)


metrics, eval_data_with_metrics = evaluation_report(ground_truth_data, answers_data)
col1, col2 = st.columns(2)
with col1:
    st.dataframe(metrics)
with col2:
    st.bar_chart(metrics)


def save_results():
    evaluation_report_name = (
        f"{ground_truth_path.display_name}_vs_{answers_path.display_name}.parquet"
    )
    try:
        write_evaluation_results(evaluation_report_name, eval_data_with_metrics)
        st.success(f"Saved evaluation results as {evaluation_report_name}")
    except Exception as e:
        st.error(f"Failed to save evaluation results: {e}")


st.button(
    "Save evaluation results",
    on_click=save_results,
    type="primary",
)

st.markdown("Eval data with metrics")

show_as_highlighted_table = st.toggle("Highlighted table", value=False)
if show_as_highlighted_table:
    highlight_df = highlight_details(eval_data_with_metrics, canonize_strings=True)
    st.markdown(highlight_df.to_html(), unsafe_allow_html=True)
else:
    metrics_columns = ["Recall", "Precision", "F1", "MRR"]
    column_order = metrics_columns + [
        col for col in eval_data_with_metrics.columns if col not in metrics_columns
    ]
    st.dataframe(
        eval_data_with_metrics,
        use_container_width=True,
        column_order=column_order,
        column_config={
            "Recall": st.column_config.NumberColumn(
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
                help="Recall metric",
            ),
        },
    )
