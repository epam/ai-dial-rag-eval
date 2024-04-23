import pandas as pd
import streamlit as st

from rag_eval.data import get_evaluation_results_datasets_list, read_dataset

st.set_page_config(
    page_title="Comparison",
    layout="wide",
)

st.markdown("# Comparison")

# refresh with icon
if st.button("Refresh"):
    st.cache_data.clear()


all_results = pd.DataFrame()

with st.spinner("Loading evaluation results..."):
    res_datasets = get_evaluation_results_datasets_list()
    for dataset in res_datasets:
        data = read_dataset(dataset)
        # Add data as a row to the dataframe
        row = pd.DataFrame(
            {
                "Dataset": [dataset.display_name],
                "Recall": [data["Recall"].mean()],
                "Recall details": [data["Recall"].tolist()],
            }
        )
        print(data["Recall"].tolist())
        print()

        all_results = pd.concat([all_results, row], ignore_index=True)


st.dataframe(
    all_results,
    use_container_width=True,
    column_config={
        "Dataset": st.column_config.TextColumn(width="medium"),
        "Recall": st.column_config.NumberColumn(
            min_value=0,
            max_value=1,
        ),
        "Recall details": st.column_config.BarChartColumn(
            width="large",
            y_min=0,
            y_max=1,
        ),
    },
)
