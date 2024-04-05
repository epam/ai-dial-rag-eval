from pathlib import Path

import pandas as pd
import streamlit as st
from data import get_all_datasets_list, read_dataset

st.set_page_config(
    page_title="View data",
    layout="wide",
)

st.markdown("# View data")

data_dir = Path("data")


datasets = get_all_datasets_list()
dataset_path = st.sidebar.selectbox("Select dataset", datasets)


if st.sidebar.button("Refresh cache"):
    st.cache_data.clear()

st.markdown(f"Dataset: {dataset_path}")

show_as_dataset = st.toggle("View as dataframe", value=True)


@st.cache_data
def get_data(dataset_path):
    return pd.read_parquet(data_dir.joinpath(dataset_path))


data = read_dataset(dataset_path)

if show_as_dataset:
    st.dataframe(data, use_container_width=True)
else:
    st.table(data)
