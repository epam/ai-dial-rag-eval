import streamlit as st
import pandas as pd
from data import write_questions
from datetime import datetime


st.set_page_config(
    page_title="Questions",
    layout="wide",
)


FILE_READERS = {
    "csv": pd.read_csv,
    "json": pd.read_json,
    "parquet": pd.read_parquet,
    "xlsx": pd.read_excel,
    "xls": pd.read_excel,
}

def read_file_by_extension(uploaded_file) -> pd.DataFrame:
    extension = uploaded_file.name.split(".")[-1]
    return FILE_READERS[extension](uploaded_file)


if not "questions_data_imported" in st.session_state:
    st.session_state.questions_data_imported = pd.DataFrame({
        "question": pd.Series(dtype="str")
    }
)


with st.expander("Import file", expanded=False):
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type=list(FILE_READERS.keys()))
    if uploaded_file is not None:
        df = read_file_by_extension(uploaded_file)
        st.dataframe(df)
        suggested_question_index = df.columns.get_loc("question") if "question" in df.columns else 0
        question_column = st.selectbox(
            "Question column",
            df.columns,
            index=suggested_question_index,
        )

        if st.button("Import", type="primary"):
            questions_data = df[[question_column]].copy()
            questions_data.columns = ["question"]
            questions_data.reset_index(drop=True, inplace=True)
            st.session_state.questions_data_imported = questions_data
            st.success("Questions imported successfully")


st.session_state.questions_data_edited = st.data_editor(
    st.session_state.questions_data_imported,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "question": st.column_config.TextColumn(required=True)
    }
)


if not "dataset_name" in st.session_state:
    st.session_state.dataset_name = f"questions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.parquet"

st.session_state.dataset_name = st.text_input("Question dataset name", value=st.session_state.dataset_name)


overwrite = st.checkbox('Overwrite existing dataset')

if st.button(
    "Save questions",
    type="primary",
    disabled=st.session_state.questions_data_edited.empty
):
    try:
        questions_data_edited = st.session_state.questions_data_edited.copy()
        questions_data_edited.reset_index(drop=True, inplace=True)
        write_questions(st.session_state.dataset_name, questions_data_edited, rewrite=overwrite)
        st.success(f"Questions saved successfully as `{st.session_state.dataset_name}`")
    except Exception as e:
        st.error(f"Failed to save questions: `{e}`")
