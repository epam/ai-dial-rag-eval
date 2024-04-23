import pandas as pd
import streamlit as st
from collect_answers import ask_dial_app
from data import get_questions_datasets_list, read_dataset, write_answers
from stqdm import stqdm

st.set_page_config(
    page_title="Questions",
    layout="wide",
)

stqdm.pandas()

if st.button("Refresh"):
    st.cache_data.clear()


questions_path = st.selectbox("Questions", get_questions_datasets_list())

questions_data = read_dataset(questions_path)
with st.expander(f"Questions: {questions_path}"):
    st.dataframe(questions_data, use_container_width=True)


def ask_dial_rag(question, document_url):
    return ask_dial_app(
        question,
        "dial-rag",
        messages_template='[{{{{"role": "user", "content": "/attach {}\\n{{}}"}}}}]'.format(
            document_url
        ),
        retrieval_stage="Combined search",
    )


def ask_epam10k_semantic_search(question, document_url):
    return ask_dial_app(question, "epam10k-semantic-search")


ASK_APP = {
    "dial-rag": ask_dial_rag,
    "epam10k-semantic-search": ask_epam10k_semantic_search,
}

selected_app_key = st.radio("Select the app", list(ASK_APP.keys()))
ask_selected_app = ASK_APP[selected_app_key]

document_url = st.text_input(
    "Document URL",
    value="https://d18rn0p25nwr6d.cloudfront.net/CIK-0001352010/7ea14ea9-d8f7-4039-8342-674cfbece898.pdf",
)
# st.page_link(document_url, label="Open document")


if "collected_answers" not in st.session_state:
    st.session_state.collected_answers = pd.DataFrame()


if st.button("Collect answers", type="primary"):
    st.session_state.collected_answers = questions_data.progress_apply(
        lambda row: ask_selected_app(row["question"], document_url), axis=1
    )


st.dataframe(
    st.session_state.collected_answers,
    use_container_width=True,
)

suggested_name = questions_path.display_name.replace(
    ".parquet", f"_answers_{selected_app_key}.parquet"
)
answers_name = st.text_input("Answers name", value=suggested_name)


if st.button(
    "Save answers", type="primary", disabled=st.session_state.collected_answers.empty
):
    try:
        write_answers(answers_name, st.session_state.collected_answers)
        st.success(f"Answers saved successfully as `{answers_name}`")
    except Exception as e:
        st.error(f"Failed to save answers: `{e}`")
