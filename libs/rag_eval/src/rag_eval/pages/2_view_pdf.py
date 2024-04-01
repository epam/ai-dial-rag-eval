import base64

import streamlit as st

st.markdown("# View pdf")

st.markdown("Not working for now")


def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'  # noqa: B950
    st.markdown(pdf_display, unsafe_allow_html=True)
