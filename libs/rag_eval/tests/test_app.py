import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

os.environ["DATA_ROOT_PATH"] = "../../data"

APP_ROOT_PATH = Path(__file__).parent.parent / "src/rag_eval"

PAGES_PATH = [str(p) for p in APP_ROOT_PATH.glob("pages/*.py")]


@pytest.mark.parametrize("page_path", PAGES_PATH)
def test_app(page_path):
    app = AppTest.from_file(page_path)
    app.run()

    assert not app.exception
