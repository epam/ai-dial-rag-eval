def pytest_addoption(parser):
    parser.addoption(
        "--llm-mode",
        action="append",
        default=None,
        help="Fake or real llm mode",
    )
