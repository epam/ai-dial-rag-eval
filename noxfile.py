import nox

nox.options.sessions = ("lint", "test")
nox.options.reuse_existing_virtualenvs = True

LOCATIONS = ("src", "tests", "noxfile.py")
PYTHON_VERSIONS = ["3.11", "3.12"]


@nox.session
@nox.parametrize(
    ("python", "numpy", "langchain_core"),
    [
        ("3.11", "1.26.4", "0.3.83"),
        ("3.11", "2.4.2", "1.2.0"),
        ("3.12", "1.26.4", "0.3.83"),
        ("3.12", "2.4.2", "1.2.0"),
        ("3.13", "2.4.2", "1.2.0"),
        ("3.14", "2.4.2", "1.2.0"),
    ],
)
def test(session: nox.Session, numpy: str, langchain_core: str):
    session_args = [arg.split("=")[0] for arg in session.posargs]
    mode_args = ["--llm-mode"]
    if set(session_args).issubset(mode_args):
        args = session.posargs + [
            "--cov=src",
            "--cov-report",
            "xml:coverage.xml",
            "--cov-report",
            "term",
            "--junitxml=junit.xml",
        ]
    else:
        args = session.posargs
    session.run("poetry", "sync", external=True)
    session.install(f"langchain-core=={langchain_core}", f"numpy=={numpy}")
    session.run("pytest", *args)


@nox.session(python=["3.11"])
def lint(session):
    args = session.posargs or LOCATIONS
    session.run("poetry", "sync", "--with", "lint", external=True)
    session.run("flake8", *args)
    session.run("pyright", *args)


@nox.session(python=["3.11"])
def black(session):
    args = session.posargs or LOCATIONS
    session.run("poetry", "sync", "--with", "lint", external=True)
    session.run("black", *args)


@nox.session(python=["3.11"])
def isort(session):
    args = session.posargs or LOCATIONS
    session.run("poetry", "sync", "--with", "lint", external=True)
    session.run("isort", *args)


@nox.session(python=["3.11"])
def format(session):
    session.run("poetry", "sync", "--with", "lint", external=True)
    session.notify("black")
    session.notify("isort")
