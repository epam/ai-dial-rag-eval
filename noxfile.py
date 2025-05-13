import nox

nox.options.sessions = ("lint", "test")
nox.options.reuse_existing_virtualenvs = True

LOCATIONS = ("src", "tests", "noxfile.py")
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def test(session):
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
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)


@nox.session(python=["3.11"])
def lint(session):
    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--with", "lint", external=True)
    session.run("flake8", *args)
    session.run("pyright", *args)


@nox.session(python=["3.11"])
def black(session):
    args = session.posargs or LOCATIONS
    session.install("black")
    session.run("black", *args)


@nox.session(python=["3.11"])
def isort(session):
    args = session.posargs or LOCATIONS
    session.install("isort")
    session.run("isort", *args)


@nox.session(python=["3.11"])
def format(session):
    session.notify("black")
    session.notify("isort")
