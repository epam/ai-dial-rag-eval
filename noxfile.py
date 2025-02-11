import nox

nox.options.sessions = ("lint", "tests", "typecheck")
nox.options.reuse_existing_virtualenvs = True

LOCATIONS = ("src", "tests", "noxfile.py")
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
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
    session.install(
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-isort",
        "pyright",
    )
    session.run("flake8", *args)


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session):
    args = session.posargs or ("src", "tests")
    session.run("poetry", "install", external=True)
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
