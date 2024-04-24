import nox

nox.options.sessions = ("lint", "tests", "typecheck")
nox.options.reuse_existing_virtualenvs = True
locations = ("src", "tests", "noxfile.py")


@nox.session(python=["3.11"])
def tests(session):
    args = session.posargs or [
        "--cov=src",
        "--cov-report",
        "xml:coverage.xml",
        "--cov-report",
        "term",
    ]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)


@nox.session(python=["3.11"])
def lint(session):
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-isort",
        "pyright",
    )
    session.run("flake8", *args)


@nox.session(python=["3.11"])
def typecheck(session):
    args = session.posargs or ("src", "tests")
    session.run("poetry", "install", external=True)
    session.run("pyright", *args)


@nox.session(python=["3.11"])
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python=["3.11"])
def isort(session):
    args = session.posargs or locations
    session.install("isort")
    session.run("isort", *args)
