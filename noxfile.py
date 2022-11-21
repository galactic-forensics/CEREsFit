"""Configuration file for testing the package with ``nox``."""

import nox


nox.options.sessions = "lint", "tests", "xdoctest"

package = "curefit"
locations = "curefit", "tests", "noxfile.py"
python_suite = ["3.11", "3.10", "3.9", "3.8"]
python_main = "3.10"


@nox.session(python=python_main)
def build(session):
    """Pack iniabu for release on PyPi."""
    session.install("flit")
    session.run("flit", "build")


@nox.session(python=python_main)
def lint(session):
    """Lint project using ``flake8``."""
    args = session.posargs or locations
    session.install("--upgrade", "pip")
    session.install(".[test]")
    session.run("flake8", *args)


@nox.session(python=python_suite)
def tests(session):
    """Test the project using ``pytest``."""
    session.install("--upgrade", "pip")
    session.install(".[test]")
    session.run("pytest")


@nox.session(python=python_main)
def safety(session):
    """Safety check for all dependencies."""
    session.install("--upgrade", "pip")
    session.install("safety")
    session.install(".[dev,test]")
    session.run(
        "safety",
        "check",
        "--full-report",
    )


@nox.session(python=python_main)
def xdoctest(session):
    """Test docstring examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install("--upgrade", "pip")
    session.install("xdoctest[all]")
    session.install(".[dev,test]")
    session.run("python", "-m", "xdoctest", package, *args)
