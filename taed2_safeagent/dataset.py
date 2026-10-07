from pathlib import Path

import typer

from taed2_safeagent.data.audit import audit as run_audit
from taed2_safeagent.data.common import load_params
from taed2_safeagent.data.source import fetch as run_fetch
from taed2_safeagent.data.split import prepare as run_prepare
from taed2_safeagent.data.validation import validate as run_validate

app = typer.Typer(
    help="Prepare and check the SafeAgent dataset.", pretty_exceptions_show_locals=False
)


def run(action, params):
    try:
        action(load_params(params))
    except (ValueError, TypeError, FileNotFoundError) as error:
        typer.echo(f"Data check failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command()
def fetch(params: Path = Path("params.yaml")):
    run(run_fetch, params)
    typer.echo("Raw snapshot checked")


@app.command()
def audit(params: Path = Path("params.yaml")):
    run(run_audit, params)
    typer.echo("Data audit complete")


@app.command()
def prepare(params: Path = Path("params.yaml")):
    run(run_prepare, params)
    typer.echo("Data splits prepared")


@app.command()
def validate(params: Path = Path("params.yaml")):
    run(run_validate, params)
    typer.echo("Data validation passed")


if __name__ == "__main__":
    app()
