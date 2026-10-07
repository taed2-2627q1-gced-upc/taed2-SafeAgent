from pathlib import Path

import typer

from taed2_safeagent.data.audit import audit as run_audit
from taed2_safeagent.data.common import load_params
from taed2_safeagent.data.source import fetch as run_fetch
from taed2_safeagent.data.split import prepare as run_prepare

app = typer.Typer(help="Prepare and check the SafeAgent dataset.")


@app.command()
def fetch(params: Path = Path("params.yaml")):
    run_fetch(load_params(params))
    typer.echo("Raw snapshot checked")


@app.command()
def audit(params: Path = Path("params.yaml")):
    run_audit(load_params(params))
    typer.echo("Data audit complete")


@app.command()
def prepare(params: Path = Path("params.yaml")):
    run_prepare(load_params(params))
    typer.echo("Data splits prepared")


if __name__ == "__main__":
    app()
