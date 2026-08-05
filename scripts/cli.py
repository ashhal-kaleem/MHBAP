"""
MHBAP CLI — developer utility commands.
Usage: mhbap <command>
"""
import typer

app = typer.Typer(name="mhbap", help="MHBAP developer CLI")


@app.command()
def dev() -> None:
    """Start FastAPI backend in dev mode."""
    import subprocess
    subprocess.run(["uvicorn", "app.Main:app", "--reload", "--port", "8000"])


@app.command()
def migrate() -> None:
    """Run Alembic migrations to latest revision."""
    import subprocess
    subprocess.run(["alembic", "upgrade", "head"], cwd="backend")


@app.command()
def seed() -> None:
    """Seed the database with demo session data (Phase 10)."""
    typer.echo("Seed not yet implemented — available in Phase 10.")


@app.command()
def status() -> None:
    """Print current phase status from PROJECT_STATUS.md."""
    with open("PROJECT_STATUS.md") as f:
        typer.echo(f.read())


if __name__ == "__main__":
    app()
