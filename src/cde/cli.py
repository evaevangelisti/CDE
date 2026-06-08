import typer

app: typer.Typer = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    pass

if __name__ == "__main__":
    app()
