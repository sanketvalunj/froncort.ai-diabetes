from pathlib import Path
import typer

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.loader import load_dataset
from app.retrieval.parser import parse_trials
from app.retrieval.vectorstore import FAISSVectorStore
from config.settings import settings

app = typer.Typer(help="Build FAISS index from trial dataset.")


@app.command()
def build(
    data_path:   Path = typer.Option(settings.paths.data,         "--data-path",   "-d"),
    output_path: Path = typer.Option(settings.paths.vector_store, "--output-path", "-o"),
) -> None:
    """Build FAISS index from trial dataset."""
    typer.echo(f"Loading data from {data_path}...")
    raw    = load_dataset(data_path)
    trials = parse_trials(raw if isinstance(raw, list) else raw.get("trials", raw))
    typer.echo(f"Parsed {len(trials)} trials.")

    chunks:   list[str]  = []
    metadata: list[dict] = []
    for trial in trials:
        overview = f"{trial.title}. {trial.description}".strip()
        chunks.append(overview)
        metadata.append({"trial_id": trial.id, "source": "overview", "text": overview})
        for criterion in trial.inclusion_criteria + trial.exclusion_criteria:
            label = "Inclusion" if criterion.is_inclusion else "Exclusion"
            text  = f"{label}: {criterion.description}"
            chunks.append(text)
            metadata.append({"trial_id": trial.id, "criterion_id": criterion.id,
                              "source": label.lower(), "text": text})

    typer.echo(f"Embedding {len(chunks)} chunks...")
    emb_svc    = EmbeddingService(settings.embeddings.model)
    embeddings = emb_svc.embed_batch(chunks)

    typer.echo("Building FAISS index...")
    store = FAISSVectorStore(output_path)
    store.build(embeddings, metadata)
    store.save()
    typer.echo(f"Index saved to {output_path}")
    typer.echo(f"  • {len(chunks)} vectors, dimension {embeddings.shape[1]}")


if __name__ == "__main__":
    app()
