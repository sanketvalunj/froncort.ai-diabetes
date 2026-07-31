"""
Build the FAISS vector index from the trial dataset.

Chunking strategy (requirement 6):
  - Each criterion description is already a single short chunk and is indexed as-is.
  - Trial overview text (title + description) is split into overlapping word-based
    chunks of at most settings.retrieval.chunk_size words with a
    settings.retrieval.chunk_overlap word overlap.
  - This keeps every indexed vector within the configured token budget.
"""

from pathlib import Path
import typer

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.loader import load_dataset
from app.retrieval.parser import parse_trials
from app.retrieval.vectorstore import FAISSVectorStore
from config.settings import settings

app = typer.Typer(help="Build FAISS index from trial dataset.")


def _word_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split *text* into overlapping word-based chunks.

    Each chunk contains at most *chunk_size* words; consecutive chunks share
    the last *chunk_overlap* words of the previous chunk so context is not
    lost at boundaries.
    """
    words = text.split()
    if not words:
        return []
    step   = max(1, chunk_size - chunk_overlap)
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += step
    return chunks


@app.command()
def build(
    data_path:   Path = typer.Option(settings.paths.data,         "--data-path",   "-d"),
    output_path: Path = typer.Option(settings.paths.vector_store, "--output-path", "-o"),
) -> None:
    """Build FAISS index from trial dataset."""
    chunk_size    = settings.retrieval.chunk_size    # 250 words
    chunk_overlap = settings.retrieval.chunk_overlap # 30  words

    typer.echo(f"Loading data from {data_path}...")
    raw    = load_dataset(data_path)
    trials = parse_trials(raw if isinstance(raw, list) else raw.get("trials", raw))
    typer.echo(f"Parsed {len(trials)} trials.")

    chunks:   list[str]  = []
    metadata: list[dict] = []

    for trial in trials:
        # ── Overview — split into word-bounded chunks ──────────────────────
        overview = f"{trial.title}. {trial.description}".strip()
        for i, chunk in enumerate(_word_chunks(overview, chunk_size, chunk_overlap)):
            chunks.append(chunk)
            metadata.append({
                "trial_id": trial.id,
                "source":   "overview",
                "chunk":    i,
                "text":     chunk,
            })

        # ── Criteria — each description is already short; index as one chunk
        for criterion in trial.inclusion_criteria + trial.exclusion_criteria:
            label = "Inclusion" if criterion.is_inclusion else "Exclusion"
            text  = f"{label}: {criterion.description}"
            # Still apply chunking in case a criterion description is unusually long
            for i, chunk in enumerate(_word_chunks(text, chunk_size, chunk_overlap)):
                chunks.append(chunk)
                metadata.append({
                    "trial_id":    trial.id,
                    "criterion_id": criterion.id,
                    "source":      label.lower(),
                    "chunk":       i,
                    "text":        chunk,
                })

    typer.echo(f"Embedding {len(chunks)} chunks "
               f"(chunk_size={chunk_size}, overlap={chunk_overlap})...")
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
