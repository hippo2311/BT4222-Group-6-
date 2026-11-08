"""Deterministic, self-describing names for reranking experiment artifacts."""

from __future__ import annotations


def token(value: float | str) -> str:
    """Filename-safe value token; 0.10 becomes 0p10."""
    return str(value).replace(".", "p").replace("-", "m")


def artifact_name(stage: int, family: str, **parameters: float | str) -> str:
    """Return ``03_hybrid_nrms__loss_bce__heads2__seed4222`` style names."""
    required = {"objective", "seed"}
    missing = required - parameters.keys()
    if missing:
        raise ValueError("missing naming parameters: " + ", ".join(sorted(missing)))
    parts = [f"{stage:02d}_{family}"]
    parts.extend(f"{key}{token(value)}" for key, value in parameters.items())
    return "__".join(parts)


def sequence_name(
    *,
    rnn_type: str,
    objective: str,
    history_cap: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    epochs: int,
    batch_size: int,
    seed: int,
    **extra,
) -> str:
    """Name a hybrid sequence experiment, including all architecture/loss settings."""
    return artifact_name(
        3,
        "hybrid_sequence",
        rnn=rnn_type,
        objective=objective,
        hist=history_cap,
        mlp="x".join(map(str, hidden_dims)),
        drop=dropout,
        epochs=epochs,
        batch=batch_size,
        seed=seed,
        **extra,
    )
