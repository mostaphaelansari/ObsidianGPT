"""CLI entry point for ObsidianGPT."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from the current working directory if it exists."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(override=False)   # don't override values already in env
    except ImportError:
        pass  # python-dotenv not installed — silently skip


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="obsidiangpt",
        description=(
            "Goal-driven research over an Obsidian vault.\n"
            "Uses a cloud LLM when ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY is set\n"
            "(or LLM_PROVIDER forces one), otherwise a local Ollama model.\n"
            "Falls back to offline/deterministic mode when no LLM is reachable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vault",
        type=Path,
        required=True,
        help="Path to the vault root (folder containing your .md notes).",
    )
    parser.add_argument(
        "-g",
        "--goal",
        default=None,
        help="High-level goal. If omitted, all remaining arguments are joined as the goal.",
    )
    parser.add_argument(
        "goal_tail",
        nargs="*",
        help="Goal words when --goal is not used.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for new notes (default: <vault>/_obsidiangpt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; print the planned output path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of evidence chunks to retrieve (default: 12).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model for the active provider (default: that provider's *_MODEL env var, "
            "e.g. OLLAMA_MODEL, or its built-in default)."
        ),
    )

    args = parser.parse_args(argv)

    # Resolve goal
    if args.goal:
        goal = args.goal.strip()
    else:
        goal = " ".join(args.goal_tail).strip()
    if not goal:
        parser.error("Provide a goal via --goal or as trailing arguments.")

    # Resolve vault
    vault = args.vault.expanduser().resolve()
    if not vault.is_dir():
        print(f"Vault path is not a directory: {vault}", file=sys.stderr)
        return 2

    # Imports deferred so CLI starts fast even when deps are missing
    from .config import RunConfig
    from .llm import active_provider, resolve_model
    from .pipeline import run_goal

    cfg = RunConfig(
        vault_root=vault,
        output_dir=args.out.expanduser().resolve() if args.out else None,
        dry_run=args.dry_run,
        top_k_chunks=args.top_k,
        model=args.model,
    )

    provider = active_provider()
    print(f"Model  : {resolve_model(provider, args.model)}  ({provider})")
    print(f"Vault  : {vault}")
    print()

    plan_obj, chunks, synthesis, write_res = run_goal(cfg, goal)

    print(f"Subtasks ({len(plan_obj.subtasks)}):")
    for st in plan_obj.subtasks:
        print(f"  [{st.id}] {st.question}")
    print(f"\nEvidence chunks retrieved: {len(synthesis.source_paths)}")
    print()

    if cfg.dry_run:
        print(f"[dry-run] Would write: {write_res.path}")
    else:
        print(f"✓ Wrote: {write_res.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
