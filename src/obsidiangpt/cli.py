"""CLI entry point for ObsidianGPT."""
from __future__ import annotations

import argparse
import os
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
            "When GEMINI_API_KEY is set, uses Google Gemini for planning and synthesis.\n"
            "Falls back to offline/deterministic mode otherwise."
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
            "Gemini model to use (default: env GEMINI_MODEL or 'gemini-2.0-flash'). "
            "Example: gemini-2.5-flash"
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

    # Resolve model
    model = (
        args.model
        or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # Imports deferred so CLI starts fast even when deps are missing
    from .config import RunConfig
    from .pipeline import run_goal

    cfg = RunConfig(
        vault_root=vault,
        output_dir=args.out.expanduser().resolve() if args.out else None,
        dry_run=args.dry_run,
        top_k_chunks=args.top_k,
        gemini_model=model,
    )

    api_key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    ollama_model = os.environ.get("OLLAMA_MODEL", "").strip()

    if ollama_model:
        backend_label = f"{ollama_model}  (Ollama local)"
    elif api_key_present:
        backend_label = f"{model}  (Gemini cloud)"
    else:
        backend_label = f"{model}  (offline/deterministic)"

    print(f"Model  : {backend_label}")
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
