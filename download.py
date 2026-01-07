#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def _read_env_token(env_path: Path) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"HUGGINGFACE_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"}:
            return value or None
    return None


def _get_token() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    token = _read_env_token(repo_root / ".env")
    if token:
        return token
    return (
        os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )


def _cmd_search(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Missing dependency: huggingface_hub. Please install it first.")
        return 1

    api = HfApi(token=_get_token())
    results = api.list_models(search=args.query, limit=args.limit)
    for model in results:
        print(model.modelId)  # pyright: ignore[reportAttributeAccessIssue]
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: huggingface_hub. Please install it first.")
        return 1

    token = _get_token()
    local_dir = args.local_dir
    if local_dir is None:
        local_dir = str(Path("models") / args.model_id)
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    print(f"Downloaded to: {local_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hugging Face model utility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search models by keyword.")
    search_parser.add_argument("query", help="Search keyword, e.g. Qwen")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.set_defaults(func=_cmd_search)

    download_parser = subparsers.add_parser("download", help="Download a model by ID.")
    download_parser.add_argument(
        "model_id", help="Model ID, e.g. Qwen/Qwen2.5-1.5B-Instruct"
    )
    download_parser.add_argument("--revision", default=None)
    download_parser.add_argument("--local-dir", default=None)
    download_parser.set_defaults(func=_cmd_download)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
