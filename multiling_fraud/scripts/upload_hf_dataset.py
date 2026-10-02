#!/usr/bin/env python3
"""Push the hf_dataset/ tree to a HuggingFace dataset repo.

Needs `huggingface_hub` and an HF token (write scope). Install/run inside a
conda env, e.g.:

    conda activate fraudgen           # or any env with pip
    pip install -U huggingface_hub
    export HF_TOKEN=$(cat /users/msingh/GPU/.hf_token)   # or your write token
    python scripts/upload_hf_dataset.py --repo <user>/<dataset-name>

Uploads only what you point at, so phases can go up incrementally:
    --include "README.md" "data/**"          # Phase 1: text only (default)
    --include "audio/**"                      # Phase 2: audio
    --include "preferences/**"                # Phase 3: preference pairs
"""
import argparse, os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. girish/multiling-fraud")
    ap.add_argument("--root", default=os.path.join(PROJ, "hf_dataset"))
    ap.add_argument("--include", nargs="*",
                    default=["README.md", "data/**"],
                    help="glob patterns to upload (default: text only)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--message", default="Update multilingual fraud dataset")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo
    if not args.token:
        raise SystemExit("Set HF_TOKEN (write scope) or pass --token")

    create_repo(args.repo, repo_type="dataset", private=args.private,
                exist_ok=True, token=args.token)
    api = HfApi(token=args.token)
    # upload_large_folder handles big/many files (audio) with resume + retries
    api.upload_large_folder(
        repo_id=args.repo, repo_type="dataset", folder_path=args.root,
        allow_patterns=args.include,
    )
    print(f"[hf] pushed {args.include} -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
