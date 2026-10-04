"""Download a pinned public checkpoint and verify every weight shard's LFS SHA256."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    root = Path(os.environ["LAB_DIR"])
    model_dir = Path(os.environ["MODEL_DIR"])
    repo = os.environ["MODEL_ID"]
    revision = os.environ["MODEL_REVISION"]
    assert len(revision) == 40, "Use a complete immutable model commit."
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "model-manifest.json"

    if args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        assert manifest["repo"] == repo and manifest["revision"] == revision
    else:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        assert info.sha == revision
        files = []
        for sibling in info.siblings:
            if not sibling.rfilename.endswith(".safetensors"):
                continue
            lfs = sibling.lfs
            sha = getattr(lfs, "sha256", None)
            if isinstance(lfs, dict):
                sha = lfs.get("sha256")
            if not sha:
                raise RuntimeError(f"Missing authoritative SHA256: {sibling.rfilename}")
            files.append({"path": sibling.rfilename, "bytes": sibling.size,
                          "sha256": sha})
        if not files:
            raise RuntimeError("No weight shards in model metadata.")
        manifest = {"repo": repo, "revision": revision, "files": files,
                    "weight_bytes": sum(f["bytes"] for f in files)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Weight payload: {manifest['weight_bytes'] / 2**30:.2f} GiB", flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=model_dir,
                          max_workers=4,
                          allow_patterns=["*.safetensors", "*.json", "*.model",
                                          "*.txt", "*.jinja"])

    for item in manifest["files"]:
        path = model_dir / item["path"]
        assert path.stat().st_size == item["bytes"], f"Size mismatch: {path}"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        assert digest.hexdigest() == item["sha256"], f"SHA256 mismatch: {path}"
        print(f"SHA256 OK: {item['path']}", flush=True)
    config = json.loads((model_dir / "config.json").read_text())
    assert config["model_type"] == "qwen3_moe"
    assert config["num_experts"] == 128 and config["num_experts_per_tok"] == 8
    assert config["quantization_config"]["quant_method"] == "fp8"
    assert config["quantization_config"]["weight_block_size"] == [128, 128]
    print("Pinned model identity and weight integrity verified.")


if __name__ == "__main__":
    main()
