#!/usr/bin/env python
# clone swe-bench-v and swe-gym repo for code search env
#
# pip install datasets gitpython tqdm filelock

import os, shutil, subprocess, tempfile, argparse, json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from filelock import FileLock
from multiprocessing import Pool, cpu_count


DATA_DIR = Path("gym_data")  # snapshots per instance
CACHE_DIR = Path("_repo_cache")  # bare/partial repos (shared among workers)
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


def ensure_repo(repo_slug: str) -> Path:
    """
    Return a *partial* bare clone for <github_user/repo>.
    The clone is shared across processes; FileLock prevents races.
    """
    repo_key = repo_slug.replace("/", "__")
    bare_path = CACHE_DIR / f"{repo_key}.git"
    lock_path = bare_path.with_suffix(".lock")

    with FileLock(str(lock_path)):
        if bare_path.exists():
            return bare_path

        url = f"https://github.com/{repo_slug}.git"

        # Try different clone strategies
        clone_strategies = [
            # Strategy 1: Partial clone with blob filter
            lambda: subprocess.run([
                "git", "clone", "--bare", "--filter=blob:none", url, str(bare_path)
            ], check=True, stderr=subprocess.DEVNULL),

            # Strategy 2: Shallow clone with depth
            lambda: subprocess.run([
                "git", "clone", "--bare", "--depth=50", url, str(bare_path)
            ], check=True, stderr=subprocess.DEVNULL),

            # Strategy 3: Full clone (last resort)
            lambda: subprocess.run([
                "git", "clone", "--bare", url, str(bare_path)
            ], check=True, stderr=subprocess.DEVNULL),
        ]

        for i, strategy in enumerate(clone_strategies, 1):
            try:
                strategy()
                break  # Success, exit loop
            except subprocess.CalledProcessError as e:
                if i == len(clone_strategies):
                    # All strategies failed
                    raise RuntimeError(f"All clone strategies failed for {repo_slug}: {e}")
                # Clean up failed attempt
                if bare_path.exists():
                    shutil.rmtree(bare_path, ignore_errors=True)
                continue  # Try next strategy

    return bare_path


def have_commit(repo: Path, sha: str) -> bool:
    """True if <sha> exists in repo."""
    return (
            subprocess.run(
                ["git", "--git-dir", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
    )


def fetch_commit(repo: Path, sha: str):
    """Fetch <sha> into the bare repo *with* blobs for that commit."""
    strategies = [
        # Strategy 1: Try to fetch the specific commit
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "origin", sha
        ], check=True, stderr=subprocess.DEVNULL),

        # Strategy 2: Fetch all branches and tags
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "origin", "+refs/*:refs/*"
        ], check=True, stderr=subprocess.DEVNULL),

        # Strategy 3: Convert to full clone (unshallow)
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "--unshallow"
        ], check=True, stderr=subprocess.DEVNULL),

        # Strategy 4: Complete refetch with full history
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "--all", "--unshallow"
        ], check=True, stderr=subprocess.DEVNULL),
    ]

    for i, strategy in enumerate(strategies, 1):
        try:
            strategy()
            return  # Success, exit early
        except subprocess.CalledProcessError:
            if i == len(strategies):
                # All strategies failed, raise the last error
                raise RuntimeError(f"All fetch strategies failed for commit {sha[:7]}")
            continue  # Try next strategy


def export_commit(repo: Path, sha: str, dest: Path):
    """Export tree for <sha> (fetch first if needed)."""
    if dest.exists():
        return

    # Check if we have the commit, if not try to fetch it
    if not have_commit(repo, sha):
        fetch_commit(repo, sha)
        if not have_commit(repo, sha):
            raise RuntimeError(f"commit {sha[:7]} still missing after fetch")

    tmpdir = Path(tempfile.mkdtemp(dir=dest.parent))
    try:
        tarpath = tmpdir / "repo.tar"
        with open(tarpath, "wb") as f:
            subprocess.run(
                ["git", "--git-dir", str(repo), "archive", sha],
                check=True,
                stdout=f,
            )
        subprocess.run(["tar", "-xf", tarpath, "-C", str(tmpdir)], check=True)
        tarpath.unlink()
        tmpdir.rename(dest)
    finally:
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Per-instance worker
def process_instance(row):
    repo_slug = row["repo"]
    commit_hash = row["base_commit"]
    inst_id = row["instance_id"]
    target_dir = DATA_DIR / inst_id / "testbed"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(target_dir):
        return {"status": "success", "instance_id": inst_id}

    try:
        repo = ensure_repo(repo_slug)
        export_commit(repo, commit_hash, target_dir)
        return {"status": "success", "instance_id": inst_id}
    except Exception as e:
        return {
            "status": "error",
            "instance_id": inst_id,
            "repo": repo_slug,
            "commit": commit_hash,
            "error": str(e)
        }


# --------------------------------------------------------------------------- #

def main():
    workers = 100

    # Load both datasets
    ds1 = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    ds2 = load_dataset("SWE-Gym/SWE-Gym", split="train")
    rows = list(ds1) + list(ds2)  # make it picklable
    failed_instances = []
    failed_ids = []
    successful_count = 0

    with Pool(processes=workers) as pool:
        for res in tqdm(pool.imap_unordered(process_instance, rows),
                        total=len(rows),
                        desc="Processing instances"):
            if res["status"] == "success":
                successful_count += 1
            else:
                failed_instances.append(res)
                failed_ids.append(res["instance_id"])
                tqdm.write(f"[ERROR] {res['instance_id']}: {res['error']}")

    # Save failed instances to bad.json
    if failed_instances:
        # Check if bad.json already exists and merge
        existing_failures = []
        bad_json_path = Path("bad.json")
        if bad_json_path.exists():
            try:
                with open(bad_json_path, "r") as f:
                    existing_failures = json.load(f)
                print(f"Found {len(existing_failures)} existing failures in bad.json")
            except Exception as e:
                print(f"Warning: Could not read existing bad.json: {e}")

        # Merge and deduplicate by instance_id
        all_failures = existing_failures + failed_instances
        unique_failures = {}
        for failure in all_failures:
            unique_failures[failure["instance_id"]] = failure

        final_failures = list(unique_failures.values())

        with open("bad.json", "w") as f:
            json.dump(final_failures, f, indent=2)
        print(f"\nSaved {len(final_failures)} total failed instances to bad.json")
        print(f"  New failures: {len(failed_instances)}")
        print(f"  Previous failures: {len(existing_failures)}")
    else:
        print("\nNo new failures to save.")

    print(f"\nSummary:")
    print(f"  Successful: {successful_count}")
    print(f"  Failed: {len(failed_instances)}")
    print(f"  Total: {len(rows)}")

    # Print failed instance IDs as a list
    if failed_ids:
        print(f"\nFailed instance IDs:")
        print(failed_ids)


if __name__ == "__main__":
    main()