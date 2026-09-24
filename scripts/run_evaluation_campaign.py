"""Dynamic SSH worker queue. Inventories, state and logs are private runtime inputs.

One coordinator owns a campaign. Inventory can add ready workers while it runs.
Jobs are immutable; unfinished infrastructure-interrupted jobs may resume explicitly.
No GPU/process killing, package installation, resource copying or Git writes occur here.
"""

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def remote(worker, command, timeout=None):
    return subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=6",
            worker["host"],
            command,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def execute(worker, job, manifest, log_dir):
    root = worker["root"]
    gpu = worker["gpu"]
    probe = remote(
        worker,
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits",
        timeout=30,
    )
    if probe.returncode:
        return {"status": "unavailable", "detail": "GPU probe failed"}
    gpu_row = next(
        (line for line in probe.stdout.splitlines() if line.split(",")[0].strip() == str(gpu)), None
    )
    if gpu_row is None or any(int(x.strip()) > y for x, y in zip(gpu_row.split(",")[1:], (256, 5))):
        return {"status": "busy", "detail": "GPU is not idle"}
    backend = job["backend"]
    resource = worker["backends"][backend]
    run = f"outputs/{manifest['campaign']}/runs/{job['id']}"
    command = [
        resource["python"],
        "-m",
        "efficient_vla_benchmark.cli",
        "evaluate",
        "--config",
        job["config"],
        "--model-path",
        resource["model"],
        "--libero-root",
        "third_party/benchmarks/libero",
        "--episode-manifest",
        manifest["manifest"],
        "--shard-index",
        str(job["shard_index"]),
        "--num-shards",
        str(job["num_shards"]),
        "--no-record-traces",
        "--output-dir",
        run,
    ]
    if job.get("artifact"):
        command += ["--artifact-path", resource["artifact"]]
    environment = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in environment.items())
    shell = (
        f"cd {shlex.quote(root)} && mkdir -p {shlex.quote(str(Path(run).parent))} && "
        f"flock -n outputs/dev200-gpu-{gpu}.lock env {assignments} "
        f"{shlex.join(command)} > {shlex.quote(run + '.log')} 2>&1"
    )
    started = time.time()
    result = remote(worker, shell)
    (log_dir / f"{job['id']}.ssh.log").write_text(result.stdout)
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "wall_seconds": time.time() - started,
        "host": worker["host"],
        "gpu": gpu,
        "run": run,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    # POSIX advisory lock prevents competing coordinators from double-dispatching.
    import fcntl

    lock = (args.state_dir / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.state_dir / "state.json"
    states = read(state_path) if state_path.exists() else {}
    if any(v["status"] == "running" for v in states.values()):
        raise ValueError("Inspect interrupted running jobs before explicitly resuming coordinator")
    active, cooldown = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=80) as pool:
        while True:
            specification = read(args.jobs)
            jobs = specification["jobs"]
            if len({j["id"] for j in jobs}) != len(jobs):
                raise ValueError("Duplicate campaign job IDs")
            for job in jobs:
                if job["id"] in states and states[job["id"]]["job"] != job:
                    raise ValueError("Do not change an existing job identity")
            for future in list(active):
                if future.done():
                    worker, job = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"status": "failed", "detail": type(exc).__name__}
                    states[job["id"]] = {"job": job, **result}
                    if result["status"] in {"busy", "unavailable"}:
                        cooldown[(worker["host"], worker["gpu"])] = time.time() + 60
                    print(json.dumps({"job": job["id"], **result}), flush=True)
            occupied = {(w["host"], w["gpu"]) for w, _ in active.values()}
            pending = [
                j
                for j in jobs
                if states.get(j["id"], {}).get("status") not in {"running", "completed", "failed"}
            ]
            for worker in read(args.inventory)["workers"]:
                slot = (worker["host"], worker["gpu"])
                if slot in occupied or cooldown.get(slot, 0) > time.time():
                    continue
                job = next((j for j in pending if j["backend"] in worker["backends"]), None)
                if job is None:
                    continue
                pending.remove(job)
                states[job["id"]] = {
                    "status": "running",
                    "job": job,
                    "host": worker["host"],
                    "gpu": worker["gpu"],
                }
                active[pool.submit(execute, worker, job, specification, args.state_dir)] = (
                    worker,
                    job,
                )
                occupied.add(slot)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2, sort_keys=True))
            temporary.replace(state_path)
            counts = {
                s: sum(v["status"] == s for v in states.values())
                for s in ("running", "completed", "failed", "busy", "unavailable")
            }
            print(json.dumps({"progress": counts, "total_jobs": len(jobs)}), flush=True)
            if specification.get("closed", False) and not active and not pending:
                break
            time.sleep(15)


if __name__ == "__main__":
    main()
