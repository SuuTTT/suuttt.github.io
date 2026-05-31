---
title: "research-os: A File-Backed Control Plane for GPU Research Automation"
date: 2026-05-31
description: "How to build a lightweight, inspectable research automation system that turns a high-level idea into a structured workflow — from idea queue to paper artifacts — without a database or daemon."
layout: "post"
showTableOfContents: true
math: false
katex: false
tags: ["research-infrastructure", "vastai", "gpu-automation", "mlops", "research-workflow"]
---

> **TL;DR**: research-os is a file-backed control plane that manages the full lifecycle of an ML research project — idea → experiment queue → Vast.ai GPU worker dispatch → metric collection → paper artifacts. Everything is stored as human-readable JSON/YAML/Markdown. No database. No daemon. Fully inspectable and git-committable.

---

## 1. The Problem: Research Overhead Kills Research Velocity

When you run dozens of GPU experiments across multiple projects, the management overhead compounds fast:

- Which runs are pending? Which finished? Which failed and why?
- Which Vast.ai instance is alive right now, and what's its SSH port today?
- Did that auto-rented worker ever get registered so the scheduler can see it?
- Where did I put the metrics from Phase B, seed 3?

The typical answer is a jumble of shell scripts, sticky notes in Notion, and `ssh` commands you re-derive every session. **research-os replaces all of that with a single CLI** (`ros.py`) and a folder of human-readable files you can `cat`, `git commit`, and hand off to a colleague.

---

## 2. Architecture

### 2.1 Core Principle: Files All the Way Down

```
research-os/
├── scripts/ros.py            ← single CLI entry point
├── queues/                   ← file-backed task queues (JSON)
│   ├── idea_queue.json
│   ├── benchmark_queue.json
│   ├── run_queue.json
│   └── publication_queue.json
├── workers/workers.yaml      ← worker registry
├── schemas/                  ← JSON schemas for validation
├── templates/                ← project & paper scaffolds
└── research/
    └── <project-id>/         ← one folder per research project
        ├── project.yaml
        ├── evidence/
        ├── paper/
        └── ...
```

Every queue is a JSON array. Every worker is a YAML block. You can inspect, edit, and version-control the entire system state with no special tooling — `grep`, `jq`, or a text editor all work.

### 2.2 The Workflow Pipeline

```
High-level idea
      ↓
  init-project   →  research/<id>/project.yaml scaffolded
                    idea added to idea_queue.json
      ↓
  add-run        →  command + env appended to run_queue.json
      ↓
  dispatch-runs  →  SSH to an idle Vast.ai worker
                    script launched in background (setsid + log file)
      ↓
  check-runs     →  poll process via kill -0 / exit_code file
                    status updated in run_queue.json
      ↓
  results        →  parse metric files, write to evidence/
      ↓
  publish        →  paper LaTeX + blog announcement
```

No daemon needs to run. You invoke `ros.py` whenever you want to advance the pipeline. The queue files are the durable state; the CLI reads and writes them atomically.

### 2.3 Workers

Workers are defined in `workers/workers.yaml`. Each entry specifies a `kind` (`local` or `ssh`), connection details, resource constraints, and health-check commands.

```yaml
- worker_id: vastai_auto_20260531_003650
  instance_id: 38664456
  kind: ssh
  enabled: true
  worker_pool: vastai
  ssh:
    host: ssh1.vast.ai
    port: 24456
    user: root
    key: ~/.ssh/vastai_id_ed25519
  workspace: /root
  resources:
    gpu: 1
    disk_gb: 32
  health:
    heartbeat_command: "echo ok"
    busy_command: "pgrep -f python || true"
  notes: >-
    AUTO-RENTED at 2026-05-31 00:35:14. RTX A4000, $0.094/hr,
    DLP/$=230.8. Washington, US.
```

The dispatcher SSH-es into idle workers, writes a run script, and launches it detached. No persistent connection required — the worker runs the job independently and writes an exit code file when done.

---

## 3. Vast.ai Integration

Vast.ai provides cheap on-demand GPU instances, but they have a practical pain point: **SSH ports and hosts change when instances restart**. research-os handles this with three CLI commands.

### 3.1 `vastai-instances` — Live Overview

```bash
python3 scripts/ros.py vastai-instances
```

```
        ID  GPU                     Status        $/hr  SSH                           Worker
────────────────────────────────────────────────────────────────────────────────────────────────────────
  37906248  RTX 3060                running     0.0659  ssh8.vast.ai:26248            vastai_worker_2
  38342607  GTX 1660 S              running     0.0967  ssh4.vast.ai:22606            vastai_worker_3
  38664456  RTX A4000               running     0.1027  ssh1.vast.ai:24456            vastai_auto_20260531_003650
  38702735  RTX 3070 laptop         running     0.0556  ssh5.vast.ai:22734            vastai_auto_20260531_050014
  34824701  RTX 3060                running     0.0566  ssh5.vast.ai:24700            (untracked)
```

This fetches live state from the Vast.ai API and cross-references it with `workers.yaml`. Untracked instances — visible on the account but not registered as workers — are flagged so you can decide whether to add or ignore them.

### 3.2 `vastai-reconcile` — Drift Detection and Auto-Fix

SSH ports change silently. Without reconciliation, the dispatcher will keep trying a stale port and silently skip those workers.

```bash
# detect drift
python3 scripts/ros.py vastai-reconcile

# auto-apply all fixes
python3 scripts/ros.py vastai-reconcile --fix
```

Example output:

```
=== Worker Reconciliation at 2026-05-31 06:49:22 ===
Live instances: 10  |  Tracked workers: 6

  [✓] vastai_worker_2   inst=37906248  running   ssh8.vast.ai:26248  [RTX 3060  $0.0659/hr]
      ⚠ PORT CHANGED: yaml=26249  live=26248
      ✓ Updated port 26249 → 26248
  [✓] vastai_worker_3   inst=38342607  running   ssh4.vast.ai:22606  [GTX 1660 S  $0.0967/hr]
      ⚠ PORT CHANGED: yaml=22607  live=22606
      ✓ Updated port 22607 → 22606
  [GONE]    vastai_auto_20260530_175359 — instance 38614511 not found
            ✓ Marked vastai_auto_20260530_175359 enabled: false

All tracked workers are consistent with live state.
```

`--fix` patches the YAML directly, so the next `dispatch-runs` sees correct SSH coordinates.

### 3.3 `vastai-watcher` — Background GPU Hunter

A background shell script (`~/vastai/watch_cheap.sh`) polls every 5 minutes for GPU offers under \$0.10/hr and auto-rents any instance with a DLP/$ ratio above 180. research-os manages this process:

```bash
python3 scripts/ros.py vastai-watcher status     # is it running?
python3 scripts/ros.py vastai-watcher start      # launch in background
python3 scripts/ros.py vastai-watcher stop       # kill
python3 scripts/ros.py vastai-watcher log        # tail the last 40 lines of the offers log
python3 scripts/ros.py vastai-watcher blacklist  # show banned machine IDs
```

When the watcher auto-rents a new instance, it appends a fully-formed worker entry to `workers.yaml` — including the `instance_id` field so `vastai-reconcile` can track it.

---

## 4. Key CLI Commands Reference

| Command | What it does |
|---|---|
| `init-project` | Scaffold a new project folder + manifest, add first idea to queue |
| `add-run` | Append a pending run (command + env) to `run_queue.json` |
| `dispatch-runs` | SSH to idle worker, launch run script detached |
| `check-runs` | Poll running jobs, update status, print last log lines |
| `worker-status` | SSH heartbeat + `nvidia-smi` for all enabled workers |
| `add-worker` | Register a new SSH worker (Vast.ai or any machine) |
| `vastai-reconcile` | Detect + fix port drift and destroyed instances |
| `vastai-instances` | Live instance list vs workers.yaml |
| `vastai-watcher` | Start/stop/status the background GPU hunter |
| `vast-hunt` | Search for offers meeting a project's hardware requirements |
| `project-status` | Dashboard: all projects with run counts and current state |
| `key-smoke` | Validate API keys (Vast.ai, W&B, HuggingFace, GitHub) |

---

## 5. End-to-End Example: Running a Sweep on a Cheap GPU

Here is a complete walkthrough of how research-os was used for the SE-TS paper experiments — testing whether spectral entropy regularisation improves neural time-series forecasting on ETT datasets.

### Step 1: Initialise the Project

```bash
python3 scripts/ros.py init-project \
  --title "Spectral Entropy as Regulariser for Time-Series Forecasting" \
  --idea "Add spectral entropy penalty to TimesNet loss; does it improve ETT MSE?" \
  --target "match or beat TimesNet paper baseline MSE on ETTh1/ETTm1" \
  --metric "MSE"
```

This creates `research/structural_entropy_timeseries/` with subdirectories for benchmarks, probes, evidence, and the paper LaTeX source. An idea entry lands in `idea_queue.json`.

### Step 2: Find a Cheap GPU

```bash
python3 scripts/ros.py vast-hunt \
  --profile pytorch \
  --max-dph 0.10 \
  --ref-hours 4.5 \
  --ref-dlp 12.4
```

Output shows ranked offers with estimated total cost for the sweep. Pick the best one, note the offer ID.

### Step 3: Register and Set Up the Worker

```bash
# Auto-fill SSH details from Vast.ai API
python3 scripts/ros.py add-worker \
  --worker-id vastai_worker_2 \
  --vastai-instance-id 37906248 \
  --pool vastai

# Install dependencies on the worker
python3 scripts/ros.py setup-worker --worker-id vastai_worker_2
```

`setup-worker` SSH-es in, installs git/python3, creates workspace directories, and syncs `.env.local` (which carries the W&B key).

### Step 4: Queue the Runs

One entry per hyperparameter configuration:

```bash
python3 scripts/ros.py add-run \
  --project-id structural_entropy_timeseries \
  --worker-pool vastai \
  --command "cd /root/TimesNet && python run.py \
    --model TimesNet --data ETTh1 \
    --pred_len 96 --lambda_se 0.0 --seed 1" \
  --priority 5

# repeat for lambda_se in {0.01, 0.05, 0.10} × seeds {1,2,3} × horizons {96,192}
```

### Step 5: Dispatch

```bash
python3 scripts/ros.py dispatch-runs --worker-pool vastai
```

```
dispatched r1a2b3c4 → vastai_worker_2  pid=12043
dispatched r5d6e7f8 → vastai_worker_3  pid=9871
...
8/8 dispatched
```

Each run launches as a detached `setsid` process, writing stdout to a log file on the worker. The dispatcher does not need to stay connected.

### Step 6: Monitor Progress

```bash
python3 scripts/ros.py check-runs
```

```
r1a2b3c4: running  pid=12043  | Epoch 3/10 | train_loss=0.4821 val_loss=0.5103
r5d6e7f8: done     exit=0
    Epoch 10/10 | MSE=0.3847 MAE=0.4021
```

Completed runs get their status flipped to `done` or `failed` in `run_queue.json` with the exit code and timestamp.

### Step 7: Reconcile Workers Before the Next Batch

Vast.ai can reassign SSH ports when an instance restarts overnight. Before dispatching a new batch:

```bash
python3 scripts/ros.py vastai-reconcile --fix
```

Any stale ports are patched automatically so the next dispatch works first try.

### Result

After six experiment phases (96 + 24 + 12 + 49 + 37 + 73 = 291 run-result rows), the evidence folder held everything needed to write the paper: aggregate summaries per phase, per-seed breakdowns, and a clear null result. Pre-projection spectral entropy is not a productive regularisation target for TimesNet or PatchTST on ETT. A clean null result, ready to publish.

---

## 6. Design Decisions Worth Stealing

**File queues, not a database.** JSON arrays are trivially inspectable with `jq`, `cat`, and a text editor. You can hand-edit a stuck job, `grep` for a run ID in logs, or `git diff` the queue to see what changed. A SQLite database would be more robust but far less transparent.

**No daemon.** The CLI is invoked manually (or via cron). This means there is no background process to crash, no port to open, and no service to restart after a reboot. The entire system survives a control-plane reboot because all state is in files.

**`setsid` for detached runs.** Jobs are launched with `setsid ... </dev/null > logfile 2>&1 &`, which detaches them from the SSH session. The connection can drop and the experiment keeps running. Only the exit code file matters for status.

**Reconcile as a regular operation.** Vast.ai port drift is expected, not exceptional. Running `vastai-reconcile --fix` before every dispatch batch ensures stale SSH coordinates are corrected silently. Think of it like `git pull` before a push.

**`instance_id` in every worker entry.** The watcher auto-rents instances and appends worker YAML. If the `instance_id` field is missing, reconcile cannot match the entry to a live instance. All paths that create workers (CLI `add-worker`, watcher script) must write this field.

---

## 7. Getting Started

### Prerequisites

- Python 3.10+, `pyyaml`, a Vast.ai account with CLI configured (`vastai set api-key ...`)
- An SSH keypair for Vast.ai workers (`~/.ssh/vastai_id_ed25519`)
- Optional: W&B, HuggingFace, GitHub tokens in `~/.env.local`

### Clone and Initialise

```bash
git clone https://github.com/YOUR_ORG/research-os.git
cd research-os

# Verify API keys
python3 scripts/ros.py key-smoke

# Check for cheap GPUs
python3 scripts/ros.py vast-hunt --profile pytorch --max-dph 0.10

# Create your first project
python3 scripts/ros.py init-project \
  --title "My First Experiment" \
  --idea "Does X outperform Y on dataset Z?" \
  --target "beat Y baseline by 2% MSE"
```

### Minimal Loop

```bash
# 1. Queue a run
python3 scripts/ros.py add-run \
  --project-id my_first_experiment \
  --worker-pool vastai \
  --command "python train.py --config baseline.yaml"

# 2. Dispatch to idle workers
python3 scripts/ros.py dispatch-runs

# 3. Check back later
python3 scripts/ros.py check-runs

# 4. Reconcile if workers restarted overnight
python3 scripts/ros.py vastai-reconcile --fix
```

That's the whole loop. Add more `add-run` calls, adjust priorities, read the log files on the worker — everything else is an alias for this cycle.

---

## 8. What's Next

- **`vastai-blacklist add <machine_id>`** — convenience command to ban ghost machines without hand-editing a text file
- **Watcher config externalised** — `AUTO_RENT_THRESHOLD` and `INTERVAL` currently hardcoded in the shell script; move to a config file that `vastai-watcher status` can display
- **Host-change auto-fix in reconcile** — port changes are auto-patched; SSH host changes still require a manual YAML edit
- **Auto-parsing of metric files** — right now `check-runs` just tails the log; a `metric_parser` field in the run spec could auto-extract the final validation loss and write it to the evidence folder

---

The core insight behind research-os is simple: **a well-structured folder of text files is more durable than any database**, and a CLI you can `cat` is more trustworthy than a web dashboard you can't inspect. When the RTX 3060 drops offline at 2 AM and restarts on a different port, `vastai-reconcile --fix` fixes it in one command and the morning dispatch batch finds healthy workers.

Research moves faster when the infrastructure gets out of the way.
