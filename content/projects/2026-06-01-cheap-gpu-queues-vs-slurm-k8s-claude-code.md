---
title: "Scheduling a Bag of Disposable GPUs: File-Backed Queues vs Slurm vs Kubernetes (and Whether Cheap vast.ai Is Worth It)"
date: 2026-06-01
description: "Three home-grown file-backed schedulers for cheap, ephemeral vast.ai GPUs driven by a Claude Code agent — compared against Slurm and Kubernetes, with a best-practice recipe and a many-cheap-vs-one-big-instance cost analysis."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["infrastructure", "gpu", "vastai", "scheduling", "slurm", "kubernetes", "claude-code", "research-automation"]
---

{{< katex >}}

Most scheduling write-ups assume you own a cluster: a fixed set of trusted,
networked machines with a shared filesystem and a sysadmin. Our setup is the
opposite. We rent **a bag of disposable consumer GPUs on [vast.ai](https://vast.ai)** —
an RTX 2060 here, an A4000 there, a Titan V that may vanish at 3 a.m. — and we
drive the whole thing with a **Claude Code agent** that watches the runs, fixes
the breakages, and decides what to launch next.

That inverted setting changes which scheduler is "best." This post compares the
three file-backed schedulers we actually run — **gpu-fleet**, **research-os**, and
the **tdmpc-glass queue** — introduces **Slurm** and **Kubernetes** for readers
who've only heard the names, and then answers two practical questions:

1. What's the best-practice scheduling setup for a researcher using cheap vast.ai
   GPUs + a Claude Code agent?
2. Higher-level: for conference-paper research, is renting many `<$0.10/hr` GPUs
   actually a good idea — or should you just rent one `~$1.3/hr` high-end box?

---

## The environment that dictates everything

Before comparing schedulers, pin down the constraints, because they're unusual:

- **Ephemeral & untrusted.** A box can disappear mid-run (we've lost several). You
  did not provision it; you rent it by the hour from a stranger's datacenter.
- **No shared filesystem.** Each box is an island reached only over SSH. There is
  no NFS, no cluster network, no common `/home`.
- **Heterogeneous.** 6 GB 2060s next to 16 GB A4000s next to a 12 GB Titan V, with
  different CUDA versions, drivers, Python versions, and throughput (we saw
  ~80–150 steps/s spread across the fleet).
- **API-reported quirks.** vast.ai reports an SSH port one below the real `sshd`;
  the "richest" Python is often a venv, not `/usr/bin/python3`.
- **A non-human operator.** A Claude Code agent reads the state, reasons about it,
  and acts. State that a model can *read and explain* beats a binary it can only
  poke.

Hold this list; it's the scorecard every option below is graded against.

---

## The three home-grown schedulers

These aren't three competing tools — they're three **layers** that grew at three
different scopes. From widest to narrowest:

### 1. gpu-fleet — the cross-project visibility layer

**What it is.** A discovery + status layer that answers *"what GPUs exist across
all my projects, which are free, and who's using them?"* Core pieces:

- `core.py` — calls the vast.ai API for live instances, then SSH-probes each box
  in parallel for GPU util/mem, CPU quota, disk, installed libs, and outbound
  network. It's the single source of truth for "what do I have and what's free."
- `web.py` — a non-blocking dashboard (port 5050): a background thread refreshes
  the snapshot so HTTP requests never block on SSH.
- `cli.py` / `client.py` — projects **self-report** usage: `report start/ping/done`
  tags boxes in `assignments.json` (the dashboard's *project* column) and updates
  `fleet_status.json` (the live *Projects* panel). A `run` subcommand syncs code to
  the cheapest matching free GPU and launches it.

**Model:** *pull-based discovery + push-based status.* It does **not** own a
control loop that claims and launches work; it tells you (and other projects)
what's free and who's doing what.

| Pros | Cons |
|---|---|
| Cross-project: many research efforts share one fleet view | Not a scheduler — won't auto-place or auto-retry work |
| Read-only discovery can't corrupt anything | Status is *self-reported* — a crashed job lies until reconciled |
| Dashboard makes "free GPU?" a 1-second question | No notion of priority, queueing, or fairness |
| Per-box CPU-quota probe (vast rents a cgroup quota, not cores) | You still need a per-project launcher on top |

### 2. research-os — the per-researcher research OS

**What it is.** A file-backed control plane for the *whole research lifecycle*,
not just GPUs:

```
Idea → project manifest → deep-research brief → benchmark plan
     → run queue → worker dispatch → metric collection → paper artifacts
```

Everything is human-readable JSON/YAML/Markdown — `idea_queue.json`,
`run_queue.json`, `workers/workers.yaml`, project manifests, paper templates.
The CLI is `python3 scripts/ros.py <command>`: `add-idea`, `add-run`,
`dispatch-runs`, `check-runs`, `setup-worker`, plus vast.ai automation
(`vastai-watcher` auto-rents boxes with DLPerf/$ > 180 at `<$0.10/hr`;
`vastai-reconcile` heals port/host drift; a blacklist for ghost offers).

**Model:** *inspect-driven, no always-on daemon required.* You (or the agent, or
cron) run `dispatch-runs` / `check-runs`. State lives in git, so a crash is
recoverable by reading files. A documented **worker contract** ("never destroy an
instance without explicit confirmation") encodes the safety rules.

| Pros | Cons |
|---|---|
| Spans idea→paper, multi-project — not just compute | Heavier surface area; more concepts to learn |
| No daemon: nothing to crash, everything git-committable | Dispatch is manual/cron — not continuous reactive scheduling |
| Auto-renting + reconciliation tuned for vast.ai churn | The lifecycle scaffolding is overkill for a single quick sweep |
| Agent-friendly: a model can read the queues and reason | Worker registry can drift from reality if not reconciled |

### 3. The tdmpc-glass queue — the single-project autonomous daemon

**What it is.** The narrowest and most *active* layer: the always-on scheduler
that actually runs one project's experiment sweep end to end. (It's also the
historical origin of research-os's queue design.)

- `central_queue.json` — file-backed task list; `fcntl`-locked; writes are
  tmp-file + atomic rename.
- `task_queue_daemon.py` — polls every 60 s, claims the highest-priority
  `pending` task, finds an idle GPU via a hardened `is_box_idle` check
  (a box with `n_run_benchmark_procs ≥ n_gpu` is busy; a GPU is free only at
  `mem_used ≤ 100 MiB`), **rsyncs `scripts/` + `src/` to the worker**, and
  SSH-launches the job detached. Multi-GPU boxes get a `CUDA_VISIBLE_DEVICES`
  mask per slot.
- **Auto-promotion**: when a run finishes, the daemon reads its best reward and
  appends follow-up seeds (≥600 → +5 seeds, ≥500 → +2, ≥380 → +1).
- A web dashboard (port 5055) with live learning curves; an append-only
  `runs_archive.jsonl`; a strict **one-master rule** (exactly one daemon per fleet).

**Model:** *continuous reactive control loop.* It is the only one of the three
that closes the loop — claim → launch → detect completion → promote → repeat —
without a human in the loop.

| Pros | Cons |
|---|---|
| Truly autonomous: fills idle GPUs, promotes winners, 24/7 | Single-project; the auto-promote thresholds are domain-specific |
| Tolerates box death: re-queue, archive, recover from JSON | Two daemons = duplicate launches (a real bug we had to fix) |
| Code shipped per-launch via rsync — no pre-baked images | `CODE_SHA` is a *label*, not a pin: edit code mid-flight and provenance silently rots |
| `fcntl` + tmp-rename make concurrent edits safe | Needs a per-worker env pre-built (the daemon ships code, not the venv) |

**The honest scar tissue.** Building this taught us where file-backed SSH
scheduling bites: a duplicate daemon once launched the *same* seed on a second GPU
and corrupted a CSV (two writers, one file); a stale `CODE_SHA` once made a "clean"
result irreproducible; a fresh box needed a 20-minute JAX + MuJoCo env bootstrap
before it could run anything. None of these are exotic — they're exactly the
failure modes a real scheduler is supposed to handle for you. Which raises the
obvious question…

---

## …why not just use Slurm or Kubernetes?

If you've only heard the names, here's the one-paragraph orientation for each, and
then why neither fits a bag of disposable SSH boxes.

### Slurm — the HPC batch scheduler

**What it is.** Slurm is the de-facto scheduler on academic supercomputers. A
central controller (`slurmctld`) tracks a fixed inventory of nodes; a daemon
(`slurmd`) runs on each node. You submit batch scripts with `sbatch`, request
resources (`--gres=gpu:2`, `--mem`, `--time`), and Slurm queues them into
**partitions** with **fair-share** accounting, **QOS** limits, gang scheduling,
and cgroup enforcement. It assumes a **trusted, persistent, homogeneous cluster
with a shared filesystem** (your `sbatch` script and data live on NFS that every
node sees).

**Why it doesn't fit here.** Every assumption is violated: our nodes are
untrusted spot rentals that appear and vanish, there's no shared filesystem
(Slurm jobs normally `cd` into a path every node can see — ours can't), and you'd
have to install and register `slurmd` on each ephemeral box and tear it down when
the box dies. Slurm is superb when you *own* the cluster; it has no story for
"this node might not exist in five minutes."

### Kubernetes — the container orchestrator

**What it is.** Kubernetes (K8s) is declarative: you describe the *desired state*
(a `Job` wanting 1 GPU, this image, these resources) and controllers continuously
reconcile reality toward it — rescheduling failed pods, self-healing, autoscaling.
With the NVIDIA device plugin it schedules GPUs; it handles churn and failure
*gracefully by design*. The cost is a heavy control plane (API server, etcd,
scheduler, kubelets), a container/registry pipeline, and real networking/RBAC.

**Why it doesn't fit here.** K8s wants to *join nodes into a cluster* it controls,
with container runtimes, an overlay network, and a registry. vast.ai gives you a
root SSH login to one box — not cluster membership. You'd be running a kubelet
inside someone else's container, wiring an overlay network across rented boxes
behind NAT, and pushing images to pull on `<$0.10/hr` nodes with metered
bandwidth. The failure-tolerance is exactly what we want; the operational surface
is wildly out of proportion to "run 10 RL seeds on whatever's cheap tonight."

### The scorecard

| Dimension | gpu-fleet | research-os | tdmpc-glass queue | Slurm | Kubernetes |
|---|---|---|---|---|---|
| Assumes owned/stable cluster | No | No | No | **Yes** | **Yes** |
| Needs shared filesystem | No | No | No (rsync) | Usually | No (images) |
| Tolerates nodes vanishing | Partial | Yes | Yes | Poorly | **Yes** |
| Setup cost | Low | Low | Low | High | **Very high** |
| Auto-claim + launch loop | No | Manual/cron | **Yes** | **Yes** | **Yes** |
| Priorities / fairness / accounting | No | Minimal | Priority only | **Rich** | Rich |
| Human/agent-readable state | **Yes (JSON)** | **Yes** | **Yes** | DB/CLI | API/YAML |
| Right scope | Fleet view | Idea→paper | One sweep | Owned HPC | Cloud-native infra |

The pattern is clear: **Slurm and K8s are better schedulers for clusters you own;
the file-backed trio is better for a bag of rented, disposable SSH boxes** — chiefly
because their state is plain files an agent can read, they need nothing installed
on the node beyond SSH + a venv, and they treat a vanished node as a normal event
rather than an outage. You're not "missing out" by skipping Slurm/K8s here; you'd
be paying their cluster-shaped overhead for a fleet that has no cluster shape.

---

## Best practice: cheap vast.ai + a Claude Code agent

Synthesizing the three layers into one recommended setup:

1. **Use all three layers for what each is good at.**
   - *gpu-fleet* as the shared dashboard + discovery across projects ("what's free?").
   - *research-os* as the per-researcher lifecycle + vast.ai automation (auto-rent,
     reconcile, blacklist, worker contract).
   - *a tdmpc-glass-style daemon* per active sweep as the autonomous claim/launch/
     promote loop.

2. **Keep state in flat files, in git.** JSON/YAML the agent can read, diff, and
   repair. This is the single biggest reason the home-grown stack beats Slurm/K8s
   *for an agent operator*: when something breaks, Claude Code reads
   `central_queue.json` and reasons; it can't reason about an opaque scheduler DB.

3. **Make the launcher idempotent and provenance-pinned.** Ship code per launch
   (rsync), but stamp the **real git SHA**, not a vanity label, and *freeze code
   while a batch is in flight* — our worst irreproducibility bug was editing the
   working tree while pending tasks still pointed at an old `CODE_SHA`.

4. **Assume churn; design for re-queue.** One-master rule, `fcntl` locks +
   atomic renames, an append-only run archive, and a guard so a box can't be
   double-booked (the duplicate-launch collision is the canonical failure).

5. **Pre-bake or fast-bootstrap the worker env.** The agent ships `src/`, not the
   venv. Keep a one-shot bootstrap script (pinned `jax[cuda12]==…`, `mujoco`,
   the planner repo at a fixed commit) so a fresh `<$0.10/hr` box is productive in
   minutes, not hand-holding.

6. **Let the agent absorb the ops tax.** The hidden cost of cheap GPUs is
   operational (env builds, drift, dead boxes, port quirks). A Claude Code agent on
   a monitoring loop turns that recurring tax into something close to free — which
   is precisely what tips the economics below.

---

## The big question: many cheap GPUs, or one big instance?

Now the strategic one. Two concrete vast.ai offers to anchor it:

| Offer | VRAM | CPU / RAM | CUDA | DLPerf | DLP/$/hr | Price | Reliability |
|---|---|---|---|---|---|---|---|
| **host:348700** | 96 GB, 1390 GB/s, PCIe 5.0 | EPYC 9655 96-core, 384 GB | 13.2 | 278.4 | **218.3** | $1.276/hr | 99.64% |
| **host:376923** | 32 GB, 780 GB/s | Xeon 8168 96-core, 387 GB | 12.9 | 205.7 | 160.0 | $1.286/hr | 99.48% |

And the alternative: ~10 consumer boxes (2060/3060/A4000-class) at
`$0.03–0.15/hr` each — the fleet this whole post is about.

### Reading the numbers

DLP/$ (DLPerf per dollar) is the throughput-per-dollar figure of merit. Between
the two big boxes, **host:348700 wins decisively**: same price (~$1.28/hr), but
**+36% DLP/$** (218.3 vs 160), **3× the VRAM** (96 GB vs 32 GB), PCIe 5.0 and
1390 GB/s bandwidth, newer CUDA (13.2), and *higher* reliability. Unless you
specifically need the WD_BLACK NVMe / Xeon profile of 376923, **348700 is the
strictly better high-end pick.** So the real contest is **348700 vs the cheap fleet.**

But note what the DLP/$ number *hides*: consumer cards (3060/3090-class) are
notorious for **even higher DLP/$** than datacenter parts — often 2–4× — because
you're not paying the datacenter/ECC/NVLink premium. On pure throughput-per-dollar,
the cheap fleet usually *wins*.

### When many-cheap wins

- **Embarrassingly-parallel work.** RL seed sweeps, hyperparameter grids,
  ablations — N independent jobs. Ten boxes run ten seeds at once for roughly the
  price of one big box, at higher DLP/$. This is the tdmpc-glass workload exactly,
  and it's the bread-and-butter of conference-paper experiments (you need 5 seeds ×
  many configs, not one heroic run).
- **Scale-to-zero.** Pay only while exploring; rent 20 boxes for an afternoon
  sweep, drop to 2 overnight.
- **No single point of failure.** One box dies → re-queue one seed, not lose the run.
- **Resilience to price spikes.** Spot prices move; a fleet rebalances.

### When one-big wins

- **The model doesn't fit.** A 70B LLM, a big diffusion model, long-context
  training — you need the 96 GB and the fast interconnect. A 6 GB 2060 simply can't
  hold it, and you can't shard across boxes with no cluster network.
- **Tight multi-GPU coupling.** Data/model/tensor parallelism needs NVLink/PCIe-5
  bandwidth *inside one host*; ten SSH islands can't do all-reduce at speed.
- **Wall-clock deadlines.** Heterogeneous slow boxes + churn add latency. In this
  very project, off-handoff runs crawled at ~80 steps/s on 2060s; a single fast box
  would have finished sooner even at lower DLP/$.
- **Reproducibility & simplicity.** One stable, reliable box (99.6%), one env to
  build, one provenance to track. Ephemeral boxes are where provenance bugs breed.

### The honest verdict

For **conference-paper-oriented research driven by Claude Code**, the bottleneck is
almost never peak FLOPs — it's *iteration count, seed robustness, ablation breadth,
and agent oversight*. That profile favors **many cheap GPUs + a file-backed queue +
an agent that absorbs the ops tax**: you get the highest DLP/$, parallel breadth,
and fault tolerance, and the agent handles the churn that would otherwise make
cheap GPUs a part-time sysadmin job.

**But cheap is not free, and the costs are real** — we hit every one of them in a
single project: 20-minute env bootstraps per fresh box, heterogeneous throughput,
boxes vanishing mid-run, a stale-port quirk, a duplicate-launch CSV corruption, and
a provenance bug that made a "clean" result fail to reproduce. The cheap-fleet
economics only work *because* an agent pays that tax continuously.

So the pragmatic best practice is **a hybrid, matched to the workload**:

- **Breadth/exploration phase** (sweeps, ablations, seed robustness) → **the cheap
  fleet**, orchestrated by the autonomous queue. Highest DLP/$, parallel, scale-to-zero.
- **Depth phase** (a model that needs 96 GB, multi-GPU training, or the final
  camera-ready reproductions) → **one reliable high-end box** like host:348700,
  rented for the hours you actually need it.
- **Anchor** the whole thing with a stable mid-tier box that always has the env
  built, so the agent always has somewhere to smoke-test and never starts cold.

Rent cheap to *think*; rent big to *finish*. And keep the scheduler's state in flat
files the agent can read — because in this regime, the scheduler's real job isn't
packing a cluster, it's giving a language model a world it can understand and repair.

---

*This post is grounded in a live system: a Claude Code agent running an autonomous
TD-MPC-Glass experiment sweep across a ~10-GPU vast.ai fleet, with gpu-fleet,
research-os, and the tdmpc-glass queue as the three scheduling layers described
above. The scars are real; so is the leverage.*
