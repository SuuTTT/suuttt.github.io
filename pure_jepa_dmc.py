"""PURE self-predictive JEPA on DMControl — anti-collapse taxonomy test (Proposal D2).

CORRECTION context: the prior DMControl anti-collapse study was on TD-MPC2, which is
value-anchored (reward+value+policy heads) => redundancy, NOT a pure JEPA. Here we build
a GENUINELY pure JEPA: encoder + jumpy (1-step) latent predictor + EMA target-encoder,
trained by latent self-prediction ONLY. NO reward loss, NO value loss, NO policy.

Then FREEZE the encoder and linear-probe held-out R^2 for two readouts:
  (a) GEOMETRIC : decode the true physical config qpos (pose geometry)
  (b) VALUE     : decode discounted return-to-go (RTG) computed from task reward

Anti-collapse arms (matched arch/data/steps; seed varies init+order):
  none          : latent self-prediction ONLY (does a pure JEPA collapse?)
  uniformity    : Wang&Isola uniformity term on the batch latents
  vicreg        : VICReg variance-hinge + covariance
  se            : differentiable 2D structural entropy (selib-faithful), GRAD-NORM-MATCHED
                  to the uniformity arm at init (like the DMC SE-arm)
Backbone flag --simnorm {0,1}: SimNorm (softmax simplex, TD-MPC2 default, itself an
  implicit anti-collapse prior) vs a raw LayerNorm latent. The no-simnorm 'none' run is
  the honest collapse control (does the collapse claim survive without the simplex prior?).

USAGE
  python pure_jepa_dmc.py collect --task WalkerWalk
  python pure_jepa_dmc.py train --task WalkerWalk --cond se --seed 0 --simnorm 1 --gpu 0
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HELIOS = Path("/root/tdmpc_glass/helios-rl")
sys.path.insert(0, str(HELIOS / "src"))
sys.path.insert(0, "/root/tdmpc_glass/mujoco_playground_repo")
sys.path.insert(0, "/root/tdmpc_glass/selib")

ROOT = Path("/root/tdmpc_glass/exp/proposal_D2_pure_jepa")
ROOT.mkdir(parents=True, exist_ok=True)

LD = 64           # latent dim (proposal: L=32 or 64)
V = 8             # SimNorm groups
HID = (512, 512)
BATCH = 256
STEPS = 12000
LR = 3e-4
EMA = 0.99
KNN = 15
KC = 16
TAU_S = 0.5
LAM_SE = 1.0
VICREG_W = 3.0
VICCOV_W = 1.0
VIC_GAMMA = 0.05
UNIF_W = 1.0      # uniformity weight
GAMMA = 0.97      # RTG discount


def cache_path(task):
    return ROOT / f"data_{task}.npz"


# ---------------------------------------------------------------------------
# 1. Data collection (random policy; store obs/act/next/reward + qpos + RTG)
# ---------------------------------------------------------------------------
def collect(task, n_steps_per_env=2500, num_envs=16, seed=0):
    import jax, jax.numpy as jnp
    from mujoco_playground import registry, wrapper
    env = registry.load(task, config_overrides={"impl": "jax"})
    env = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1)
    obs_dim, act_dim = int(env.observation_size), int(env.action_size)
    _reset = jax.jit(env.reset); _step = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    st = _reset(jax.random.split(rk, num_envs))
    O, A, O2, R, D, G = [], [], [], [], [], []
    t0 = time.time()
    for t in range(n_steps_per_env):
        key, ak = jax.random.split(key)
        a = jax.random.uniform(ak, (num_envs, act_dim), minval=-1, maxval=1)
        o = np.asarray(st.obs)
        qpos = np.asarray(st.data.qpos)          # geometry target at time t
        nst = _step(st, a)
        O.append(o); A.append(np.asarray(a)); O2.append(np.asarray(nst.obs))
        R.append(np.asarray(nst.reward).astype(np.float32))
        D.append(np.asarray(nst.done).astype(np.float32))
        G.append(qpos.astype(np.float32))
        st = nst
        if t % 500 == 0:
            print(f"[collect {task}] {t}/{n_steps_per_env} {time.time()-t0:.0f}s", flush=True)
    O = np.stack(O); A = np.stack(A); O2 = np.stack(O2)      # (T,ne,·)
    R = np.stack(R); D = np.stack(D); G = np.stack(G)
    T, ne = R.shape
    # discounted return-to-go, backward, reset at episode boundary (done)
    RTG = np.zeros_like(R)
    running = np.zeros(ne, np.float32)
    for t in range(T - 1, -1, -1):
        running = R[t] + GAMMA * running * (1.0 - D[t])
        RTG[t] = running
    # flatten (T*ne, ·) and drop transitions that crossed an episode boundary
    flat = lambda x: x.reshape(T * ne, -1) if x.ndim == 3 else x.reshape(T * ne)
    O, A, O2, G = flat(O), flat(A), flat(O2), flat(G)
    R, D, RTG = flat(R), flat(D), flat(RTG)
    keep = D < 0.5
    O, A, O2, G, R, RTG = O[keep], A[keep], O2[keep], G[keep], R[keep], RTG[keep]
    print(f"[collect {task}] obs={obs_dim} act={act_dim} qdim={G.shape[1]} N={O.shape[0]} "
          f"rew(mean={R.mean():.3f} std={R.std():.3f}) rtg(mean={RTG.mean():.2f} std={RTG.std():.2f})",
          flush=True)
    np.savez_compressed(cache_path(task), O=O.astype(np.float32), A=A.astype(np.float32),
                        O2=O2.astype(np.float32), G=G.astype(np.float32),
                        R=R.astype(np.float32), RTG=RTG.astype(np.float32),
                        obs_dim=obs_dim, act_dim=act_dim)
    print(f"[collect {task}] saved {cache_path(task)}", flush=True)


# ---------------------------------------------------------------------------
# 2. Differentiable 2D structural entropy (validated vs selib)
# ---------------------------------------------------------------------------
def build_se_fns():
    import jax, jax.numpy as jnp

    def knn_adj(z):
        zc = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        sim = zc @ zc.T
        B = z.shape[0]
        sim = sim - jnp.eye(B) * 10.0
        thr = jax.lax.top_k(sim, KNN)[0][:, -1:]
        mask = jax.lax.stop_gradient((sim >= thr).astype(z.dtype))
        A = mask * jax.nn.relu(sim)
        A = jnp.maximum(A, A.T)
        return A

    def se2d_soft(A, S):
        deg = A.sum(1)
        two_m = deg.sum() + 1e-12
        Vol = S.T @ deg
        intra = jnp.diag(S.T @ A @ S)
        cut = jnp.clip(Vol - intra, 0.0, None)
        logVol = jnp.log2(Vol + 1e-12)
        leaf = -jnp.sum((deg / two_m) * (jnp.log2(deg + 1e-12) - S @ logVol))
        cutt = -jnp.sum((cut / two_m) * jnp.log2((Vol + 1e-12) / two_m))
        return leaf + cutt

    return knn_adj, se2d_soft


def validate_se():
    import jax, jax.numpy as jnp
    import networkx as nx
    from selib.metrics import structural_entropy_2d
    _, se2d_soft = build_se_fns()
    rng = np.random.default_rng(0)
    n = 60
    G = nx.gnm_random_graph(n, 240, seed=1)
    labels = rng.integers(0, 5, n)
    A = nx.to_numpy_array(G, nodelist=list(range(n)))
    S = np.eye(5)[labels]
    mine = float(se2d_soft(jnp.asarray(A), jnp.asarray(S.astype(np.float32))))
    ref = float(structural_entropy_2d(G, list(labels)))
    err = abs(mine - ref)
    print(f"[validate_se] diff-SE={mine:.6f} selib={ref:.6f} abs_err={err:.2e}", flush=True)
    return {"diff_se": mine, "selib_se": ref, "abs_err": err, "match": err < 1e-4}


# ---------------------------------------------------------------------------
# 3. numpy probe + health helpers
# ---------------------------------------------------------------------------
def ridge_r2(Ztr, Ytr, Zte, Yte, alpha=1.0):
    n = Ztr.shape[0]
    Zc = np.concatenate([Ztr, np.ones((n, 1))], 1)
    Zce = np.concatenate([Zte, np.ones((Zte.shape[0], 1))], 1)
    A = Zc.T @ Zc
    reg = alpha * np.eye(A.shape[0]); reg[-1, -1] = 0.0
    W = np.linalg.solve(A + reg, Zc.T @ Ytr)
    pred = Zce @ W
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(np.mean(r2)), [float(x) for x in r2]


def latent_health(Z, simnorm):
    Z = np.asarray(Z, np.float64)
    n, D = Z.shape
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None); s = ev.sum()
    eff_rank = float((s * s) / (np.square(ev).sum() + 1e-12)) if s > 0 else 0.0
    out = {"z_eff_rank": eff_rank, "z_std_mean": float(Z.std(0).mean())}
    if simnorm:
        grp = Z.reshape(n, V, D // V); codes = grp.argmax(-1)
        ent = []
        for g in range(V):
            cnt = np.bincount(codes[:, g], minlength=D // V).astype(np.float64)
            p = cnt / cnt.sum(); p = p[p > 0]; ent.append(float(-(p * np.log(p)).sum()))
        out["code_entropy_frac"] = float(np.mean(ent) / np.log(D // V))
    return out


# ---------------------------------------------------------------------------
# 4. Train one encoder under one condition, then probe
# ---------------------------------------------------------------------------
def train(task, cond, seed, simnorm, lam_se=LAM_SE):
    import jax, jax.numpy as jnp, flax.linen as nn, optax
    from helios.algorithms.tdmpc2 import simnorm as simnorm_fn, NormMLP
    fname_cond = cond if cond != "se_fixed" else f"sefix{lam_se}"

    d = np.load(cache_path(task))
    O, A_, O2, G, RTG = d["O"], d["A"], d["O2"], d["G"], d["RTG"]
    obs_dim, act_dim = int(d["obs_dim"]), int(d["act_dim"])
    N = O.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N); ntr = int(0.85 * N)
    tr, te = perm[:ntr], perm[ntr:]

    def head(x):
        h = NormMLP(HID, LD)(x)
        return simnorm_fn(h, V) if simnorm else (h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-6))

    class Encoder(nn.Module):
        @nn.compact
        def __call__(self, o):
            return head(o)

    class Pred(nn.Module):
        @nn.compact
        def __call__(self, z, a):
            return head(jnp.concatenate([z, a], -1))

    enc, pred = Encoder(), Pred()
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)
    p_enc = enc.init(k1, jnp.zeros((1, obs_dim)))
    p_pred = pred.init(k2, jnp.zeros((1, LD)), jnp.zeros((1, act_dim)))
    C0 = jax.random.normal(k3, (KC, LD)) * 0.1
    params = {"enc": p_enc, "pred": p_pred, "C": C0}
    target = {"enc": jax.tree.map(lambda x: x, p_enc)}
    tx = optax.chain(optax.clip_by_global_norm(20.0), optax.adam(LR))
    opt = tx.init(params)

    knn_adj, se2d_soft = build_se_fns()

    def vicreg(z):
        B, D = z.shape
        std = jnp.sqrt(z.var(0) + 1e-4)
        v = jnp.mean(jax.nn.relu(VIC_GAMMA - std))
        zc = z - z.mean(0, keepdims=True); cov = (zc.T @ zc) / (B - 1)
        off = cov - jnp.diag(jnp.diag(cov)); c = jnp.sum(off ** 2) / D
        return VICREG_W * v + VICCOV_W * c

    def uniformity(z):
        zc = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        sq = jnp.clip(2.0 - 2.0 * (zc @ zc.T), 0.0, None)   # squared dist on unit sphere
        B = z.shape[0]
        off = sq + jnp.eye(B) * 1e9
        return UNIF_W * jnp.log(jnp.mean(jnp.exp(-2.0 * off)) + 1e-12)

    def se_raw(params, z):
        A = knn_adj(z)
        S = jax.nn.softmax((z @ params["C"].T) / TAU_S, axis=1)
        return se2d_soft(A, S)

    # ---- grad-norm match SE to uniformity at init (like the DMC SE-arm) ----
    lam_used = lam_se
    if cond == "se_fixed":
        cond = "se"; lam_used = lam_se          # SE at a FIXED (unmatched) weight
    elif cond == "se":
        o0 = jnp.asarray(O[rng.integers(0, N, BATCH)])
        gnorm = lambda f: optax.global_norm(jax.grad(f)(params["enc"]))
        gn_unif = float(gnorm(lambda pe: uniformity(enc.apply(pe, o0))))
        gn_se = float(gnorm(lambda pe: se_raw(params, enc.apply(pe, o0))))
        lam_used = float(gn_unif / (gn_se + 1e-8))
        print(f"[gradmatch] gn_unif={gn_unif:.4f} gn_se_raw={gn_se:.4f} -> lam_se={lam_used:.4f}", flush=True)

    def struct_loss(params, z_t, z_tk, zhat):
        if cond == "none":
            return 0.0 * jnp.sum(z_t[:1])
        if cond == "vicreg":
            return vicreg(z_t) + vicreg(z_tk) + vicreg(zhat)
        if cond == "uniformity":
            return uniformity(z_t) + uniformity(z_tk)
        if cond == "se":
            return lam_used * se_raw(params, z_t)
        raise ValueError(cond)

    def loss_fn(params, target, o, a, o2):
        z_t = enc.apply(params["enc"], o)
        zhat = pred.apply(params["pred"], z_t, a)
        z_tgt = jax.lax.stop_gradient(enc.apply(target["enc"], o2))
        l_pred = jnp.mean(jnp.sum((zhat - z_tgt) ** 2, -1))
        z_tk = enc.apply(params["enc"], o2)
        l_struct = struct_loss(params, z_t, z_tk, zhat)
        return l_pred + l_struct, (l_pred, l_struct)

    @jax.jit
    def step(params, target, opt, o, a, o2):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, target, o, a, o2)
        upd, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, upd)
        target = {"enc": jax.tree.map(lambda t, o_: EMA * t + (1 - EMA) * o_,
                                      target["enc"], params["enc"])}
        return params, target, opt, loss, aux

    t0 = time.time()
    Otr, Atr, O2tr = O[tr], A_[tr], O2[tr]
    log = []
    for it in range(STEPS):
        idx = rng.integers(0, Otr.shape[0], BATCH)
        o = jnp.asarray(Otr[idx]); a = jnp.asarray(Atr[idx]); o2 = jnp.asarray(O2tr[idx])
        params, target, opt, loss, aux = step(params, target, opt, o, a, o2)
        if it % 2000 == 0 or it == STEPS - 1:
            lp, ls = float(aux[0]), float(aux[1])
            log.append({"it": it, "loss": float(loss), "l_pred": lp, "l_struct": ls})
            print(f"[{task}/{cond}/sn{simnorm} s{seed}] it={it} loss={float(loss):.4f} "
                  f"l_pred={lp:.4f} l_struct={ls:.4f} {time.time()-t0:.0f}s", flush=True)

    # ---- freeze + probe ----
    enc_j = jax.jit(lambda p, o: enc.apply(p["enc"], o))
    def encode_all(obs):
        out = []
        for i in range(0, obs.shape[0], 4096):
            out.append(np.asarray(enc_j(params, jnp.asarray(obs[i:i + 4096]))))
        return np.concatenate(out)
    Ztr, Zte = encode_all(O[tr]), encode_all(O[te])
    geom_r2, geom_per = ridge_r2(Ztr, G[tr], Zte, G[te])
    val_r2, val_per = ridge_r2(Ztr, RTG[tr].reshape(-1, 1), Zte, RTG[te].reshape(-1, 1))
    health = latent_health(Zte[:4000], simnorm)

    res = {"task": task, "cond": fname_cond, "simnorm": int(simnorm), "seed": seed, "steps": STEPS,
           "latent_dim": LD, "n_train": int(len(tr)), "n_test": int(len(te)),
           "lam_se_used": lam_used, "wall_s": round(time.time() - t0, 1),
           "geom_r2": geom_r2, "geom_r2_per_dim": geom_per,
           "value_r2": val_r2,
           "eff_rank": health["z_eff_rank"], "z_std_mean": health["z_std_mean"],
           "code_entropy_frac": health.get("code_entropy_frac"),
           "train_log": log}
    fn = ROOT / f"run_{task}_{fname_cond}_sn{simnorm}_seed{seed}.json"
    fn.write_text(json.dumps(res, indent=2))
    print(f"[{task}/{cond}/sn{simnorm} s{seed}] DONE geom_r2={geom_r2:.4f} "
          f"value_r2={val_r2:.4f} eff_rank={health['z_eff_rank']:.2f} -> {fn}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "validate_se", "train"])
    ap.add_argument("--task", default="WalkerWalk")
    ap.add_argument("--cond", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simnorm", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lam_se", type=float, default=LAM_SE)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.22"   # cap VRAM, leave mahjong headroom
    if args.mode == "collect":
        collect(args.task)
    elif args.mode == "validate_se":
        print(validate_se())
    elif args.mode == "train":
        v = validate_se()
        if not v["match"]:
            print("[WARN] SE faithfulness check failed", flush=True)
        train(args.task, args.cond, args.seed, bool(args.simnorm), lam_se=args.lam_se)
