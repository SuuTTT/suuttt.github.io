"""
MiniGrid hard-exploration: PPO vs PPO+RND (novelty) head-to-head.
Single-file, cleanrl-style. CPU-only. Sparse extrinsic reward (NO shaping).
Metrics from disk: REAL success (terminated=goal), coverage (unique cells),
discovery step (first success). n>=2 seeds via --seed. Matched env-step budget.
"""
import argparse, json, os, time, random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--arm", type=str, default="ppo", choices=["ppo", "rnd"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--int-coef", type=float, default=1.0)   # RND intrinsic weight
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--torch-threads", type=int, default=2)
    return p.parse_args()


class CoverageWrapper(gym.Wrapper):
    """Tracks unique (x,y) cells visited per episode and cumulatively."""
    def __init__(self, env):
        super().__init__(env)
        self.cum_cells = set()
        self.ep_cells = set()
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.ep_cells = set()
        pos = tuple(int(x) for x in self.env.unwrapped.agent_pos)
        self.ep_cells.add(pos); self.cum_cells.add(pos)
        return obs, info
    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        pos = tuple(int(x) for x in self.env.unwrapped.agent_pos)
        self.ep_cells.add(pos); self.cum_cells.add(pos)
        info["ep_coverage"] = len(self.ep_cells)
        info["cum_coverage"] = len(self.cum_cells)
        return obs, r, term, trunc, info


def make_env(env_id, seed, idx):
    def thunk():
        e = gym.make(env_id)
        e = FullyObsWrapper(e)
        e = ImgObsWrapper(e)
        e = gym.wrappers.FlattenObservation(e)
        e = CoverageWrapper(e)
        e.reset(seed=seed + idx)
        e.action_space.seed(seed + idx)
        return e
    return thunk


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(256, n_act), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
    def get_value(self, x):
        return self.critic(self.net(x))
    def get_action_and_value(self, x, action=None):
        h = self.net(x)
        logits = self.actor(h)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(h)


class RND(nn.Module):
    def __init__(self, obs_dim, feat=128):
        super().__init__()
        self.target = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, feat)),
        )
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, feat)),
        )
        for p in self.target.parameters():
            p.requires_grad = False
    def forward(self, x):
        tgt = self.target(x)
        pred = self.predictor(x)
        return ((pred - tgt) ** 2).mean(dim=1)  # per-sample novelty


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = 1e-4
    def update(self, x):
        bmean = np.mean(x, axis=0); bvar = np.var(x, axis=0); bcount = x.shape[0]
        d = bmean - self.mean
        tot = self.count + bcount
        self.mean += d * bcount / tot
        m_a = self.var * self.count; m_b = bvar * bcount
        M2 = m_a + m_b + d ** 2 * self.count * bcount / tot
        self.var = M2 / tot; self.count = tot


def main():
    args = parse()
    os.makedirs(args.outdir, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env, args.seed, i) for i in range(args.num_envs)]
    )
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    n_act = envs.single_action_space.n

    agent = Agent(obs_dim, n_act).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    use_rnd = args.arm == "rnd"
    if use_rnd:
        rnd = RND(obs_dim).to(device)
        rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=args.lr)
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        int_ret_rms = RunningMeanStd(shape=())
        int_returns = np.zeros(args.num_envs)

    N = args.num_steps; NE = args.num_envs
    obs_buf = torch.zeros((N, NE, obs_dim))
    act_buf = torch.zeros((N, NE), dtype=torch.long)
    logp_buf = torch.zeros((N, NE))
    rew_buf = torch.zeros((N, NE))
    int_rew_buf = torch.zeros((N, NE))
    done_buf = torch.zeros((N, NE))
    val_buf = torch.zeros((N, NE))

    batch_size = NE * N
    minibatch = batch_size // args.num_minibatches
    num_updates = args.total_steps // batch_size

    # metrics
    global_step = 0
    first_success_step = None
    n_success = 0
    n_episodes = 0
    ep_returns = []          # extrinsic return per finished episode
    ep_success = []          # 1/0 per finished episode
    ep_cov = []              # per-episode unique cells
    success_step_log = []    # (global_step, rolling_success_rate) checkpoints
    t0 = time.time()

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32)
    next_done = torch.zeros(NE)

    for update in range(1, num_updates + 1):
        for step in range(N):
            global_step += NE
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            val_buf[step] = value.flatten()
            act_buf[step] = action
            logp_buf[step] = logp

            nobs, reward, term, trunc, infos = envs.step(action.numpy())
            done = np.logical_or(term, trunc)
            rew_buf[step] = torch.tensor(reward, dtype=torch.float32)

            if use_rnd:
                norm_obs = np.clip(
                    (nobs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -5, 5
                ).astype("float32")
                with torch.no_grad():
                    inov = rnd(torch.tensor(norm_obs)).numpy()
                int_returns = int_returns * args.gamma + inov
                int_ret_rms.update(int_returns.reshape(-1))
                inov_norm = inov / np.sqrt(int_ret_rms.var + 1e-8)
                int_rew_buf[step] = torch.tensor(inov_norm, dtype=torch.float32)
                obs_rms.update(nobs)

            next_obs = torch.tensor(nobs, dtype=torch.float32)
            next_done = torch.tensor(done.astype("float32"))

            # episode bookkeeping (real success = terminated at goal)
            for i in range(NE):
                if done[i]:
                    n_episodes += 1
                    succ = 1 if term[i] else 0   # term=True => goal reached
                    n_success += succ
                    ep_success.append(succ)
                    ep_returns.append(float(reward[i]))
                    if "ep_coverage" in infos:
                        pass
                    if succ and first_success_step is None:
                        first_success_step = global_step
                    if use_rnd:
                        int_returns[i] = 0.0
            # coverage from final_info / infos vector
            if "ep_coverage" in infos:
                cov_arr = infos["ep_coverage"]
                for i in range(NE):
                    if done[i]:
                        ep_cov.append(int(cov_arr[i]))

        # ---- GAE ----
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
        if use_rnd:
            total_rew = rew_buf + args.int_coef * int_rew_buf
        else:
            total_rew = rew_buf
        advantages = torch.zeros_like(total_rew)
        lastgae = 0
        for t in reversed(range(N)):
            if t == N - 1:
                nextnonterm = 1.0 - next_done
                nextval = next_value
            else:
                nextnonterm = 1.0 - done_buf[t + 1]
                nextval = val_buf[t + 1]
            delta = total_rew[t] + args.gamma * nextval * nextnonterm - val_buf[t]
            advantages[t] = lastgae = (
                delta + args.gamma * args.gae_lambda * nextnonterm * lastgae
            )
        returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        inds = np.arange(batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch):
                mb = inds[start:start + minibatch]
                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )
                ratio = (newlogp - b_logp[mb]).exp()
                mbadv = b_adv[mb]
                mbadv = (mbadv - mbadv.mean()) / (mbadv.std() + 1e-8)
                pg1 = -mbadv * ratio
                pg2 = -mbadv * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                pg_loss = torch.max(pg1, pg2).mean()
                newval = newval.view(-1)
                v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # RND predictor update
        if use_rnd:
            norm_obs = np.clip(
                (b_obs.numpy() - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -5, 5
            ).astype("float32")
            no = torch.tensor(norm_obs)
            for start in range(0, batch_size, minibatch):
                mb = inds[start:start + minibatch]
                loss_rnd = rnd(no[mb]).mean()
                rnd_opt.zero_grad(); loss_rnd.backward(); rnd_opt.step()

        # periodic checkpoint of rolling metrics
        recent = ep_success[-100:] if ep_success else [0]
        roll_sr = float(np.mean(recent))
        success_step_log.append({
            "step": global_step, "rolling_success_rate": roll_sr,
            "cum_success": n_success, "n_episodes": n_episodes,
        })

    cum_cov = envs.envs[0].cum_cells if hasattr(envs, "envs") else None
    # aggregate cumulative coverage across all vector envs
    total_cov = set()
    try:
        for e in envs.envs:
            total_cov |= e.cum_cells
    except Exception:
        pass

    result = {
        "env": args.env, "arm": args.arm, "seed": args.seed,
        "total_steps": global_step,
        "num_envs": NE, "wall_sec": round(time.time() - t0, 1),
        "n_episodes": n_episodes,
        "n_success": n_success,
        "success_rate_overall": (n_success / n_episodes) if n_episodes else 0.0,
        "success_rate_last100": float(np.mean(ep_success[-100:])) if ep_success else 0.0,
        "first_success_step": first_success_step,
        "solved": first_success_step is not None,
        "cum_unique_cells_all_envs": len(total_cov),
        "mean_ep_coverage": float(np.mean(ep_cov)) if ep_cov else 0.0,
        "max_ep_coverage": int(np.max(ep_cov)) if ep_cov else 0,
        "mean_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "success_curve": success_step_log,
    }
    outpath = os.path.join(
        args.outdir, f"{args.env}__{args.arm}__seed{args.seed}.json"
    )
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", outpath, "solved=", result["solved"],
          "sr_last100=", result["success_rate_last100"],
          "cov=", result["cum_unique_cells_all_envs"])
    envs.close()


if __name__ == "__main__":
    main()
