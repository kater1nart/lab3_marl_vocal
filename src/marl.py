"""8 MARL-алгоритмов + одноагентный PPO + эвристический baseline. Общий бюджет взаимодействий и протокол.

On-policy: IPPO, MAPPO, MAA2C, COMA, PPO (single-agent).  Off-policy: IQL, VDN, QMIX, MADDPG.
Параметры сетей разделяются между агентами (во вход добавлен one-hot идентификатор агента).
Лучший чекпойнт выбирается по средней награде на val (тест не используется при обучении).
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import N_ACT, N_AGENTS, N_EXPERTS, OBS_DIM, LabelingEnv

S_DIM = LabelingEnv.state_dim()
HP = {"n_envs": 64, "iters": 150, "gamma": 0.99, "lam": 0.95, "lr": 5e-4, "hidden": 64,
      "ppo_epochs": 4, "ppo_clip": 0.2, "ent_coef": 0.01, "batch": 256, "buffer": 100_000,
      "eps_start": 1.0, "eps_end": 0.05, "target_every": 200, "eval_every": 10, "eval_episodes": 128}


def mlp(i, o, h=HP["hidden"]):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def masked(logits, mask):
    return logits.masked_fill(~mask, -1e9)


# ============================================================ оценка политики
@torch.no_grad()
def evaluate(policy, P, y, cfg, n_episodes=HP["eval_episodes"], seed=12345):
    env = LabelingEnv(P, y, n_episodes, cfg, seed=seed)
    o = env.reset()
    R = torch.zeros(n_episodes)
    stats = {k: torch.zeros(n_episodes) for k in ["correct", "wrong", "review", "unlabeled", "violation", "abstain", "final_correct"]}
    step_r = []
    while True:
        a = policy(o, env.state(), env.action_mask())
        o, r, done, info = env.step(a)
        R += r
        step_r.append(r)
        for k in stats:
            stats[k] += info[k].float()
        if done.all():
            break
    T = cfg.T
    out = {"R": R.numpy(), "step_r": torch.stack(step_r, 1).numpy()}
    out.update({k: (v / T).numpy() for k, v in stats.items()})
    return out


# ============================================================ on-policy семейство
class OnPolicy:
    """kind: ippo | mappo | maa2c | coma | ppo (одноагентный централизованный)."""

    def __init__(self, kind, seed):
        torch.manual_seed(seed)
        self.kind = kind
        if kind == "ppo":
            self.actor = mlp(S_DIM, N_AGENTS * N_ACT)
            self.critic = mlp(S_DIM, 1)
        else:
            self.actor = mlp(OBS_DIM, N_ACT)
            if kind == "ippo":
                self.critic = mlp(OBS_DIM, 1)
            elif kind in ("mappo", "maa2c"):
                self.critic = mlp(S_DIM + N_AGENTS, 1)
            elif kind == "coma":  # Q(s, действия остальных, id агента) -> Q по своим действиям
                self.critic = mlp(S_DIM + N_AGENTS * N_ACT + N_AGENTS, N_ACT)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=HP["lr"])

    def logits(self, o, s, m):
        if self.kind == "ppo":
            return masked(self.actor(s).view(-1, N_AGENTS, N_ACT), m)
        return masked(self.actor(o), m)

    def act(self, o, s, m, greedy=False):
        lg = self.logits(o, s, m)
        return lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()

    def value(self, o, s, a=None):
        n = s.shape[0]
        eye = torch.eye(N_AGENTS).expand(n, -1, -1)
        if self.kind == "ppo":
            return self.critic(s).expand(n, N_AGENTS) if s.dim() == 2 else self.critic(s)[..., 0]
        if self.kind == "ippo":
            return self.critic(o)[..., 0]
        if self.kind in ("mappo", "maa2c"):
            return self.critic(torch.cat([s[:, None].expand(-1, N_AGENTS, -1), eye], -1))[..., 0]

    def coma_q(self, s, a):
        n = s.shape[0]
        oh = F.one_hot(a, N_ACT).float()                               # (n,4,5)
        ins = []
        for i in range(N_AGENTS):
            others = oh.clone()
            others[:, i] = 0
            ins.append(torch.cat([s, others.flatten(1), F.one_hot(torch.tensor(i), N_AGENTS).float().expand(n, -1)], -1))
        return self.critic(torch.stack(ins, 1))                        # (n,4,5)

    def update(self, B):
        T, n = B["r"].shape
        flat = lambda x: x.reshape(T * n, *x.shape[2:])
        o, s, m, a, oldlp = map(flat, (B["o"], B["s"], B["m"], B["a"], B["lp"]))
        if self.kind == "coma":
            # критик: Q(s,a) -> дисконтированный return-to-go (Монте-Карло)
            G = torch.zeros(T, n)
            run = torch.zeros(n)
            for t in reversed(range(T)):
                run = B["r"][t] + HP["gamma"] * run * (1 - B["d"][t])
                G[t] = run
            G = flat(G)
            for _ in range(HP["ppo_epochs"]):
                q = self.coma_q(s, a)
                qa = q.gather(-1, a[..., None])[..., 0]
                lc = F.mse_loss(qa, G[:, None].expand_as(qa))
                pi = F.softmax(self.logits(o, s, m), -1)
                adv = (qa - (pi * q).sum(-1)).detach()                 # контрфактический baseline
                lp = torch.log(pi.gather(-1, a[..., None])[..., 0] + 1e-8)
                ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()
                loss = -(lp * adv).mean() + lc - HP["ent_coef"] * ent
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.opt.step()
            return
        with torch.no_grad():
            V = self.value(B["o"].reshape(T * n, N_AGENTS, OBS_DIM), B["s"].reshape(T * n, S_DIM)).view(T, n, N_AGENTS)
            adv = torch.zeros(T, n, N_AGENTS)
            gae = torch.zeros(n, N_AGENTS)
            for t in reversed(range(T)):
                nv = V[t + 1] if t + 1 < T else torch.zeros(n, N_AGENTS)
                nd = 1 - B["d"][t][:, None]
                delta = B["r"][t][:, None] + HP["gamma"] * nv * nd - V[t]
                gae = delta + HP["gamma"] * HP["lam"] * nd * gae
                adv[t] = gae
            ret = flat(adv + V)
            adv = flat(adv)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        epochs = 1 if self.kind == "maa2c" else HP["ppo_epochs"]
        for _ in range(epochs):
            lg = self.logits(o, s, m)
            dist = torch.distributions.Categorical(logits=lg)
            lp = dist.log_prob(a)
            if self.kind == "maa2c":
                pl = -(lp * adv).mean()
            else:
                ratio = (lp - oldlp).exp()
                pl = -torch.min(ratio * adv, ratio.clamp(1 - HP["ppo_clip"], 1 + HP["ppo_clip"]) * adv).mean()
            vl = F.mse_loss(self.value(o, s), ret)
            loss = pl + 0.5 * vl - HP["ent_coef"] * dist.entropy().mean()
            self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.opt.step()

    def train(self, Ptr, ytr, Pva, yva, cfg, seed, log):
        env = LabelingEnv(Ptr, ytr, HP["n_envs"], cfg, seed=seed)
        best, best_state = -1e9, None
        for it in range(HP["iters"]):
            o = env.reset()
            B = {k: [] for k in ["o", "s", "m", "a", "lp", "r", "d"]}
            while True:
                s, m = env.state(), env.action_mask()
                with torch.no_grad():
                    lg = self.logits(o, s, m)
                    dist = torch.distributions.Categorical(logits=lg)
                    a = dist.sample()
                o2, r, d, _ = env.step(a)
                for k, v in zip(B, (o, s, m, a, dist.log_prob(a), r, d.float())):
                    B[k].append(v)
                o = o2
                if d.all():
                    break
            B = {k: torch.stack(v) for k, v in B.items()}
            self.update(B)
            log.append({"iter": it, "steps": (it + 1) * HP["n_envs"] * cfg.T, "train_R": B["r"].sum(0).mean().item()})
            if (it + 1) % HP["eval_every"] == 0:
                vr = evaluate(self.policy(), Pva, yva, cfg)["R"].mean()
                log[-1]["val_R"] = float(vr)
                if vr > best:
                    best, best_state = vr, copy.deepcopy((self.actor.state_dict(), self.critic.state_dict()))
        self.actor.load_state_dict(best_state[0]); self.critic.load_state_dict(best_state[1])
        return best

    def policy(self):
        return lambda o, s, m: self.act(o, s, m, greedy=True)


# ============================================================ off-policy: IQL / VDN / QMIX
class Mixer(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.h = h
        self.w1, self.b1 = nn.Linear(S_DIM, N_AGENTS * h), nn.Linear(S_DIM, h)
        self.w2, self.b2 = nn.Linear(S_DIM, h), nn.Sequential(nn.Linear(S_DIM, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, q, s):  # q (B,4) -> Q_tot (B,)
        w1 = self.w1(s).abs().view(-1, N_AGENTS, self.h)
        hdn = F.elu(torch.bmm(q[:, None], w1)[:, 0] + self.b1(s))
        return (hdn * self.w2(s).abs()).sum(-1) + self.b2(s)[:, 0]


class Buffer:
    def __init__(self, cap):
        self.cap, self.i, self.full, self.d = cap, 0, False, {}

    def add(self, **kw):
        n = next(iter(kw.values())).shape[0]
        for k, v in kw.items():
            if k not in self.d:
                self.d[k] = torch.zeros((self.cap, *v.shape[1:]), dtype=v.dtype)
            idx = (torch.arange(n) + self.i) % self.cap
            self.d[k][idx] = v
        self.full |= self.i + n >= self.cap
        self.i = (self.i + n) % self.cap

    def sample(self, b):
        hi = self.cap if self.full else self.i
        ix = torch.randint(0, hi, (b,))
        return {k: v[ix] for k, v in self.d.items()}


class ValueBased:
    def __init__(self, kind, seed):
        torch.manual_seed(seed)
        self.kind = kind
        self.q = mlp(OBS_DIM, N_ACT)
        self.mix = Mixer() if kind == "qmix" else None
        params = list(self.q.parameters()) + (list(self.mix.parameters()) if self.mix else [])
        self.opt = torch.optim.Adam(params, lr=HP["lr"])
        self.tq, self.tmix = copy.deepcopy(self.q), copy.deepcopy(self.mix)

    def act(self, o, s, m, eps=0.0):
        a = masked(self.q(o), m).argmax(-1)
        if eps > 0:
            rnd = torch.distributions.Categorical(logits=masked(torch.zeros_like(m, dtype=torch.float), m)).sample()
            a = torch.where(torch.rand(a.shape) < eps, rnd, a)
        return a

    def update(self, b):
        qa = self.q(b["o"]).gather(-1, b["a"][..., None])[..., 0]           # (B,4)
        with torch.no_grad():
            na = masked(self.q(b["o2"]), b["m2"]).argmax(-1, keepdim=True)  # double-Q
            nq = self.tq(b["o2"]).gather(-1, na)[..., 0]
        nd = 1 - b["d"]
        if self.kind == "iql":
            tgt = b["r"][:, None] + HP["gamma"] * nd[:, None] * nq
            loss = F.mse_loss(qa, tgt)
        elif self.kind == "vdn":
            loss = F.mse_loss(qa.sum(1), b["r"] + HP["gamma"] * nd * nq.sum(1))
        else:
            with torch.no_grad():
                tgt = b["r"] + HP["gamma"] * nd * self.tmix(nq, b["s2"])
            loss = F.mse_loss(self.mix(qa, b["s"]), tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.opt.param_groups[0]["params"], 10.0); self.opt.step()

    def train(self, Ptr, ytr, Pva, yva, cfg, seed, log):
        env = LabelingEnv(Ptr, ytr, HP["n_envs"], cfg, seed=seed)
        buf = Buffer(HP["buffer"])
        total_steps = HP["iters"] * cfg.T
        best, best_state, k = -1e9, None, 0
        for it in range(HP["iters"]):
            o = env.reset()
            ep_r = torch.zeros(HP["n_envs"])
            while True:
                frac = min(1.0, k / (0.5 * total_steps))
                eps = HP["eps_start"] + frac * (HP["eps_end"] - HP["eps_start"])
                s, m = env.state(), env.action_mask()
                with torch.no_grad():
                    a = self.act(o, s, m, eps)
                o2, r, d, _ = env.step(a)
                buf.add(o=o, s=s, a=a, r=r, o2=o2, s2=env.state(), m2=env.action_mask(), d=d.float())
                ep_r += r
                o, k = o2, k + 1
                if buf.i > HP["batch"] or buf.full:
                    self.update(buf.sample(HP["batch"]))
                if k % HP["target_every"] == 0:
                    self.tq.load_state_dict(self.q.state_dict())
                    if self.mix:
                        self.tmix.load_state_dict(self.mix.state_dict())
                if d.all():
                    break
            log.append({"iter": it, "steps": (it + 1) * HP["n_envs"] * cfg.T, "train_R": ep_r.mean().item(), "eps": eps})
            if (it + 1) % HP["eval_every"] == 0:
                vr = evaluate(self.policy(), Pva, yva, cfg)["R"].mean()
                log[-1]["val_R"] = float(vr)
                if vr > best:
                    best, best_state = vr, copy.deepcopy(self.q.state_dict())
        self.q.load_state_dict(best_state)
        return best

    def policy(self):
        return lambda o, s, m: self.act(o, s, m)


# ============================================================ off-policy: MADDPG (дискретные действия, Gumbel-softmax)
class MADDPG:
    def __init__(self, kind, seed):
        torch.manual_seed(seed)
        self.actor = mlp(OBS_DIM, N_ACT)
        self.critic = mlp(S_DIM + N_AGENTS * N_ACT, 1)   # общий централизованный критик (кооперативная награда)
        self.ta, self.tc = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr=HP["lr"])
        self.oc = torch.optim.Adam(self.critic.parameters(), lr=HP["lr"])

    def act(self, o, s, m, explore=False):
        lg = masked(self.actor(o), m)
        if explore:
            return F.gumbel_softmax(lg, tau=1.0, hard=True).argmax(-1)
        return lg.argmax(-1)

    def Q(self, net, s, a_oh):
        return net(torch.cat([s, a_oh.flatten(1)], -1))[:, 0]

    def update(self, b):
        with torch.no_grad():
            na = F.one_hot(masked(self.ta(b["o2"]), b["m2"]).argmax(-1), N_ACT).float()
            tgt = b["r"] + HP["gamma"] * (1 - b["d"]) * self.Q(self.tc, b["s2"], na)
        a_oh = F.one_hot(b["a"], N_ACT).float()
        lc = F.mse_loss(self.Q(self.critic, b["s"], a_oh), tgt)
        self.oc.zero_grad(); lc.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0); self.oc.step()
        g = F.gumbel_softmax(masked(self.actor(b["o"]), b["m"]), tau=1.0, hard=True)
        i = torch.randint(0, N_AGENTS, (1,)).item()           # обновляем одного агента за шаг, остальные — из буфера
        joint = a_oh.clone()
        joint[:, i] = g[:, i]
        la = -self.Q(self.critic, b["s"], joint).mean()
        self.oa.zero_grad(); la.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0); self.oa.step()
        with torch.no_grad():
            for t, s_ in ((self.ta, self.actor), (self.tc, self.critic)):
                for pt, ps in zip(t.parameters(), s_.parameters()):
                    pt.mul_(0.99).add_(0.01 * ps)

    def train(self, Ptr, ytr, Pva, yva, cfg, seed, log):
        env = LabelingEnv(Ptr, ytr, HP["n_envs"], cfg, seed=seed)
        buf = Buffer(HP["buffer"])
        best, best_state = -1e9, None
        for it in range(HP["iters"]):
            o = env.reset()
            ep_r = torch.zeros(HP["n_envs"])
            while True:
                s, m = env.state(), env.action_mask()
                with torch.no_grad():
                    a = self.act(o, s, m, explore=True)
                o2, r, d, _ = env.step(a)
                buf.add(o=o, s=s, m=m, a=a, r=r, o2=o2, s2=env.state(), m2=env.action_mask(), d=d.float())
                ep_r += r
                o = o2
                if buf.i > HP["batch"] or buf.full:
                    self.update(buf.sample(HP["batch"]))
                if d.all():
                    break
            log.append({"iter": it, "steps": (it + 1) * HP["n_envs"] * cfg.T, "train_R": ep_r.mean().item()})
            if (it + 1) % HP["eval_every"] == 0:
                vr = evaluate(self.policy(), Pva, yva, cfg)["R"].mean()
                log[-1]["val_R"] = float(vr)
                if vr > best:
                    best, best_state = vr, copy.deepcopy(self.actor.state_dict())
        self.actor.load_state_dict(best_state)
        return best

    def policy(self):
        return lambda o, s, m: self.act(o, s, m)


# ============================================================ эвристический baseline («равновзвешенный портфель»)
class EqualWeight:
    """Равные веса трёх модальностей: средняя вероятность; автометка, если max >= tau, иначе — проверка.
    Все эксперты голосуют за класс усреднённого распределения. tau подбирается по val (сетка)."""

    def __init__(self, *_):
        self.tau = 0.5

    def policy(self):
        def f(o, s, m):
            p = s[:, :12].view(-1, 3, 4).mean(1)
            lab = p.argmax(-1)
            a = torch.zeros(s.shape[0], N_AGENTS, dtype=torch.long)
            a[:, :N_EXPERTS] = lab[:, None]
            a[:, 3] = (p.max(-1).values < self.tau).long()
            return a
        return f

    def train(self, Ptr, ytr, Pva, yva, cfg, seed, log):
        best = -1e9
        for tau in np.arange(0.3, 0.95, 0.025):
            self.tau = float(tau)
            vr = evaluate(self.policy(), Pva, yva, cfg)["R"].mean()
            log.append({"tau": float(tau), "val_R": float(vr)})
            if vr > best:
                best, bt = vr, float(tau)
        self.tau = bt
        return best


ALGOS = {"IQL": lambda s: ValueBased("iql", s), "VDN": lambda s: ValueBased("vdn", s),
         "QMIX": lambda s: ValueBased("qmix", s), "IPPO": lambda s: OnPolicy("ippo", s),
         "MAPPO": lambda s: OnPolicy("mappo", s), "MAA2C": lambda s: OnPolicy("maa2c", s),
         "COMA": lambda s: OnPolicy("coma", s), "MADDPG": lambda s: MADDPG("maddpg", s),
         "PPO (1 агент)": lambda s: OnPolicy("ppo", s), "Равные веса": lambda s: EqualWeight(s)}


def n_params(alg):
    mods = [getattr(alg, k) for k in ("actor", "critic", "q", "mix") if getattr(alg, k, None) is not None]
    return sum(p.numel() for m in mods for p in m.parameters())
