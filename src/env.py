"""Многоагентная среда «разметка потока треков каталога по типу фронт-вокала» (векторизованная, PyTorch).

Агенты (кооперативные, общая награда):
  0 — эксперт по тексту метаданных, 1 — эксперт по временному ряду звука, 2 — эксперт по спектрограмме:
      действие a in {0..3} — голос за класс (male/female/group/instrumental), 4 — воздержаться;
  3 — координатор качества: 0 — поставить метку автоматически (по голосам экспертов), 1 — отправить модератору.
Наблюдения строятся только из информации, доступной в момент решения (вероятности моделей по текущему треку,
остаток бюджета, доля прошедшего эпизода, «репутация» экспертов по ранее проверенным модератором трекам).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

N_AGENTS, N_EXPERTS, N_ACT, OBS_DIM = 4, 3, 5, 13
MODS = ["text", "ts", "img"]


@dataclass
class EnvConfig:
    T: int = 50                 # треков в эпизоде
    budget: int = 15            # лимит ручных проверок за эпизод
    r_correct: float = 1.0      # верная автометка
    r_wrong: float = -3.0       # ошибка в каталоге (портит поиск и рекомендации)
    r_review: float = -0.4      # стоимость ручной проверки (аналог транзакционных издержек)
    r_unlabeled: float = -0.5   # все эксперты воздержались, а координатор не отправил на проверку
    r_violation: float = -1.0   # попытка проверки при исчерпанном бюджете (нарушение ограничения)
    use_coordinator: bool = True
    allow_abstain: bool = True
    use_trust: bool = True
    drop_modality: str = ""     # стресс-тест: модальность недоступна (вероятности = равномерные)


def load_split(path, split, years=None):
    st = pd.read_parquet(path)
    d = st[st.split == split]
    if years is not None:
        d = d[(d.release_year >= years[0]) & (d.release_year <= years[1])]
    P = np.stack([d[[f"p_{m}_{k}" for k in range(4)]].values for m in MODS], 1)  # (N,3,4)
    return torch.tensor(P, dtype=torch.float32), torch.tensor(d.y.values, dtype=torch.long)


class LabelingEnv:
    def __init__(self, P, y, n_envs, cfg=EnvConfig(), seed=0):
        self.P, self.y, self.n, self.cfg = P.clone(), y, n_envs, cfg
        if cfg.drop_modality:
            self.P[:, MODS.index(cfg.drop_modality)] = 0.25
        self.g = torch.Generator().manual_seed(seed)

    # ---------------- служебное
    def _cur(self):
        idx = self.order[torch.arange(self.n), self.t.clamp(max=self.cfg.T - 1)]
        return self.P[idx], self.y[idx]

    def reset(self):
        self.order = torch.randint(0, len(self.y), (self.n, self.cfg.T), generator=self.g)
        self.t = torch.zeros(self.n, dtype=torch.long)
        self.budget = torch.full((self.n,), float(self.cfg.budget))
        self.trust_hit = torch.ones(self.n, N_EXPERTS)  # априорно 1/2 (сглаживание Лапласа)
        self.trust_n = torch.full((self.n, N_EXPERTS), 2.0)
        return self.obs()

    def obs(self):
        p, _ = self._cur()                                          # (n,3,4)
        ent = -(p * p.clamp_min(1e-8).log()).sum(-1) / np.log(4)    # (n,3)
        bfrac = (self.budget / self.cfg.budget)[:, None]
        tfrac = (self.t.float() / self.cfg.T)[:, None]
        trust = self.trust_hit / self.trust_n if self.cfg.use_trust else torch.full((self.n, 3), 0.5)
        o = torch.zeros(self.n, N_AGENTS, OBS_DIM)
        for i in range(N_EXPERTS):
            o[:, i, :4] = p[:, i]
            o[:, i, 4] = ent[:, i]
            o[:, i, 5:6] = bfrac
            o[:, i, 6:7] = tfrac
            o[:, i, 7] = trust[:, i]
            o[:, i, 9 + i] = 1.0
        am = p.argmax(-1)
        agree3 = ((am[:, 0] == am[:, 1]) & (am[:, 1] == am[:, 2])).float()
        agree2 = ((am[:, 0] == am[:, 1]) | (am[:, 1] == am[:, 2]) | (am[:, 0] == am[:, 2])).float()
        o[:, 3, :3] = p.max(-1).values
        o[:, 3, 3] = agree3
        o[:, 3, 4] = agree2
        o[:, 3, 5:6] = bfrac
        o[:, 3, 6:7] = tfrac
        o[:, 3, 7] = ent.mean(-1)
        o[:, 3, 8] = 1.0
        return o

    def state(self):
        """Глобальное состояние для централизованных критиков (CTDE)."""
        p, _ = self._cur()
        trust = self.trust_hit / self.trust_n
        return torch.cat([p.flatten(1), (self.budget / self.cfg.budget)[:, None],
                          (self.t.float() / self.cfg.T)[:, None], trust], 1)  # (n,17)

    @staticmethod
    def state_dim():
        return 17

    def action_mask(self):
        m = torch.ones(self.n, N_AGENTS, N_ACT, dtype=torch.bool)
        if not self.cfg.allow_abstain:
            m[:, :N_EXPERTS, 4] = False
        m[:, 3, 2:] = False  # координатор: только 0/1
        if not self.cfg.use_coordinator:
            m[:, 3, 1] = False
        return m

    def step(self, a):
        """a: (n,4) long. Возвращает obs, reward (n,), done (n,), info."""
        cfg = self.cfg
        p, y = self._cur()
        votes = a[:, :N_EXPERTS]
        conf = p.gather(-1, votes.clamp(max=3)[..., None])[..., 0] * (votes < 4)       # уверенность голосовавших
        score = torch.zeros(self.n, 4)
        score.scatter_add_(1, votes.clamp(max=3), (votes < 4).float() + 1e-3 * conf)    # большинство, тай-брейк уверенностью
        any_vote = (votes < 4).any(1)
        auto_label = score.argmax(1)
        review = a[:, 3] == 1
        if not cfg.use_coordinator:  # без координатора: проверка только если все воздержались
            review = ~any_vote
        can = self.budget > 0
        do_review = review & can
        violation = review & ~can
        auto = ~do_review
        correct = auto & any_vote & (auto_label == y)
        wrong = auto & any_vote & (auto_label != y)
        unlabeled = auto & ~any_vote
        r = (correct * cfg.r_correct + wrong * cfg.r_wrong + do_review * cfg.r_review
             + unlabeled * cfg.r_unlabeled + violation * cfg.r_violation).float()
        # обратная связь модератора обновляет «репутацию» экспертов (доступна только по проверенным трекам)
        hit = (votes == y[:, None]).float()
        self.trust_hit += do_review[:, None] * hit
        self.trust_n += do_review[:, None].float()
        self.budget -= do_review.float()
        info = {"correct": correct, "wrong": wrong, "review": do_review, "unlabeled": unlabeled,
                "violation": violation, "abstain": (votes == 4).float().mean(1),
                "final_correct": correct | do_review}
        self.t += 1
        done = self.t >= cfg.T
        return self.obs(), r, done, info
