"""
Knowledge tracing (BKT) + Elo-style baseline: next-response AUC and next-item policy comparison.

- **BKT**: belief P(L) per subskill, shared p_T, p_g, p_s, per-skill p_L0.
- **Elo/1PL**: P(correct)=σ(θ−b_item), online θ per session, b from training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArr = NDArray[np.float64]


def _c(x: float) -> float:
    return min(1.0 - 1e-7, max(1e-7, x))


def p_correct_bkt(p_L: float, p_g: float, p_s: float) -> float:
    return p_L * (1.0 - p_s) + (1.0 - p_L) * p_g


def p_L_posterior(p_L: float, y: int, p_g: float, p_s: float) -> float:
    """Posterior P(L) after one outcome, before learning transition."""
    p_c = _c(p_correct_bkt(p_L, p_g, p_s))
    if y == 1:
        return _c(p_L * (1.0 - p_s) / p_c)
    p_wrong = 1.0 - p_c
    p_wrong = _c(p_wrong)
    p_wrong_num = p_L * p_s + (1.0 - p_L) * (1.0 - p_g)
    p_wrong_num = max(p_wrong_num, 1e-9)
    return _c(p_L * p_s / p_wrong_num)


def p_L_with_learning(p_L_post: float, p_t: float) -> float:
    return p_L_post + (1.0 - p_L_post) * p_t


@dataclass
class BKTModel:
    n_skills: int
    p_L0: FloatArr
    p_t: float
    p_g: float
    p_s: float

    @classmethod
    def from_array(cls, n_skills: int, p: np.ndarray) -> BKTModel:
        p0 = np.asarray(p[:n_skills], dtype=np.float64)
        t, g, s = (float(p[n_skills + i]) for i in range(3))
        return cls(n_skills, p0, t, g, s)

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.p_L0, np.array([self.p_t, self.p_g, self.p_s])])

    def fresh_p_L(self) -> FloatArr:
        return np.array(self.p_L0, copy=True, dtype=np.float64)

    def p_predict(self, p_L: FloatArr, skill: int) -> float:
        return p_correct_bkt(float(p_L[skill]), self.p_g, self.p_s)

    def update_only_skill(self, p_L: FloatArr, skill: int, y: int) -> None:
        pl0 = float(p_L[skill])
        post = p_L_posterior(pl0, int(y), self.p_g, self.p_s)
        p_L[skill] = p_L_with_learning(post, self.p_t)

    @staticmethod
    def apply_observation(
        p_L: FloatArr,
        model: BKTModel,
        skill: int,
        y: int,
    ) -> None:
        p_post = p_L_posterior(
            float(p_L[skill]), int(y), model.p_g, model.p_s
        )
        p_L[skill] = p_L_with_learning(p_post, model.p_t)

    @staticmethod
    def neg_log_lik(
        p_vec: np.ndarray, sessions: list[list[tuple[int, int]]], skill_id: list[int]
    ) -> float:
        n_sk = p_vec.size - 3
        m = BKTModel.from_array(n_sk, p_vec)
        nll = 0.0
        for session in sessions:
            pl = m.fresh_p_L()
            for it, y in session:
                s = skill_id[it]
                pc = m.p_predict(pl, s)
                py = float(y) * pc + (1.0 - float(y)) * (1.0 - pc)
                nll -= np.log(_c(py))
                m.apply_observation(pl, m, s, y)
        return nll

    @classmethod
    def fit(
        cls,
        n_skills: int,
        sessions: list[list[tuple[int, int]]],
        item_skill: list[int],
        random_state: int = 0,
    ) -> BKTModel:
        bounds: list[tuple[float, float]] = [(0.01, 0.55)] * n_skills
        bounds += [(0.01, 0.45), (0.01, 0.45), (0.01, 0.45)]
        r = np.random.default_rng(random_state)
        p0 = np.zeros(n_skills + 3, dtype=np.float64)
        p0[:n_skills] = r.uniform(0.1, 0.4, n_skills)
        p0[n_skills] = float(r.uniform(0.05, 0.25))
        p0[n_skills + 1] = float(r.uniform(0.1, 0.35))
        p0[n_skills + 2] = float(r.uniform(0.1, 0.35))

        def nll(p: np.ndarray) -> float:
            return BKTModel.neg_log_lik(p, sessions, item_skill)

        res = minimize(
            nll,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "disp": False},
        )
        return BKTModel.from_array(n_skills, res.x)

    @staticmethod
    def extract_predict_sample_pairs(
        model: BKTModel,
        sessions: list[list[tuple[int, int]]],
        item_skill: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true: list[int] = []
        y_score: list[float] = []
        for session in sessions:
            p_L = model.fresh_p_L()
            for t, (it, y) in enumerate(session):
                s = item_skill[it]
                pred = model.p_predict(p_L, s)
                y_score.append(float(pred))
                y_true.append(int(y))
                model.apply_observation(p_L, model, s, y)
        return (
            np.asarray(y_true, dtype=np.int32),
            np.asarray(y_score, dtype=np.float64),
        )

    @staticmethod
    def bkt_info_score(model: BKTModel, p_L: FloatArr, skill: int) -> float:
        p_c = _c(model.p_predict(p_L, skill))
        if p_c < 1e-9 or p_c > 1.0 - 1e-9:
            return 0.0
        return -(
            p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c)
        )


# --- Elo (1PL) online baseline -------------------------------------------------


@dataclass
class EloBaseline:
    b_item: np.ndarray
    k_update: float = 0.35
    inv_temp: float = 1.0

    @classmethod
    def fit_b_from_train(
        cls, sessions: list[list[tuple[int, int]]], n_items: int, eps: float = 0.01
    ) -> np.ndarray:
        c = np.zeros(n_items, dtype=np.float64)
        n = np.zeros(n_items, dtype=np.float64)
        for session in sessions:
            for it, y in session:
                c[it] += float(y)
                n[it] += 1.0
        acc = np.divide(
            c,
            np.maximum(n, 1.0),
            out=np.full(n_items, 0.5, dtype=np.float64),
            where=n > 0,
        )
        acc = np.clip(acc, eps, 1.0 - eps)
        return np.log((1.0 - acc) / acc)

    @staticmethod
    def p_correct_elo(theta: float, b: float) -> float:
        return 1.0 / (1.0 + np.exp(b - theta))

    def p_predict(
        self, item_id: int, theta: float) -> float:
        return _c(
            1.0
            / (
                1.0
                + np.exp(
                    self.b_item[item_id] - self.inv_temp * theta
                )
            )
        )

    def update_theta(
        self, theta: float, item_id: int, y: int) -> float:
        p = self.p_predict(item_id, theta)
        return theta + self.k_update * (float(y) - p)

    @classmethod
    def from_train(
        cls, sessions: list[list[tuple[int, int]]], n_items: int) -> EloBaseline:
        b = cls.fit_b_from_train(sessions, n_items)
        return cls(b_item=b)

    @staticmethod
    def extract_elo_auc(
        elob: EloBaseline,
        sessions: list[list[tuple[int, int]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        y_t: list[int] = []
        s_t: list[float] = []
        for session in sessions:
            th = 0.0
            for it, y in session:
                s_t.append(elob.p_predict(it, th))
                y_t.append(int(y))
                th = elob.update_theta(th, it, y)
        return (
            np.asarray(y_t, dtype=np.int32),
            np.asarray(s_t, dtype=np.float64),
        )


# --- next-item from candidate pool -------------------------------------------


def bkt_choose(
    model: BKTModel,
    p_L: FloatArr,
    cands: Sequence[int],
    id_skill: list[int],
) -> int:
    best, best_i = -1.0, cands[0]
    for it in cands:
        s = id_skill[it]
        sc = BKTModel.bkt_info_score(model, p_L, s)
        if sc > best:
            best, best_i = sc, it
    return int(best_i)


def elo_choose(
    elob: EloBaseline, theta: float, cands: Sequence[int]
) -> int:
    def score(it: int) -> float:
        p = elob.p_predict(it, theta)
        return -abs(0.5 - p)

    return int(max(cands, key=score))


def item_skill_list_for_seed_curriculum(n_items: int) -> list[int]:
    """Skill index per item id, aligned with 12 seed items (3+2+3+2+2)."""
    pat = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    if n_items <= len(pat):
        return list(pat[:n_items])
    out = list(pat)
    for i in range(len(pat), n_items):
        out.append(i % 5)
    return out


def generate_bkt_synthetic_sessions(
    n_sessions: int,
    len_per: int,
    n_items: int,
    n_skills: int,
    random_state: int = 0,
) -> tuple[list[list[tuple[int, int]]], BKTModel, list[int]]:
    """Label sessions from a hidden BKT; fit is done separately on a train split."""
    rng = np.random.default_rng(random_state)
    true = np.zeros(n_skills + 3, dtype=np.float64)
    true[:n_skills] = rng.uniform(0.1, 0.35, n_skills)
    true[n_skills] = float(rng.uniform(0.04, 0.22))
    true[n_skills + 1] = float(rng.uniform(0.1, 0.32))
    true[n_skills + 2] = float(rng.uniform(0.1, 0.3))
    gen = BKTModel.from_array(n_skills, true)
    item_skill = item_skill_list_for_seed_curriculum(n_items)
    sessions: list[list[tuple[int, int]]] = []
    for _ in range(n_sessions):
        seq: list[tuple[int, int]] = []
        p_l = gen.fresh_p_L()
        for _t in range(len_per):
            it = int(rng.integers(0, n_items))
            s = item_skill[it]
            pc = gen.p_predict(p_l, s)
            y = 1 if float(rng.random()) < _c(float(pc)) else 0
            seq.append((it, y))
            BKTModel.apply_observation(p_l, gen, s, y)
        sessions.append(seq)
    return sessions, gen, item_skill


def next_item_top1_hits(
    bkt: BKTModel,
    elob: EloBaseline,
    test_sessions: list[list[tuple[int, int]]],
    item_skill: list[int],
    n_items: int,
    pool_size: int = 5,
    random_state: int = 1,
) -> tuple[float, float]:
    """Per-step: teacher’s item vs pool; compare BKT (info) vs Elo (50–50) policies."""
    rng = np.random.default_rng(random_state)
    b_h = e_h = 0.0
    ntot = 0
    ps = min(int(pool_size), n_items)
    for session in test_sessions:
        n = len(session)
        for k in range(n):
            p_l = bkt.fresh_p_L()
            th = 0.0
            for j in range(k):
                it, yv = session[j]
                BKTModel.apply_observation(p_l, bkt, item_skill[it], yv)
                th = elob.update_theta(th, it, yv)
            true = int(session[k][0])
            cands: set[int] = {true}
            while len(cands) < ps:
                cands.add(int(rng.integers(0, n_items)))
            c_list = list(cands)
            bid = bkt_choose(bkt, p_l, c_list, item_skill)
            eid = elo_choose(elob, th, c_list)
            b_h += float(bid == true)
            e_h += float(eid == true)
            ntot += 1
    if ntot == 0:
        return 0.0, 0.0
    return b_h / ntot, e_h / ntot
