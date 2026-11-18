#!/usr/bin/env python3
"""
Supplementary code for
  "Measurement-order dependence bounds uncertainty, contextuality,
   steering, and Bell nonlocality".

This file provides reproducible floating-point checks and searches for the
numerical statements in the paper; local optimization is not a certificate
of a global optimum. 

One self-contained file. Requirements: Python 3.12 (tested), NumPy, SciPy;
matplotlib only for the figure.

Usage
  python order_incompatibility_code.py verify      # numerical checks of the paper (~10 min)
  python order_incompatibility_code.py lp          # LP locality, planar grids + n=m=4 search (Prop. 2(v), ~2 min)
  python order_incompatibility_code.py three       # Sec. VI checks + searches quoted for Conjecture 2 (~1 min)
  python order_incompatibility_code.py pm          # schedule minima (Sec. IV B): fast evaluation, then exact search (~10 min)
  python order_incompatibility_code.py conjecture  # searches quoted for Conjecture 1 (~1 h)
  python order_incompatibility_code.py figure      # data + Fig. 1 (~15 min)
  python order_incompatibility_code.py all
Recorded outputs of the `conjecture`, `lp`, `three` and (fast part of) `pm` runs: numerical_search_logs.txt.

Conventions. Records are labelled by measurement, not by time. L\"uders
instruments are used throughout. Logarithms are base 2. Every check
prints "[ok]" or "[FAIL]"; the checks are floating-point tests of proved
statements and of quoted numbers, not proofs. Random seeds are fixed.
"""
import argparse
import itertools
import json
import sys

import numpy as np
from scipy.optimize import brentq, linprog, minimize

RNG = np.random.default_rng(20261111)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
KR = np.kron
FAILURES = []


# ----------------------------------------------------------------------
# Information-theoretic helpers
# ----------------------------------------------------------------------
def shannon(p):
    p = np.asarray(p, float).ravel()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def h2(x):
    return shannon([x, 1.0 - x])


def d_js(p, q):
    """Jensen-Shannon divergence in bits."""
    p = np.asarray(p, float).ravel(); q = np.asarray(q, float).ravel()
    return shannon((p + q) / 2) - 0.5 * (shannon(p) + shannon(q))


def d_tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def d_h2(p, q):
    """Squared Hellinger distance 1 - sum sqrt(pq)."""
    return 1.0 - float(np.sqrt(np.clip(np.asarray(p) * np.asarray(q), 0, None)).sum())


def phi(x):
    """phi(x) = h((1+x)/2) - h(x)/2, decreasing on [0,1]."""
    return h2((1 + x) / 2) - 0.5 * h2(x)


def fano_g(x, k):
    if k < 2:
        if k == 1 and x == 0:
            return 0.0
        raise ValueError("The nontrivial Fano bound requires k >= 2.")
    return h2(x) + x * np.log2(k - 1)


def check(name, ok, info=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILURES.append(name)


# ----------------------------------------------------------------------
# Random objects and optimization over pure probes
# ----------------------------------------------------------------------
def haar_unitary(d, rng=RNG):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_involution(d, rng=RNG):
    """Hermitian unitary with both eigenvalues +1 and -1 present."""
    k = int(rng.integers(1, d))
    u = haar_unitary(d, rng)
    return u @ np.diag([1] * k + [-1] * (d - k)).astype(complex) @ u.conj().T


def random_pvm(d, ranks, rng=RNG):
    u = haar_unitary(d, rng)
    out, k = [], 0
    for r in ranks:
        v = u[:, k:k + r]
        out.append(v @ v.conj().T)
        k += r
    return out


def binary_projectors(a):
    d = a.shape[0]
    return [(np.eye(d) + a) / 2, (np.eye(d) - a) / 2]


def maximize_pure(fun, d, starts=(), n_random=10, rng=RNG):
    """Maximize fun(psi) over unit vectors psi in C^d (multi-start)."""
    best = -np.inf
    seeds = list(starts) + [rng.normal(size=d) + 1j * rng.normal(size=d) for _ in range(n_random)]
    for s in seeds:
        f = lambda x: -fun((x[:d] + 1j * x[d:]) / np.linalg.norm(x))
        r = minimize(f, np.concatenate([np.real(s), np.imag(s)]), method="L-BFGS-B")
        r = minimize(f, r.x, method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 3000})
        best = max(best, -r.fun)
    return best


# ----------------------------------------------------------------------
# Order records
# ----------------------------------------------------------------------
def two_measurement_ops(p_list, q_list):
    """Kraus products for the two orders: rows (a,b) of Q_b P_a and P_a Q_b."""
    fwd = np.array([q @ p for p in p_list for q in q_list])
    rev = np.array([p @ q for p in p_list for q in q_list])
    return fwd, rev


def two_records(fwd, rev, psi):
    """Records p_AB and p_BA on a pure probe psi (projective measurements)."""
    return (np.linalg.norm(fwd @ psi, axis=1) ** 2,
            np.linalg.norm(rev @ psi, axis=1) ** 2)


def sequential_record(rho, proj_lists, order):
    """L\"uders sequential record for PVMs, indexed by measurement label."""
    shape = [len(p) for p in proj_lists]
    out = np.zeros(shape)
    branches = [((), rho)]
    for i in order:
        new = []
        for lab, s in branches:
            for k, p in enumerate(proj_lists[i]):
                t = p @ s @ p
                if np.real(np.trace(t)) > 1e-15:
                    new.append((lab + ((i, k),), t))
        branches = new
    for lab, s in branches:
        idx = [0] * len(proj_lists)
        for i, k in lab:
            idx[i] = k
        out[tuple(idx)] += np.real(np.trace(s))
    return out


def marginal(p, keep):
    return p.sum(axis=tuple(i for i in range(p.ndim) if i not in keep))


def beta_classical(m):
    """beta_C(M) = max over sign vectors s,t of s^T M t."""
    n = m.shape[0]
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    return float(np.abs(signs @ m).sum(axis=1).max())


def comm_score(a, b):
    """T(A,B) = ||[A,B]||/4 for binary sharp observables."""
    return np.linalg.norm(a @ b - b @ a, 2) / 4


def jordan(x, y):
    return (x @ y + y @ x) / 2


def commutator_c(x, y):
    return (x @ y - y @ x) / 2j


# ----------------------------------------------------------------------
# Checks, grouped by section of the manuscript
# ----------------------------------------------------------------------
def check_divergences():
    """Prop. 1: H^2 <= JS <= H^2/ln2 and 1-h((1+T)/2) <= JS <= T."""
    lo, hi, ok = 9.0, 0.0, True
    for _ in range(50000):
        n = int(RNG.integers(2, 6))
        a = RNG.dirichlet(np.ones(n) * RNG.choice([0.1, 1, 5]))
        b = RNG.dirichlet(np.ones(n) * RNG.choice([0.1, 1, 5]))
        j, h, t = d_js(a, b), d_h2(a, b), d_tv(a, b)
        if h > 1e-9:
            lo, hi = min(lo, j / h), max(hi, j / h)
        ok &= 1 - h2((1 + t) / 2) - 1e-12 <= j <= t + 1e-12
    check("JS versus squared Hellinger: ratio in [1, 1/ln2]",
          lo >= 1 - 1e-9 and hi <= 1 / np.log(2) + 1e-6, f"range [{lo:.4f},{hi:.4f}]")
    check("1-h((1+T)/2) <= JS <= T", ok)


def check_binary_calibration():
    """Thm. 1 and Thm. 4(c): T = ||[A,B]||/4; variance bound."""
    for d in (2, 3, 4):
        for _ in range(3):
            a, b = random_involution(d), random_involution(d)
            fwd, rev = two_measurement_ops(binary_projectors(a), binary_projectors(b))
            t_num = maximize_pure(lambda v: d_tv(*two_records(fwd, rev, v)), d)
            t_comm = comm_score(a, b)
            uvar = -maximize_pure(lambda v: -(2 - np.vdot(v, a @ v).real ** 2
                                              - np.vdot(v, b @ v).real ** 2), d)
            bound = 1 - np.sqrt(max(0.0, 1 - 4 * t_comm ** 2))
            ok = abs(t_num - t_comm) < 1e-6 and uvar <= bound + 1e-6
            if d == 2:
                ok &= abs(uvar - bound) < 1e-6
            check(f"binary calibration and U_var bound, d={d}", ok,
                  f"T={t_num:.6f} ||[A,B]||/4={t_comm:.6f} U_var={uvar:.4f}")


def qubit_js_curve(theta, grid=4001):
    """Psi: maximal order information of a qubit pair at Bloch angle theta."""
    c = np.cos(theta)

    def records(t):
        x, y = np.cos(t), np.cos(t - theta)
        p = np.array([(1 + s * x) * (1 + u * c) / 4 for s in (1, -1) for u in (1, -1)])
        q = np.array([(1 + s * u * y) * (1 + u * c) / 4 for s in (1, -1) for u in (1, -1)])
        return p, q

    ts = np.linspace(0, 2 * np.pi, grid)
    vals = [d_js(*records(t)) for t in ts]
    t0 = ts[int(np.argmax(vals))]
    res = minimize(lambda t: -d_js(*records(t[0])), [t0], method="Nelder-Mead",
                   options={"xatol": 1e-12, "fatol": 1e-14})
    hel = max(d_h2(*records(t)) for t in ts)
    return -res.fun, hel


def check_qubit_js():
    """Thm. 1: Psi(1/2) = h(3/4) - 1/2, monotonicity of Psi on 250 points, theta = 0.6 values."""
    vals = [qubit_js_curve(np.arcsin(2 * t), grid=1201)[0] for t in np.linspace(0.002, 0.5, 250)]
    check("Psi(1/2) = h(3/4)-1/2", abs(vals[-1] - (h2(0.75) - 0.5)) < 1e-9)
    check("Psi increasing on a grid of 250 points", all(np.diff(vals) > 0))
    # theta = 0.6: optimal JS value and Bloch vector of the optimal probe (in the A-B plane)
    theta = 0.6
    c = np.cos(theta)
    f = lambda t: -d_js(np.array([(1 + s * np.cos(t)) * (1 + u * c) / 4 for s in (1, -1) for u in (1, -1)]),
                        np.array([(1 + s * u * np.cos(t - theta)) * (1 + u * c) / 4 for s in (1, -1) for u in (1, -1)]))
    ts = np.linspace(-np.pi, np.pi, 20001)
    t_best = ts[int(np.argmin([f(t) for t in ts]))]
    t_best = minimize(lambda x: f(x[0]), [t_best], method="Nelder-Mead", options={"xatol": 1e-12}).x[0]
    vec = np.array([np.sin(t_best), 0.0, np.cos(t_best)])
    # The optimum is fourfold degenerate: the quoted (-0.074, 0, 0.997), its image (0.624, 0, 0.781)
    # under the symmetry A <-> B, and the antipodes of both (relabeling both outcomes).
    quoted = [np.array(v) for v in ((-0.074, 0, 0.997), (0.624, 0, 0.781))]
    vec_ok = min(np.linalg.norm(vec - sg * v) for v in quoted for sg in (1, -1)) < 2e-3
    check("theta=0.6: Psi = 0.1119; optimal Bloch vector is (-0.074, 0, 0.997) up to the stated symmetries",
          abs(-f(t_best) - 0.1119) < 5e-5 and vec_ok, f"Psi={-f(t_best):.5f} r={np.round(vec, 3)}")


def check_section_values():
    """Values quoted in Secs. II D and VI: MUB order information, Hellinger gap, Shannon minima."""
    for d in (3, 4, 5):
        f = np.array([[np.exp(2j * np.pi * a * b / d) for b in range(d)] for a in range(d)]) / np.sqrt(d)
        p_list = [np.diag(e).astype(complex) for e in np.eye(d)]
        q_list = [np.outer(f[:, k], f[:, k].conj()) for k in range(d)]
        fwd, rev = two_measurement_ops(p_list, q_list)
        starts = [np.eye(d)[k] for k in range(d)] + [f[:, k] for k in range(d)]
        j = maximize_pure(lambda v: d_js(*two_records(fwd, rev, v)), d, starts, n_random=12)
        check(f"Fourier MUB d={d}: J = phi(1/d) to 8 digits", abs(j - phi(1 / d)) < 1e-8,
              f"J={j:.10f} phi={phi(1 / d):.10f}")
    gaps = {n: n ** -0.5 - (np.floor((n + 1) / 2) * np.ceil((n + 1) / 2)) ** -0.5 for n in range(3, 200)}
    nbest = max(gaps, key=gaps.get)
    check("Hellinger gap n^{-1/2}-(rs)^{-1/2} is largest (0.135) at n = 11",
          nbest == 11 and abs(gaps[11] - 0.1348) < 1e-4, f"max={gaps[nbest]:.4f}")
    for sign, expect in ((1, 1.650), (-1, 1.812)):
        g = np.full((3, 3), sign / 3.0) + np.eye(3) * (1 - sign / 3.0)
        w, v = np.linalg.eigh(g)
        axes = v @ np.diag(np.sqrt(np.clip(w, 0, None)))      # rows: Bloch vectors with Gram matrix g
        axes = np.hstack([axes, np.zeros((3, max(0, 3 - axes.shape[1])))])[:, :3]

        def shannon_sum(x):
            r = x / max(1.0, np.linalg.norm(x))
            return sum(h2((1 + a @ r) / 2) for a in axes)
        best = min(minimize(shannon_sum, RNG.normal(size=3), method="Nelder-Mead").fun for _ in range(40))
        check(f"three-axis Gram matrix with off-diagonal {sign:+d}/3: minimal Shannon sum ~ {expect}",
              abs(best - expect) < 1e-3, f"value={best:.4f}")


def check_uncertainty():
    """Thm. 4(a),(b): J >= phi(c^2), T >= 1-c^2, q_MU <= U* <= g_k(1-c^2)."""
    for d, ranks in [(3, [1, 1, 1]), (4, [2, 2]), (4, [2, 1, 1])]:
        for _ in range(3):
            p_list, q_list = random_pvm(d, ranks), random_pvm(d, ranks)
            fwd, rev = two_measurement_ops(p_list, q_list)
            starts = [np.linalg.eigh(p)[1][:, -1] for p in p_list + q_list]
            j = maximize_pure(lambda v: d_js(*two_records(fwd, rev, v)), d, starts)
            t = maximize_pure(lambda v: d_tv(*two_records(fwd, rev, v)), d, starts)
            u = -maximize_pure(lambda v: -(shannon([np.vdot(v, p @ v).real for p in p_list])
                                           + shannon([np.vdot(v, q @ v).real for q in q_list])),
                               d, starts)
            c2 = max(np.linalg.norm(p @ q, 2) for p in p_list for q in q_list) ** 2
            k = min(len(p_list), len(q_list))
            ok = (j >= phi(c2) - 1e-7 and t >= 1 - c2 - 1e-7
                  and u <= fano_g(1 - c2, k) + 1e-7 and -np.log2(c2) <= u + 1e-7)
            check(f"uncertainty forces order information, d={d}, ranks={ranks}", ok,
                  f"J={j:.4f} phi(c^2)={phi(c2):.4f} U*={u:.4f}")


def peres_mermin():
    obs = [KR(SX, I2), KR(I2, SX), KR(SX, SX), KR(I2, SY), KR(SY, I2), KR(SY, SY),
           KR(SX, SY), KR(SY, SX), KR(SZ, SZ)]
    return [binary_projectors(o) for o in obs]


def check_contextuality():
    """Sec. IV B: Peres-Mermin and KCBS numbers, bound of Thm. 5."""
    p9 = peres_mermin()
    rho = np.eye(4) / 4
    worst3 = 0.0
    for tri in itertools.combinations(range(9), 3):
        recs = [sequential_record(rho, [p9[k] for k in tri], list(o))
                for o in itertools.permutations(range(3))]
        worst3 = max(worst3, max(d_tv(a, b) for a in recs for b in recs))
    check("PM: all three-measurement records order independent at 1/4", worst3 < 1e-12)
    sub = [p9[k] for k in (0, 1, 2, 3)]
    t4 = d_tv(sequential_record(rho, sub, [0, 1, 2, 3]), sequential_record(rho, sub, [0, 1, 3, 2]))
    check("PM: (XI,IX,XX,IY) vs (XI,IX,IY,XX) has TV 1/2", abs(t4 - 0.5) < 1e-12)
    contexts = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)]
    for _ in range(3):
        pi = list(RNG.permutation(9))
        rec = sequential_record(rho, p9, pi)
        total = 0.0
        for c in contexts:
            pic = list(c) + [k for k in pi if k not in c]
            total += d_tv(marginal(rec, c), marginal(sequential_record(rho, p9, pic), c))
        check("PM: sum of context TV distances >= 1 for a random schedule", total >= 1 - 1e-12,
              f"sum={total:.3f}")
    th = np.arccos(np.sqrt(np.cos(np.pi / 5) / (1 + np.cos(np.pi / 5))))
    vs = [np.array([np.sin(th) * np.cos(4 * np.pi * j / 5), np.sin(th) * np.sin(4 * np.pi * j / 5),
                    np.cos(th)]) for j in range(5)]
    pk = [[np.outer(v, v), np.eye(3) - np.outer(v, v)] for v in vs]
    rk = np.diag([0, 0, 1.0])
    beta = sum(np.trace(rk @ pk[i][0]) for i in range(5))
    rec = sequential_record(rk, pk, [0, 1, 2, 3, 4])
    total = 0.0
    for i in range(5):
        c = tuple(sorted((i, (i + 1) % 5)))
        pic = list(c) + [k for k in range(5) if k not in c]
        total += d_tv(marginal(rec, c), marginal(sequential_record(rk, pk, pic), c))
    pair = d_tv(sequential_record(rk, [pk[0], pk[2]], [0, 1]), sequential_record(rk, [pk[0], pk[2]], [1, 0]))
    check("KCBS: sqrt5-2 <= sum of context TV (1.067); nonadjacent pair TV 0.211",
          beta - 2 <= total and abs(total - 1.0675) < 1e-3 and abs(pair - 0.2111) < 1e-3,
          f"sum={total:.4f} pair={pair:.4f}")


def check_minimal_scenarios():
    """Cor. 2(i),(ii): steering sqrt(2+4T), CHSH 2 sqrt(1+4 T_A T_B), and the MABK formula."""
    for d in (2, 3, 4, 5):
        a0, a1 = random_involution(d), random_involution(d)
        b0, b1 = random_involution(2), random_involution(2)
        ta, tb = comm_score(a0, a1), comm_score(b0, b1)
        s = np.linalg.norm(KR(a0, SX) + KR(a1, SZ), 2)
        chsh = np.linalg.norm(KR(a0, b0 + b1) + KR(a1, b0 - b1), 2)
        check(f"steering and CHSH formulas, d={d}",
              abs(s - np.sqrt(2 + 4 * ta)) < 1e-9 and abs(chsh - 2 * np.sqrt(1 + 4 * ta * tb)) < 1e-9)
    # MABK operator of Cor. 2(ii): ||M_N||^2 = [prod(1+2T_k) + prod(1-2T_k)]/2
    dev = 0.0
    for sites in ((2, 2), (2, 3), (3, 3), (2, 2, 2), (2, 3, 2), (2, 2, 2, 2)):
        for _ in range(5):
            x = [(random_involution(d), random_involution(d)) for d in sites]
            m, mp = x[0][0], x[0][1]                      # M_1 and its primed version
            for x1, x2 in x[1:]:
                m, mp = (KR(m, x1 + x2) + KR(mp, x1 - x2)) / 2, (KR(mp, x2 + x1) + KR(m, x2 - x1)) / 2
            t = [comm_score(x1, x2) for x1, x2 in x]
            pred = (np.prod([1 + 2 * u for u in t]) + np.prod([1 - 2 * u for u in t])) / 2
            dev = max(dev, abs(np.linalg.norm(m, 2) ** 2 - pred))
    check("MABK formula of Cor. 2(ii), N = 2, 3, 4, local dimensions 2 and 3", dev < 1e-9, f"dev={dev:.1e}")


def check_unsharp():
    """Eq. (28) for unsharp qubit Lueders measurements and the numbers quoted after Corollary 1."""
    def score(eta, zeta):
        ea = [(I2 + sg * eta * SZ) / 2 for sg in (1, -1)]
        eb = [(I2 + sg * zeta * SX) / 2 for sg in (1, -1)]
        def root(e):
            w, v = np.linalg.eigh(e)
            return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T

        diff = [root(a) @ b @ root(a) - root(b) @ a @ root(b) for a in ea for b in eb]
        best = 0.0
        for mask in range(1, 16):                         # Eq. (3): maximum over events
            op = sum(dm for i, dm in enumerate(diff) if mask >> i & 1)
            best = max(best, np.linalg.eigvalsh((op + op.conj().T) / 2).max())
        return best
    dev = 0.0
    for eta, zeta in ((0.9, 0.9), (0.5, 0.8), (1.0, 0.3), (0.2, 0.2), (0.7, 1.0)):
        pred = 0.5 * max(eta * (1 - np.sqrt(1 - zeta ** 2)), zeta * (1 - np.sqrt(1 - eta ** 2)))
        dev = max(dev, abs(score(eta, zeta) - pred))
    check("Eq. (28): order score of unsharp qubit measurements", dev < 1e-9, f"dev={dev:.1e}")
    eta = 0.9
    required = ((2 * np.sqrt(2) * eta) ** 2 / 4 - 1) / 4          # Corollary 1 for CHSH
    actual = score(eta, eta) * 0.5                                 # ideal partner: T_B = 1/2
    check("unsharp example after Cor. 1: required 0.155, actual 0.127",
          abs(required - 0.155) < 5e-4 and abs(actual - 0.127) < 5e-4, f"required={required:.4f}, actual={actual:.4f}")


def concurrence(rho):
    yy = KR(SY, SY)
    ev = np.sqrt(np.clip(np.sort(np.linalg.eigvals(rho @ yy @ rho.conj() @ yy).real)[::-1], 0, None))
    return max(0.0, ev[0] - ev[1] - ev[2] - ev[3])


def check_concurrence_bound():
    """Cor. 2(iii): beta^2/4 - 1 <= min(C^2, 4 T_A T_B) for two qubits."""
    ok = True
    for _ in range(300):
        g = RNG.normal(size=(4, 4)) + 1j * RNG.normal(size=(4, 4))
        rho = g @ g.conj().T
        if RNG.random() < 0.5:
            v = RNG.normal(size=4) + 1j * RNG.normal(size=4)
            rho = np.outer(v, v.conj())
        rho /= np.trace(rho).real
        a0, a1, b0, b1 = (random_involution(2) for _ in range(4))
        beta = abs(np.trace(rho @ (KR(a0, b0 + b1) + KR(a1, b0 - b1))).real)
        ok &= beta ** 2 / 4 - 1 <= min(concurrence(rho) ** 2, 4 * comm_score(a0, a1) * comm_score(b0, b1)) + 1e-9
    check("beta^2/4 - 1 <= min(C^2, 4 T_A T_B)", ok)


def check_assemblage_bound():
    """Thm. 6(a): sum_a ||sigma_a - tau_a||_1 <= n_x T_A, with T_A exact from Eq. (3)."""
    for d, nout in [(2, 2), (3, 2), (3, 3), (4, 3)]:
        worst = 0.0
        for _ in range(20):
            def pvm(u):
                groups = np.array_split(np.arange(d), nout)
                return [u[:, gidx] @ u[:, gidx].conj().T for gidx in groups]
            e0, e1 = pvm(haar_unitary(d)), pvm(haar_unitary(d))
            psi = RNG.normal(size=d * d) + 1j * RNG.normal(size=d * d)
            psi /= np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())

            def bob(e):
                return np.trace((KR(e, np.eye(d)) @ rho).reshape(d, d, d, d), axis1=0, axis2=2)
            sig = [bob(e) for e in e1]
            tau = [bob(sum(p @ e @ p for p in e0)) for e in e1]
            dist = sum(np.abs(np.linalg.eigvalsh((s + s.conj().T) / 2 - (t + t.conj().T) / 2)).sum()
                       for s, t in zip(sig, tau))
            ta = tv_score_exact(e0, e1)
            worst = max(worst, dist / (nout * ta) if ta > 1e-9 else 0.0)
        check(f"assemblage bound, d={d}, outcomes={nout}", worst <= 1 + 1e-6, f"max ratio={worst:.3f}")


def jordan_part(m, a_list, b_list):
    n, k = m.shape
    return sum(m[x, y] * m[xp, yp] * KR(jordan(a_list[x], a_list[xp]), jordan(b_list[y], b_list[yp]))
               for x in range(n) for xp in range(n) for y in range(k) for yp in range(k))


def check_product_structure():
    """Thm. 7(a)-(c): identity (16), two-setting bound, Eq. (18), cut decomposition; Thm. 8."""
    ok_id = ok_two = True
    for trial in range(150):
        n = 2 if trial < 100 else int(RNG.integers(3, 5))
        k = int(RNG.integers(2, 5))
        da, db = int(RNG.integers(2, 4)), int(RNG.integers(2, 4))
        m = RNG.normal(size=(n, k))
        a_list = [random_involution(da) for _ in range(n)]
        b_list = [random_involution(db) for _ in range(k)]
        bell = sum(m[x, y] * KR(a_list[x], b_list[y]) for x in range(n) for y in range(k))
        jpart = jordan_part(m, a_list, b_list)
        cpart = sum(m[x, y] * m[xp, yp] * KR(commutator_c(a_list[x], a_list[xp]), commutator_c(b_list[y], b_list[yp]))
                    for x in range(n) for xp in range(n) for y in range(k) for yp in range(k))
        ok_id &= np.allclose(bell @ bell, jpart - cpart, atol=1e-9)
        bc = beta_classical(m)
        if n == 2:
            rhs = bc ** 2 + 8 * comm_score(a_list[0], a_list[1]) * sum(
                abs(m[0, y] * m[1, yp]) * comm_score(b_list[y], b_list[yp])
                for y in range(k) for yp in range(k) if y != yp)
            ok_two &= np.linalg.norm(bell, 2) ** 2 <= rhs + 1e-9
    check("identity B^2 = Gamma - sum MM K(x)K (d_A, d_B in {2,3})", ok_id)
    check("two-setting bilinear bound (d_A, d_B in {2,3})", ok_two)
    ok_exact = True
    for _ in range(60):
        th = RNG.uniform(0, np.pi)
        k = int(RNG.integers(2, 5)); db = int(RNG.integers(2, 4))
        m = RNG.normal(size=(2, k))
        b_list = [random_involution(db) for _ in range(k)]
        p0 = sum(m[0, y] * b_list[y] for y in range(k))
        p1 = sum(m[1, y] * b_list[y] for y in range(k))
        base = p0 @ p0 + p1 @ p1 + np.cos(th) * (p0 @ p1 + p1 @ p0)
        r = 1j * (p0 @ p1 - p1 @ p0)
        pred = max(np.linalg.eigvalsh(base + np.sin(th) * r).max(), np.linalg.eigvalsh(base - np.sin(th) * r).max())
        a1 = np.cos(th) * SZ + np.sin(th) * SX
        ok_exact &= abs(np.linalg.norm(KR(SZ, p0) + KR(a1, p1), 2) ** 2 - pred) < 1e-9
    check("exact two-setting formula", ok_exact)
    ok_s = eq_s = True
    for _ in range(120):
        n = int(RNG.integers(2, 4)); da = int(RNG.integers(2, 5))
        a_list = [random_involution(da) for _ in range(n)]
        s = np.linalg.norm(sum(KR(a_list[x], [SX, SZ, SY][x]) for x in range(n)), 2)
        bound = np.sqrt(n + 4 * sum(comm_score(a_list[x], a_list[xp]) for x in range(n) for xp in range(x + 1, n)))
        ok_s &= s <= bound + 1e-9
        if n == 2:
            eq_s &= abs(s - bound) < 1e-9
    s3 = np.linalg.norm(sum(KR(p, p) for p in (SX, SY, SZ)), 2)
    check("steering with an anticommuting partner", ok_s and eq_s and abs(s3 - 3) < 1e-12)
    ok_w = ok_l = True
    n_dec = 0
    pats = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    for _ in range(200):
        da, db, k = int(RNG.integers(2, 5)), int(RNG.integers(2, 4)), int(RNG.integers(2, 5))
        a_list = [random_involution(da) for _ in range(3)]
        b_list = [random_involution(db) for _ in range(k)]
        m = RNG.normal(size=(3, k))
        ident = np.eye(da)
        w = {}
        for s in pats:
            ssum = sum(si * ai for si, ai in zip(s, a_list))
            w[s] = (ssum @ ssum - ident) / 8
        ok_w &= np.allclose(sum(w.values()), ident)
        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            ok_w &= np.allclose(sum(ws * s[i] * s[j] for s, ws in w.items()), jordan(a_list[i], a_list[j]))
        if all(np.linalg.eigvalsh(ws).min() >= -1e-12 for ws in w.values()):
            n_dec += 1
            ok_l &= np.linalg.eigvalsh(jordan_part(m, a_list, b_list)).max() <= beta_classical(m) ** 2 + 1e-9
    check("cut decomposition W_s = ((sum s A)^2 - 1)/8", ok_w)
    check("Jordan bound on cut-decomposable triples", ok_l, f"({n_dec} triples)")


def check_hs_grothendieck(trials=40):
    """Prop. 2: max ||sum M u v^T||_F^2 / beta_C^2 over unit vectors stays <= 1."""
    worst = 0.0
    for _ in range(trials):
        n, k = int(RNG.integers(2, 6)), int(RNG.integers(2, 6))
        m = RNG.normal(size=(n, k)); r = max(n, k)

        def neg_frob(p):
            u = p[:n * r].reshape(n, r); v = p[n * r:].reshape(k, r)
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            return -((u.T @ m @ v) ** 2).sum()
        best = max(-minimize(neg_frob, RNG.normal(size=(n + k) * r), method="L-BFGS-B").fun for _ in range(4))
        worst = max(worst, best / beta_classical(m) ** 2)
    check("Hilbert-Schmidt Grothendieck inequality (numerical)", worst <= 1 + 1e-6, f"max ratio={worst:.6f}")


def check_revision_items():
    """Statements added in the revision (Sec. II D, IV B, V B; Prop. 2(iii))."""
    # Remark 1: full record versus optimized marginal disturbance
    u = np.array([[99, -12, -16], [-12, 29, -96], [-16, -96, -27]]) / 101.0
    p_list = [np.diag(e).astype(complex) for e in np.eye(3)]
    q_list = [np.outer(u[:, k], u[:, k]).astype(complex) for k in range(3)]
    fwd, rev = two_measurement_ops(p_list, q_list)
    psi = np.array([39, 68, 62]) / np.sqrt(9989)
    t_probe = d_tv(*two_records(fwd, rev, psi))

    def marg(v, axis):
        x, y = two_records(fwd, rev, v)
        return d_tv(x.reshape(3, 3).sum(axis), y.reshape(3, 3).sum(axis))
    # A local maximization gives a LOWER bound, not the upper bound needed
    # to certify this strict separation. Enumerate all marginal events.
    def directed_disturbance(pl, ql):
        vals = []
        for mask in range(1, 2 ** len(pl) - 1):
            e = sum(pl[i] for i in range(len(pl)) if mask >> i & 1)
            delta = e - sum(q @ e @ q for q in ql)
            vals.append(np.max(np.abs(np.linalg.eigvalsh(delta))))
        return max(vals, default=0.0)
    dist_a = directed_disturbance(p_list, q_list)
    dist_b = directed_disturbance(q_list, p_list)
    check("Remark 1: both marginal disturbances < 0.3152 < record TV",
          max(dist_a, dist_b) < 0.3152 < 0.31553 <= t_probe + 1e-12,
          f"disturbances={dist_a:.5f},{dist_b:.5f} record={t_probe:.5f}")
    # Sec. IV B: three qubit measurements on 1/2 differ by t/2
    ok = True
    for t in (0.5, 0.9, 0.99):
        obs = [SZ, SY, np.sqrt(1 - t * t) * SX + t * SZ]
        ps = [binary_projectors(o) for o in obs]
        tv = d_tv(sequential_record(I2 / 2, ps, [0, 1, 2]), sequential_record(I2 / 2, ps, [1, 0, 2]))
        ok &= abs(tv - t / 2) < 1e-12
    check("qubit three-measurement order effect equals t/2", ok)
    # Sec. IV B: KCBS beta' <= 2 for every 0/1 assignment
    vals = [sum(x[i] - x[i] * x[(i + 1) % 5] for i in range(5)) for x in itertools.product([0, 1], repeat=5)]
    check("KCBS beta' <= 2 for all assignments", max(vals) == 2)
    # Sec. V B: counterexample to the Jordan-part estimate
    m = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 0]], float)

    def family(p):
        v = np.array([np.sqrt(p), np.sqrt((1 - p) / 2), np.sqrt((1 - p) / 2)])
        a3 = np.eye(3) - 2 * np.outer(v, v)
        a_list = [np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1]), a3]
        b_list = [np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1]), -a3]
        bell = sum(m[x, y] * KR(a_list[x], b_list[y]) for x in range(3) for y in range(3))
        rhs = 16 + 4 * sum(abs(m[x, y] * m[xp, yp]) * comm_score(a_list[x], a_list[xp]) * comm_score(b_list[y], b_list[yp])
                           for x in range(3) for xp in range(3) for y in range(3) for yp in range(3) if x != xp and y != yp)
        return np.linalg.eigvalsh(jordan_part(m, a_list, b_list)).max(), np.linalg.norm(bell, 2) ** 2, rhs
    lam, bq2, rhs = family(0.5)
    best = max(family(p)[0] for p in np.linspace(0.5, 0.65, 301)) / 16
    check("Jordan-part estimate fails (16.538 > 16) while beta_Q^2 = 16 <= 22",
          abs(lam - 16.538234) < 1e-5 and abs(bq2 - 16) < 1e-9 and abs(rhs - 22) < 1e-9 and abs(best - 1.03457) < 1e-4,
          f"ratio max={best:.5f}")
    # Prop. 2(iii): Z = U W V^T with ||W||_HS = 1 is local when n = 3
    worst = 0.0
    for _ in range(30):
        n, k, r = 3, int(RNG.integers(3, 6)), int(RNG.integers(2, 4))
        uu = RNG.normal(size=(n, r)); uu /= np.linalg.norm(uu, axis=1, keepdims=True)
        vv = RNG.normal(size=(k, r)); vv /= np.linalg.norm(vv, axis=1, keepdims=True)
        w = RNG.normal(size=(r, r)); w /= np.linalg.norm(w)
        worst = max(worst, local_norm(uu @ w @ vv.T))
    check("local norm of U W V^T <= 1 for three settings (random instances)", worst <= 1 + 1e-9, f"max={worst:.6f}")


def conjecture_search_adversarial():
    """Sec. V B evidence (i): gradient ascent of beta_Q^2 - (bilinear bound) over involutions
    with a one-dimensional +1 eigenspace in (d_A, d_B) = (2,2), (3,3), (2,3), four restarts each,
    for the chained and I3322 correlation functionals and seven random 3x3 functionals.
    Uses the operator norm (both spectral endpoints), not just lambda_max.
    The historical search log predates this correction; see the dated audit logs."""
    from scipy.linalg import expm
    rng = np.random.default_rng(11)

    def herm_(v, d):
        hh = np.zeros((d, d), complex); iu = np.triu_indices(d, 1); k = len(iu[0])
        hh[np.diag_indices(d)] = v[:d]; hh[iu] = v[d:d + k] + 1j * v[d + k:d + 2 * k]
        return hh + np.triu(hh, 1).conj().T

    def obs(v, d, k):
        u = expm(1j * herm_(v, d)); return u @ np.diag([1] * k + [-1] * (d - k)) @ u.conj().T

    def gap(params, m, da, db):
        n, k = m.shape; la, lb = da * da, db * db
        a_list = [obs(params[i * la:(i + 1) * la], da, 1) for i in range(n)]
        b_list = [obs(params[n * la + j * lb:n * la + (j + 1) * lb], db, 1) for j in range(k)]
        bell = sum(m[x, y] * np.kron(a_list[x], b_list[y]) for x in range(n) for y in range(k))
        bq = np.max(np.abs(np.linalg.eigvalsh(bell)))
        kk = sum(abs(m[x, y] * m[xp, yp]) * comm_score(a_list[x], a_list[xp]) * comm_score(b_list[y], b_list[yp])
                 for x in range(n) for xp in range(n) for y in range(k) for yp in range(k) if x != xp and y != yp)
        return bq ** 2 - beta_classical(m) ** 2 - 4 * kk
    tests = [("chained3", np.array([[1, 1, 0], [0, 1, 1], [-1, 0, 1]], float)),
             ("I3322corr", np.array([[1, 1, 1], [1, 1, -1], [1, -1, 0]], float))]
    tests += [(f"rand{i}", rng.normal(size=(3, 3))) for i in range(4)]
    tests += [(f"sign{i}", rng.choice([-1, 1], size=(3, 3)).astype(float)) for i in range(3)]
    for name, m in tests:
        best = -1e9
        for da, db in [(2, 2), (3, 3), (2, 3)]:
            for _ in range(4):
                n, k = m.shape
                p0 = rng.normal(size=n * da * da + k * db * db) * rng.choice([0.05, 0.5, 2])
                res = minimize(lambda p: -gap(p, m, da, db), p0, method="L-BFGS-B", options={"maxiter": 400})
                best = max(best, -res.fun)
        print(f"{name:10s} beta_C={beta_classical(m):.3f}  max[beta_Q^2 - bilinear bound] = {best:.3e}", flush=True)
        check(f"Conjecture 1, adversarial search {name}", best <= 1e-7)


def conjecture_search_random():
    """Sec. V B evidence (ii): 4000 random instances with 2-3 settings per party, d_A, d_B in {2,3},
    40% of them nearly commuting. Conjecture 1 is a theorem if one party has two settings
    (Theorem 7(c)) and for two qubits (Proposition 2(iii) with Theorem 7(a)), so only the instances
    with three settings on both sides and at least one qutrit test it. Prints the number of such
    instances that violate beta_C and their largest (beta_Q^2-beta_C^2)/(4 sum|MM| T T)."""
    rng = np.random.default_rng(3)

    def haar_(d):
        z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); q, r = np.linalg.qr(z)
        return q * (np.diag(r) / abs(np.diag(r)))

    def inv_(d, eps=None, base=None):
        k = rng.integers(1, d); s = np.diag([1] * k + [-1] * (d - k)).astype(complex)
        if base is None:
            u = haar_(d)
        else:
            hh = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            u, _ = np.linalg.qr(base @ (np.eye(d) + eps * ((hh + hh.conj().T) / 2) * 1j))
        return u @ s @ u.conj().T
    worst, worst_open = -1.0, -1.0
    n_viol = n_open = n_open_viol = 0
    for _ in range(4000):
        n = rng.integers(2, 4); k = rng.integers(2, 4); da = rng.integers(2, 4); db = rng.integers(2, 4)
        m = rng.choice([-1, 0, 1], size=(n, k)).astype(float) if rng.random() < 0.5 else rng.normal(size=(n, k))
        if rng.random() < 0.4:
            ua, ub = haar_(da), haar_(db)
            ea, eb = 10 ** rng.uniform(-3, -0.5), 10 ** rng.uniform(-3, -0.5)
            a_list = [inv_(da, ea, ua) for _ in range(n)]; b_list = [inv_(db, eb, ub) for _ in range(k)]
        else:
            a_list = [inv_(da) for _ in range(n)]; b_list = [inv_(db) for _ in range(k)]
        bell = sum(m[x, y] * np.kron(a_list[x], b_list[y]) for x in range(n) for y in range(k))
        bq, bc = np.max(np.abs(np.linalg.eigvalsh(bell))), beta_classical(m)
        kk = sum(abs(m[x, y] * m[xp, yp]) * comm_score(a_list[x], a_list[xp]) * comm_score(b_list[y], b_list[yp])
                 for x in range(n) for xp in range(n) for y in range(k) for yp in range(k) if x != xp and y != yp)
        is_open = n == 3 and k == 3 and (da == 3 or db == 3)      # not covered by a theorem
        n_open += is_open
        if bq > bc + 1e-9:
            ratio = (bq ** 2 - bc ** 2) / (4 * kk) if kk > 1e-14 else np.inf
            n_viol += 1
            worst = max(worst, ratio)
            if is_open:
                n_open_viol += 1
                worst_open = max(worst_open, ratio)
    print("largest (beta_Q^2 - beta_C^2)/(4 sum|MM| T T) over violating instances:", worst)
    print(f"instances: 4000; with beta_Q > beta_C: {n_viol}; outside the proved cases: {n_open}; "
          f"outside the proved cases with beta_Q > beta_C: {n_open_viol}, largest ratio among them: {worst_open:.4f}")
    check("Conjecture 1, random instances: ratio <= 1", worst < 1 + 1e-8)


def schedule_minima_fast(samples=20):
    """Sec. IV B, fast evaluation (a few seconds). Peres-Mermin on 1/4: the marginal of a schedule
    on a context is either exact or uniform (TV 0 or 1/2). With the context members in time order
    (f, s, t), it is uniform exactly when a measurement outside the context lies between f and s
    and anticommutes with f, or lies between s and t and anticommutes with t. The rule is checked
    against explicit records on random schedules and then applied to all 9! schedules.
    KCBS: explicit records of all 120 schedules."""
    obs = [KR(SX, I2), KR(I2, SX), KR(SX, SX), KR(I2, SY), KR(SY, I2), KR(SY, SY), KR(SX, SY), KR(SY, SX), KR(SZ, SZ)]
    ctx = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)]
    anti = np.array([[np.linalg.norm(a @ b + b @ a) < 1e-12 for b in obs] for a in obs])
    pl = [binary_projectors(o) for o in obs]
    rho = np.eye(4) / 4
    rng = np.random.default_rng(17)

    def rule(perm, c):
        pos = {o: i for i, o in enumerate(perm)}
        f, mid, t = sorted(c, key=lambda o: pos[o])
        return (any(o not in c and anti[o, f] for o in perm[pos[f] + 1:pos[mid]])
                or any(o not in c and anti[o, t] for o in perm[pos[mid] + 1:pos[t]]))

    dev = 0.0
    for _ in range(samples):
        perm = [int(v) for v in rng.permutation(9)]
        rec = sequential_record(rho, pl, perm)
        for c in ctx:
            first = list(c) + [o for o in perm if o not in c]
            explicit = d_tv(marginal(rec, c), marginal(sequential_record(rho, pl, first), c))
            dev = max(dev, abs(explicit - 0.5 * rule(perm, c)))
    check(f"Peres-Mermin: commutation rule for the context marginals ({samples} random schedules)", dev < 1e-12,
          f"dev={dev:.1e}")
    perms = np.array(list(itertools.permutations(range(9))), dtype=np.int8)
    pos = np.empty_like(perms)
    pos[np.arange(len(perms))[:, None], perms] = np.arange(9, dtype=np.int8)[None, :]
    total = np.zeros(len(perms))
    for c in ctx:
        times = pos[:, list(c)].astype(int)
        order = np.argsort(times, axis=1)
        members = np.array(c)
        f, t = members[order[:, 0]], members[order[:, 2]]
        ts = np.sort(times, axis=1)
        uniform = np.zeros(len(perms), bool)
        for o in range(9):
            if o not in c:
                po = pos[:, o].astype(int)
                uniform |= (po > ts[:, 0]) & (po < ts[:, 1]) & anti[o, f]
                uniform |= (po > ts[:, 1]) & (po < ts[:, 2]) & anti[o, t]
        total += 0.5 * uniform
    print(f"Peres-Mermin, all 9! schedules: min of sum_C TV = {total.min()}, attained by "
          f"{int((total == total.min()).sum())} schedules; max = {total.max()}")
    check("Peres-Mermin: minimum over all 9! schedules of sum_C TV = 1.5 (commutation rule)", total.min() == 1.5)
    th = np.arccos(np.sqrt(np.cos(np.pi / 5) / (1 + np.cos(np.pi / 5))))
    vs = [np.array([np.sin(th) * np.cos(4 * np.pi * j / 5), np.sin(th) * np.sin(4 * np.pi * j / 5), np.cos(th)])
          for j in range(5)]
    pk = [[np.outer(v, v).astype(complex), np.eye(3) - np.outer(v, v)] for v in vs]
    rk = np.diag([0, 0, 1.0]).astype(complex)
    best = np.inf
    for perm in itertools.permutations(range(5)):
        rec = sequential_record(rk, pk, perm)
        tot = 0.0
        for i in range(5):
            c = tuple(sorted((i, (i + 1) % 5)))
            first = list(c) + [o for o in perm if o not in c]
            tot += d_tv(marginal(rec, c), marginal(sequential_record(rk, pk, first), c))
        best = min(best, tot)
    print(f"KCBS, all 120 schedules: min of sum_C TV = {best:.4f}; required sqrt(5) - 2 = {np.sqrt(5) - 2:.4f}")
    check("KCBS: minimum over all 120 schedules of sum_C TV = 1.067", abs(best - 1.0675) < 1e-3)


def run_pm():
    """Sec. IV B: minimum over all schedules of sum_C TV for Peres-Mermin (1/4) and KCBS (exact,
    depth-first enumeration with pruning; several minutes)."""
    proj = lambda o: np.array([(np.eye(o.shape[0]) + o) / 2, (np.eye(o.shape[0]) - o) / 2])

    def run(ps, rho, contexts):
        pc = []
        for c in contexts:
            p = np.zeros([2] * len(c))
            for xs in itertools.product(range(2), repeat=len(c)):
                op = np.eye(rho.shape[0], dtype=complex)
                for kk, xx in zip(c, xs):
                    op = op @ ps[kk][xx]
                p[xs] = np.real(np.trace(rho @ op))
            pc.append(p)
        best = [np.inf, None]

        def dfs(prefix, remaining, states, placed, partial):
            if partial >= best[0] - 1e-15:
                return
            if not remaining:
                best[0], best[1] = partial, list(prefix)
                return
            for kk in sorted(remaining):
                new_states, new_placed, add = [], [], 0.0
                for ci, c in enumerate(contexts):
                    st, pl = states[ci], placed[ci]
                    if st is None:
                        new_states.append(None); new_placed.append(pl); continue
                    pp = ps[kk]
                    if kk in c:
                        st2 = np.einsum("jab,nbc,jcd->njad", pp, st, pp).reshape(-1, *st.shape[1:])
                        pl2 = pl + [kk]
                        if len(pl2) == len(c):
                            probs = np.real(np.einsum("naa->n", st2)).reshape([2] * len(c))
                            add += 0.5 * np.abs(np.transpose(probs, [pl2.index(cc) for cc in c]) - pc[ci]).sum()
                            new_states.append(None)
                        else:
                            new_states.append(st2)
                        new_placed.append(pl2)
                    else:
                        new_states.append(np.einsum("jab,nbc,jcd->nad", pp, st, pp)); new_placed.append(pl)
                dfs(prefix + [kk], remaining - {kk}, new_states, new_placed, partial + add)
        dfs([], set(range(len(ps))), [rho[None].astype(complex) for _ in contexts], [[] for _ in contexts], 0.0)
        return best
    obs = [KR(SX, I2), KR(I2, SX), KR(SX, SX), KR(I2, SY), KR(SY, I2), KR(SY, SY), KR(SX, SY), KR(SY, SX), KR(SZ, SZ)]
    pm = run([proj(o) for o in obs], np.eye(4) / 4, [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)])
    th = np.arccos(np.sqrt(np.cos(np.pi / 5) / (1 + np.cos(np.pi / 5))))
    vs = [np.array([np.sin(th) * np.cos(4 * np.pi * j / 5), np.sin(th) * np.sin(4 * np.pi * j / 5), np.cos(th)])
          for j in range(5)]
    pk = [np.array([np.outer(v, v), np.eye(3) - np.outer(v, v)]).astype(complex) for v in vs]
    kc = run(pk, np.diag([0, 0, 1.0]).astype(complex), [(i, (i + 1) % 5) for i in range(5)])
    check("Peres-Mermin: minimum over all 9! schedules of sum_C TV = 1.5", abs(pm[0] - 1.5) < 1e-9, f"{pm[0]}")
    check("KCBS: minimum over all 120 schedules of sum_C TV = 1.067", abs(kc[0] - 1.0675) < 1e-3, f"{kc[0]:.4f}")


def tv_score_exact(p_list, q_list):
    """Finite event enumeration, with floating-point Hermitian eigensolves."""
    diffs = [p @ q @ p - q @ p @ q for p in p_list for q in q_list]
    best = 0.0
    for mask in range(1, 2 ** len(diffs)):
        s = sum(diffs[i] for i in range(len(diffs)) if mask >> i & 1)
        best = max(best, np.linalg.eigvalsh(s).max())
    return best


def marginal_disturbance_exact(p_list, q_list):
    """Largest optimized disturbance of one marginal by the other L\"uders measurement."""
    def one(pl, ql):
        best = 0.0
        for mask in range(1, 2 ** len(pl) - 1):
            px = sum(pl[i] for i in range(len(pl)) if mask >> i & 1)
            best = max(best, np.abs(np.linalg.eigvalsh(px - sum(q @ px @ q for q in ql))).max())
        return best
    return max(one(p_list, q_list), one(q_list, p_list))


def check_marginal_gap(samples=1500):
    """Remark 1 and Sec. II D: exact T of the support pair (0.640 < 2/3) and the largest
    T - (marginal disturbance) over random qutrit bases (< 7e-4)."""
    p_list = [np.diag(e).astype(complex) for e in np.eye(3)]
    u0 = np.array([[.5, .5, 1 / np.sqrt(2)], [.5, .5, -1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2), 0]])
    t_support = tv_score_exact(p_list, [np.outer(u0[:, k], u0[:, k]).astype(complex) for k in range(3)])
    check("support pair n=3 has T = 0.640 < 2/3", abs(t_support - 0.6404) < 1e-4, f"T={t_support:.4f}")
    rng = np.random.default_rng(7)
    gap = 0.0
    for _ in range(samples):
        z = rng.normal(size=(3, 3)) + (1j * rng.normal(size=(3, 3)) if rng.random() < 0.5 else 0)
        q, _ = np.linalg.qr(z)
        q_list = [np.outer(q[:, k], q[:, k].conj()) for k in range(3)]
        gap = max(gap, tv_score_exact(p_list, q_list) - marginal_disturbance_exact(p_list, q_list))
    check(f"random qutrit bases: T - marginal disturbance < 7e-4 ({samples} samples)", gap < 7e-4, f"max={gap:.2e}")


# ----------------------------------------------------------------------
# Sec. VI: three binary sharp measurements and the KCBS pentagon
# ----------------------------------------------------------------------
def check_three_measurements(trials=400):
    """Theorem 9, Eqs. (E1)-(E2) and the statements of Sec. VI A."""
    sg = np.array([1.0, -1.0])
    dev = dict.fromkeys(["moments", "middle", "tv", "bell", "alt", "opid", "steer"], 0.0)
    kbound, sched, chsh = -np.inf, -np.inf, -np.inf

    def moment(p, f):
        return float(sum(f(sg[i], sg[j], sg[k]) * p[i, j, k]
                         for i in range(2) for j in range(2) for k in range(2)))

    def triples():
        for _ in range(trials):
            d = int(RNG.integers(2, 7))
            yield tuple(random_involution(d) for _ in range(3))
        for _ in range(20):        # degenerate triples: trivial observable, equal or commuting pairs
            d = int(RNG.integers(2, 6))
            a, c = random_involution(d), random_involution(d)
            u = haar_unitary(d)
            diag = lambda: (u * RNG.choice([1.0, -1.0], size=d)) @ u.conj().T
            for t in ((a, np.eye(d, dtype=complex), c), (a, a, c), (a, c, a), (diag(), diag(), c),
                      (diag(), c, diag())):
                yield t

    count = 0
    for a, b, c in triples():
        count += 1
        d = a.shape[0]
        eye = np.eye(d)
        mm = eye / d
        tau = lambda x: float(np.real(np.trace(x))) / d
        pl = [binary_projectors(x) for x in (a, b, c)]
        kab, kbc = commutator_c(a, b), commutator_c(b, c)
        cab, cbc, cac = jordan(a, b), jordan(b, c), jordan(a, c)
        kap_a = tau(commutator_c(b, a) @ commutator_c(a, c))     # middle A
        kap_b = tau(kab @ kbc)                                   # middle B
        kap_c = tau(commutator_c(a, c) @ commutator_c(c, b))     # middle C
        # Eq. (E1): moments of the record of (A,B,C) on a random pure input
        psi = RNG.normal(size=d) + 1j * RNG.normal(size=d)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        ex = lambda x: float(np.real(np.trace(rho @ x)))
        lud = lambda y, z: (z + y @ z @ y) / 2          # nonselective Lueders map of y
        p = sequential_record(rho, pl, (0, 1, 2))
        dev["moments"] = max(
            dev["moments"],
            abs(moment(p, lambda x, y, z: x * y) - ex(cab)),
            abs(moment(p, lambda x, y, z: x * z) - ex(jordan(a, lud(b, c)))),
            abs(moment(p, lambda x, y, z: y * z) - ex(lud(a, cbc))),
            abs(moment(p, lambda x, y, z: x * y * z) - ex(jordan(a, cbc))))
        # schedule bound for general inputs (end of Sec. VI A)
        p_bc = sequential_record(rho, pl, (1, 2))[0]        # unmeasured labels stay at index 0
        p_ac = sequential_record(rho, pl, (0, 2))[:, 0]
        rhs = 1 + 2 * d_tv(p.sum(axis=0), p_bc) + 2 * d_tv(p.sum(axis=1), p_ac)
        sched = max(sched, ex(cab + cbc - cac) - rhs)
        # Theorem 9(a): maximally mixed input; the record depends on the middle measurement only
        p_abc = sequential_record(mm, pl, (0, 1, 2))
        p_cba = sequential_record(mm, pl, (2, 1, 0))
        p_acb = sequential_record(mm, pl, (0, 2, 1))
        p_bac = sequential_record(mm, pl, (1, 0, 2))
        dev["middle"] = max(dev["middle"], d_tv(p_abc, p_cba))
        dev["tv"] = max(dev["tv"],
                        abs(d_tv(p_abc, p_acb) - max(abs(kap_b), abs(kap_c)) / 2),
                        abs(d_tv(p_abc, p_bac) - max(abs(kap_b), abs(kap_a)) / 2),
                        abs(d_tv(p_acb, p_bac) - max(abs(kap_c), abs(kap_a)) / 2))
        kbound = max(kbound, abs(kap_b) - 4 * comm_score(a, b) * comm_score(b, c))
        # steered ensemble: TV(p_ABC, p_ACB) = sum_a p(a) T(B,C | rho_a)
        avg = 0.0
        for pa in pl[0]:
            w = float(np.real(np.trace(pa))) / d
            if w > 1e-12:
                ra = pa / np.trace(pa)
                avg += w * d_tv(sequential_record(ra, pl[1:], (0, 1)), sequential_record(ra, pl[1:], (1, 0)))
        dev["steer"] = max(dev["steer"], abs(avg - d_tv(p_abc, p_acb)))
        # Theorem 9(b): Bell 1964 on |Phi> = sum_j |jj>/sqrt(d), transpose in the same basis
        phi = np.eye(d).reshape(-1) / np.sqrt(d)
        e = lambda x, y: float(np.real(phi.conj() @ KR(x, y.T) @ phi))
        beta3 = e(a, b) + e(b, c) - e(a, c)
        alt = p_abc[0, 1, 0] + p_abc[1, 0, 1]
        dev["bell"] = max(dev["bell"], abs(beta3 - (1 + kap_b - 4 * alt)))
        dev["alt"] = max(dev["alt"], abs(4 * alt - tau((eye - cab) @ (eye - cbc))))
        chsh = max(chsh, beta3 - (2 * np.sqrt(1 + 4 * comm_score(a, b) * comm_score(b, c)) - 1))
        # Eq. (E2): operator identity for general inputs
        rhs_op = (jordan(kab, kbc) - jordan(eye - cab, eye - cbc)
                  - 0.5j * ((cab @ kbc - kbc @ cab) + (kab @ cbc - cbc @ kab)))
        dev["opid"] = max(dev["opid"], np.linalg.norm(cab + cbc - cac - eye - rhs_op, 2))
    info = lambda key: f"dev={dev[key]:.1e}"
    check(f"Eq. (E1): moments of three-measurement records ({count} triples)", dev["moments"] < 1e-10, info("moments"))
    check("Thm 9(a): record of 1/d depends on the middle measurement only", dev["middle"] < 1e-10, info("middle"))
    check("Thm 9(a): TV = max(|kappa_Y|,|kappa_Y'|)/2 for all three pairs of middles", dev["tv"] < 1e-10, info("tv"))
    check("Thm 9(a): |kappa_B| <= 4 T(A,B) T(B,C)", kbound < 1e-10, f"max excess={kbound:.1e}")
    check("Thm 9(b): beta_3 = 1 + kappa_B - 4 p_ABC(alternating)", dev["bell"] < 1e-10, info("bell"))
    check("4 p_ABC(alternating) = Tr[(1-A.B)(1-B.C)]/d", dev["alt"] < 1e-10, info("alt"))
    check("TV(p_ABC,p_ACB) = average order effect on the steered ensemble", dev["steer"] < 1e-10, info("steer"))
    check("beta_3 <= 2 sqrt(1 + 4 T(A,B) T(B,C)) - 1", chsh < 1e-10, f"max excess={chsh:.1e}")
    check("Eq. (E2): operator identity for the Leggett-Garg operator", dev["opid"] < 1e-10, info("opid"))
    check("schedule bound for general inputs", sched < 1e-10, f"max excess={sched:.1e}")
    # qubit statement: beta_3 = 3/2 at T = sqrt(3)/4
    th = np.pi / 3
    a, b = SZ, np.cos(th) * SZ + np.sin(th) * SX
    c = np.cos(2 * th) * SZ + np.sin(2 * th) * SX
    beta3 = float(np.real(np.trace(a @ b + b @ c - a @ c))) / 2
    check("qubits: beta_3 = 3/2 at T(A,B) = T(B,C) = sqrt(3)/4",
          abs(beta3 - 1.5) < 1e-12 and abs(comm_score(a, b) - np.sqrt(3) / 4) < 1e-12
          and abs(comm_score(b, c) - np.sqrt(3) / 4) < 1e-12)
    # example with commuting A, B on C^3 (Appendix E)
    a = np.diag([1.0, -1.0, 1.0]).astype(complex)
    b = np.diag([-1.0, 1.0, 1.0]).astype(complex)
    w = np.array([np.sqrt(3), 0, 1]) / 2
    c = (np.eye(3) - 2 * np.outer(w, w)).astype(complex)
    psi = np.array([1, 0, np.sqrt(3)]) / 2
    ex = lambda x: float(np.real(psi @ x @ psi))
    vals = (ex(jordan(a, b)), ex(jordan(b, c)), -ex(jordan(a, c)))
    check("C^3 example: [A,B]=0 and Leggett-Garg combination 3/2",
          comm_score(a, b) < 1e-15 and max(abs(v - 0.5) for v in vals) < 1e-12, f"values={np.round(vals, 6)}")


def _pentagons(n, real):
    """n random pentagons of unit vectors in C^3 with v_i orthogonal to v_{i+1}."""
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)

    def perp(v):
        r = RNG.normal(size=v.shape) + (0 if real else 1j * RNG.normal(size=v.shape))
        for _ in range(2):                  # second pass removes round-off in the projection
            r = unit(r - np.sum(v.conj() * r, axis=-1, keepdims=True) * v)
        return r

    v1 = np.zeros((n, 3), dtype=complex)
    v1[:, 2] = 1
    v2 = perp(v1); v3 = perp(v2); v4 = perp(v3)
    v5 = np.cross(v4, v1).conj()
    ok = np.linalg.norm(v5, axis=-1) > 1e-3     # skip nearly degenerate pentagons
    return np.stack([x[ok] for x in (v1, v2, v3, v4)] + [unit(v5[ok])], axis=1)


def _pentagon_stats(vs):
    """V, chord overlaps z_i, chord scores T_i and crossing products T_i T_(i+1) of pentagons in C^3."""
    gram = np.einsum("nia,nja->nij", vs.conj(), vs)
    v = np.linalg.eigvalsh(gram)[:, -1] - 2
    z = np.stack([np.abs(gram[:, i, (i + 2) % 5]) ** 2 for i in range(5)], axis=1)
    t = np.sqrt(np.clip(z * (1 - z), 0, None))
    return v, z, t, t * np.roll(t, -1, axis=1)


def check_kcbs_pentagon(n=200000):
    """Theorem 10 on random real and complex pentagons in C^3 and on degenerate pentagons."""
    for real in (True, False):
        vs = _pentagons(n, real)
        v, z, t, cross = _pentagon_stats(vs)
        delta = 2 - z.sum(1)
        z3 = np.roll(z, -3, axis=1)
        tag = "real" if real else "complex"
        d1 = np.abs(z3 - (1 - z) * (1 - np.roll(z, -1, axis=1))).max()
        d2 = np.abs(delta[:, None] * (1 - z3) - cross ** 2).max()
        d3 = np.abs(delta - v ** 2 * (1 + v) / (1 - v)).max()
        # delta is of order V^2, so the two signs are compared with matching tolerances
        d4 = int((((delta > 1e-9) & (v <= 0)) | ((v > 1e-4) & (delta <= 0))).sum())
        pos = v > 1e-4                 # below this, sqrt(1-z) loses digits near z = 1
        d5 = np.abs(cross[pos] - (v[pos] * np.sqrt((1 + v[pos]) / (1 - v[pos])))[:, None]
                    * np.sqrt(1 - z3[pos])).max()
        gap = (v[pos] - cross[pos].max(1)).max()
        check(f"Thm 10(a), {tag}: z_(i+3) = (1-z_i)(1-z_(i+1))", d1 < 1e-9, f"dev={d1:.1e} ({vs.shape[0]} pentagons)")
        check(f"Eq. (E3), {tag}: delta (1-z_(i+3)) = (T_i T_(i+1))^2", d2 < 1e-9, f"dev={d2:.1e}")
        check(f"Thm 10(b), {tag}: sum z_i <= 2, V >= 0, V>0 iff strict; 2 - sum z_i = V^2(1+V)/(1-V)",
              d3 < 1e-9 and d4 == 0 and delta.min() > -1e-12 and v.min() > -1e-12,
              f"dev={d3:.1e}, exceptions={d4}")
        check(f"Thm 10(c), {tag}: formula for T_i T_(i+1)", d5 < 1e-9, f"dev={d5:.1e}")
        check(f"Thm 10(c), {tag}: V <= max_i T_i T_(i+1) for V > 1e-4", gap < 1e-10, f"max(V - rhs)={gap:.1e}")
    # degenerate pentagons with v_3 parallel to v_1: overlaps (1, cos^2, 0, 0, sin^2), V = 0
    worst = 0.0
    for ang in np.linspace(0.0, np.pi / 2, 31):
        e1, e2, e3 = np.eye(3)
        vs = np.array([[e1, e2, e1, np.cos(ang) * e2 + np.sin(ang) * e3, -np.sin(ang) * e2 + np.cos(ang) * e3]],
                      dtype=complex)
        v, z, t, cross = _pentagon_stats(vs)
        worst = max(worst, abs(v[0]), abs(2 - z.sum()), np.abs(cross).max(),
                    np.abs(np.roll(z, -3, axis=1) - (1 - z) * (1 - np.roll(z, -1, axis=1))).max())
    check("Thm 10, degenerate pentagons (some z_j = 1): V = 0, sum z_i = 2, T_i T_(i+1) = 0", worst < 1e-12,
          f"dev={worst:.1e}")
    zs = (3 - np.sqrt(5)) / 2
    check("KCBS optimum: V = T^2 = sqrt(5) - 2", abs(zs * (1 - zs) - (np.sqrt(5) - 2)) < 1e-14)


def check_necessary_chords(trials=300):
    """Proposition 3 on random instances."""
    worst_a = worst_b = -np.inf
    for _ in range(trials):
        # (a) A_5 = s central; A_2, A_3 commute; A_1 commutes with A_2 and A_4 with A_3.
        #     The chords (1,3), (2,4) and (4,1) are noncommuting.
        a2 = np.diag([1, 1, 1, -1, -1, -1]).astype(complex)
        a3 = np.diag([1, 1, -1, 1, -1, -1]).astype(complex)

        def block_involution(idx_sets):
            m = np.zeros((6, 6), dtype=complex)
            for idx in idx_sets:
                m[np.ix_(idx, idx)] = random_involution(len(idx))
            return m

        a1 = block_involution(([0, 1, 2], [3, 4, 5]))
        a4 = block_involution(([0, 1, 3], [2, 4, 5]))
        for s in (1.0, -1.0):
            a = [a1, a2, a3, a4, s * np.eye(6)]
            minus_s = -sum(a[i] @ a[(i + 1) % 5] for i in range(5))
            worst_a = max(worst_a, np.linalg.eigvalsh((minus_s + minus_s.conj().T) / 2).max() - 3)
        # (b) exclusive pentagon on C^6 with commuting chord (3,5): P_3 and P_5 share one eigenvector
        u = haar_unitary(6)
        proj = lambda cols: cols @ cols.conj().T
        orth = lambda m: np.linalg.qr(m)[0]
        p3, p5 = proj(u[:, [0, 1]]), proj(u[:, [0, 2]])
        rand = lambda k: RNG.normal(size=(6, k)) + 1j * RNG.normal(size=(6, k))
        p4 = proj(orth(u[:, 3:] @ (RNG.normal(size=(3, 1)) + 1j * RNG.normal(size=(3, 1)))))
        q2 = orth((np.eye(6) - p3) @ rand(1))
        p2 = proj(q2)
        span = orth(np.hstack([u[:, [0, 2]], q2]))
        p1 = proj(orth((np.eye(6) - span @ span.conj().T) @ rand(1)))
        ps = [p1, p2, p3, p4, p5]
        excl = max(np.linalg.norm(ps[i] @ ps[(i + 1) % 5]) for i in range(5))
        assert excl < 1e-10 and np.linalg.norm(p3 @ p5 - p5 @ p3) < 1e-10
        worst_b = max(worst_b, np.linalg.eigvalsh(sum(ps)).max() - 2)
    check(f"Prop. 3(a): central A_5 gives -S <= 3 ({2 * trials} instances with three noncommuting chords)",
          worst_a < 1e-10, f"max excess={worst_a:.1e}")
    check(f"Prop. 3(b): exclusive pentagon with one commuting chord has lambda_max <= 2 ({trials} instances)",
          worst_b < 1e-10, f"max excess={worst_b:.1e}")


def pentagon_rank_one_scan(n=400000, seed=5):
    """Conjecture 2 for rank-one projectors in every dimension. The Gram matrix of an exclusive
    pentagon of unit vectors is 1 + H >= 0 with H supported on the chord 5-cycle (1,3,5,2,4), and
    every such matrix is a Gram matrix of vectors in dimension rank(1 + H) <= 5."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1, size=(n, 5)).astype(complex)
    h[:, 0] *= np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))     # only the total phase matters
    mat = np.zeros((n, 5, 5), dtype=complex)
    for k in range(5):
        mat[:, k, (k + 1) % 5] = h[:, k]
        mat[:, (k + 1) % 5, k] = h[:, k].conj()
    lam = np.linalg.eigvalsh(mat)
    # scale H so that 1 + H >= 0; half of the samples lie on the boundary (rank at most 4)
    scale = np.where(rng.random(n) < 0.5, 1.0, rng.uniform(0.05, 1.0, size=n)) / np.abs(lam[:, 0])
    v = scale * lam[:, -1] - 1
    z = (scale[:, None] * np.abs(h)) ** 2
    t = np.sqrt(np.clip(z * (1 - z), 0, None))
    cross = t * np.roll(t, -2, axis=1)       # crossing chords are non-adjacent edges of the chord cycle
    pos = v > 1e-4
    ratio = (v[pos] / cross[pos].max(1)).max()
    print(f"\nConjecture 2, rank one in any dimension: {n} random Gram matrices, {int(pos.sum())} with V > 1e-4; "
          f"largest V / max_i T_i T_(i+1) = {ratio:.6f}", flush=True)
    check("Conjecture 2, rank-one Gram-matrix scan: ratio <= 1", ratio < 1 + 1e-9)


def pentagon_search(configs=((4, (1,) * 5), (5, (1,) * 5), (6, (2,) * 5), (7, (2,) * 5), (9, (3,) * 5),
                             (5, (1, 1, 1, 1, 2))),
                    restarts=6, steps=800, vmin=0.02, seed=11):
    """Random local search quoted for Conjecture 2: exclusive pentagons of projectors with the given
    ranks on C^d, maximizing rho = V / max_i T_i T_(i+1) subject to V >= vmin."""
    rng = np.random.default_rng(seed)
    orth = lambda m: np.linalg.qr(m)[0]

    def build(g, d):
        eye = np.eye(d)
        q = [orth(g[0])]
        for i in range(1, 4):
            q.append(orth((eye - q[-1] @ q[-1].conj().T) @ g[i]))
        span = orth(np.hstack([q[3], q[0]]))
        q.append(orth((eye - span @ span.conj().T) @ g[4]))
        return [u @ u.conj().T for u in q]

    def objective(p):
        v = np.linalg.eigvalsh(sum(p)).max() - 2
        t = np.array([np.linalg.norm(p[i] @ p[(i + 2) % 5] - p[(i + 2) % 5] @ p[i], 2) for i in range(5)])
        rho = v / (t * np.roll(t, -1)).max()
        return (rho if v >= vmin else v - vmin), rho, v      # below vmin: first increase V

    print(f"\nConjecture 2: random local search over exclusive pentagons, maximizing V / max_i T_i T_(i+1) "
          f"at V >= {vmin} ({restarts} restarts x {steps} steps, seed {seed})")
    for d, ranks in configs:
        best_rho, best_v = -np.inf, 0.0
        for _ in range(restarts):
            gauss = lambda: [rng.normal(size=(d, r)) + 1j * rng.normal(size=(d, r)) for r in ranks]
            g = gauss()
            cur, rho, v = objective(build(g, d))
            eps = 0.3
            for it in range(steps):
                g2 = [x + eps * y for x, y in zip(g, gauss())]
                new, rho2, v2 = objective(build(g2, d))
                if new > cur:
                    g, cur, rho, v = g2, new, rho2, v2
                elif it % 60 == 59:
                    eps *= 0.7
            if v >= vmin and rho > best_rho:
                best_rho, best_v = rho, v
        print(f"  d = {d}, ranks {ranks}: largest ratio = {best_rho:.5f}  (at V = {best_v:.5f})", flush=True)
        check(f"Conjecture 2 search, d={d}, ranks={ranks}: ratio <= 1", best_rho < 1 + 1e-9)


def run_verify():
    check_revision_items()
    check_marginal_gap()
    check_section_values()
    check_divergences()
    check_binary_calibration()
    check_qubit_js()
    check_uncertainty()
    check_contextuality()
    check_minimal_scenarios()
    check_unsharp()
    check_concurrence_bound()
    check_assemblage_bound()
    check_product_structure()
    check_hs_grothendieck()
    check_three_measurements()
    check_kcbs_pentagon()
    check_necessary_chords()
    print("\nALL CHECKS PASSED" if not FAILURES else f"\nFAILED: {FAILURES}")


# ----------------------------------------------------------------------
# Linear-programming locality certificates (Prop. 2, planar case)
# ----------------------------------------------------------------------
def local_norm(z):
    """Finite-strategy LP for the local norm, solved in floating point.

    Check solver status, reconstruction and primal/dual objective residuals.
    This is NOT an exact-arithmetic locality certificate.
    """
    n, k = z.shape
    s_all = np.array(list(itertools.product([1, -1], repeat=n - 1)))
    s_all = np.hstack([np.ones((len(s_all), 1)), s_all])
    t_all = np.array(list(itertools.product([1, -1], repeat=k)))
    cols = np.einsum("ix,jy->ijxy", s_all, t_all).reshape(len(s_all) * len(t_all), n * k).T
    mat = np.hstack([cols, -cols])
    res = linprog(np.ones(2 * cols.shape[1]), A_eq=mat, b_eq=z.ravel(),
                  bounds=(0, None), method="highs")
    if not res.success or res.fun is None or not np.isfinite(res.fun):
        raise RuntimeError(f"Locality LP failed: {res.message}")
    primal = np.max(np.abs(mat @ res.x - z.ravel()))
    dual = np.asarray(res.eqlin.marginals)
    dual_violation = max(0.0, float(np.max(mat.T @ dual - 1)))
    gap = abs(float(res.fun) - float(z.ravel() @ dual))
    scale = max(1.0, float(np.max(np.abs(z))), abs(float(res.fun)))
    if min(res.x) < -1e-8 or max(primal, dual_violation, gap) > 1e-7 * scale:
        raise RuntimeError(f"Uncertified LP residuals: primal={primal}, dual={dual_violation}, gap={gap}")
    return float(res.fun)


def run_lp(n=8):
    worst = 0.0
    for ph in np.linspace(0, np.pi / 4, 9):
        dmat = np.diag([np.cos(ph), np.sin(ph)])
        for off in np.linspace(0, np.pi / n, 5):
            a = np.array([[np.cos(np.pi * j / n), np.sin(np.pi * j / n)] for j in range(n)])
            b = np.array([[np.cos(off + np.pi * j / n), np.sin(off + np.pi * j / n)] for j in range(n)])
            val = local_norm(a @ dmat @ b.T)
            worst = max(worst, val)
        print(f"phi={ph:.4f}: largest local norm so far {worst:.6f}", flush=True)
    check(f"planar <u, diag(c,s) v> local on {n}-direction grids", worst <= 1 + 1e-7, f"max={worst:.6f}")


def planar_search(sizes=(4,), restarts=4, seed=5):
    """Prop. 2(v), n = m >= 4 (open case): Nelder-Mead over all 2n angles and over
    (c, s) = (cos phi, sin phi), maximizing a floating-point estimate of the local norm of
    Z_xy = c cos a_x cos b_y + s sin a_x sin b_y. Prop. 2(v) predicts max <= 1."""
    rng = np.random.default_rng(seed)
    for n in sizes:
        def neg_norm(p):
            a, b, ph = p[:n], p[n:2 * n], p[-1]
            z = np.cos(ph) * np.outer(np.cos(a), np.cos(b)) + np.sin(ph) * np.outer(np.sin(a), np.sin(b))
            return -local_norm(z)
        worst = 0.0
        for _ in range(restarts):
            res = minimize(neg_norm, rng.uniform(0, np.pi, 2 * n + 1), method="Nelder-Mead",
                           options={"maxfev": 400, "xatol": 1e-7, "fatol": 1e-11})
            worst = max(worst, -res.fun)
        check(f"adversarial planar search, n = m = {n} ({restarts} restarts): max local norm <= 1 + 1e-8",
              worst <= 1 + 1e-8, f"max local norm={worst:.12f}")


# ----------------------------------------------------------------------
# Figure 1
# ----------------------------------------------------------------------
def figure_data(n_pairs=260, path="fig1_data.json"):
    rng = np.random.default_rng(11)
    panel_a = []
    for t in np.linspace(0, 0.5, 51):
        js_val, hel = qubit_js_curve(np.arcsin(min(2 * t, 1.0))) if t > 0 else (0.0, 0.0)
        panel_a.append(dict(T=t, JS=js_val, H2=hel))
    d = 3
    p_list = [np.outer(e, e) for e in np.eye(d)]

    def pair(u):
        q_list = [np.outer(u[:, k], u[:, k].conj()) for k in range(d)]
        fwd, rev = two_measurement_ops(p_list, q_list)
        starts = [np.eye(d)[k] for k in range(d)] + [u[:, k] for k in range(d)]
        j = maximize_pure(lambda v: d_js(*two_records(fwd, rev, v)), d, starts, 8, rng)
        us = -maximize_pure(lambda v: -(shannon(np.abs(v) ** 2) + shannon(np.abs(u.conj().T @ v) ** 2)),
                            d, starts, 8, rng)
        return dict(U=us, J=j)
    panel_b = [pair(haar_unitary(d, rng)) for _ in range(n_pairs)]
    f3 = np.array([[np.exp(2j * np.pi * a * b / 3) for b in range(3)] for a in range(3)]) / np.sqrt(3)
    u0 = np.array([[.5, .5, 1 / np.sqrt(2)], [.5, .5, -1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2), 0]])
    ds = np.eye(3, dtype=complex)
    ds[1:, 1:] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    special = {name: pair(u) for name, u in [("MUB", f3), ("support", u0), ("direct_sum", ds)]}
    boundary = []
    for uu in np.linspace(0, np.log2(3), 200):
        x = brentq(lambda x: fano_g(x, 3) - uu, 0, 2 / 3) if uu > 0 else 0.0
        boundary.append(dict(U=uu, J=phi(1 - x)))
    data = dict(a=panel_a, b=panel_b, special=special, boundary=boundary)
    json.dump(data, open(path, "w"), indent=0)
    return data


def plot_figure(data, out="fig1.pdf"):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "legend.fontsize": 7,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "xtick.direction": "in", "ytick.direction": "in"})
    colors = ["#2C3E6B", "#5B7DB1", "#E8915A", "#9B7DB8", "#D4726A", "#6AAF6E"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.35), constrained_layout=True)
    t = np.array([r["T"] for r in data["a"]])
    js_val = np.array([r["JS"] for r in data["a"]])
    ax.plot(t, js_val / js_val[-1], "-", color=colors[0], lw=1.4, label=r"order information $\mathcal{J}$")
    ax.plot(t, 2 * t, "--", color=colors[1], lw=1.4, label="TV score; maximal Robertson bound")
    ax.plot(t, (np.sqrt(1 + 2 * t) - 1) / (np.sqrt(2) - 1), "-.", color=colors[2], lw=1.4,
            label="maximal steering and CHSH excess")
    ax.plot(t, 1 - np.sqrt(np.clip(1 - 4 * t ** 2, 0, None)), ls=(0, (5, 1, 1, 1, 1, 1)), color=colors[3],
            lw=1.4, label=r"$U_{\rm var}=1-\sqrt{1-4\mathcal{T}^{2}}$ (qubit pairs)")
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"order score $\mathcal{T}=\Vert[A,B]\Vert/4$"); ax.set_ylabel("normalized signature")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, frameon=False)
    ax.text(0.97, 0.04, "(a)", transform=ax.transAxes, ha="right")
    ub = np.array([r["U"] for r in data["boundary"]]); jb = np.array([r["J"] for r in data["boundary"]])
    bx.fill_between(ub, 0, jb, color="#EDEDED", lw=0)
    bx.plot(ub, jb, "k", lw=1.1, label=r"bound $\varphi(1-g_3^{-1}(U_*))$")
    bx.scatter([r["U"] for r in data["b"]], [r["J"] for r in data["b"]], s=6, color=colors[1], alpha=0.55,
               lw=0, label="Haar-random qutrit pairs")
    sp = data["special"]
    bx.scatter([sp["MUB"]["U"]], [sp["MUB"]["J"]], marker="s", s=26, color=colors[4], label="MUB")
    bx.scatter([sp["support"]["U"]], [sp["support"]["J"]], marker="D", s=24, color=colors[2],
               label=r"support pair ($r=s=2$)")
    bx.scatter([sp["direct_sum"]["U"]], [sp["direct_sum"]["J"]], marker="^", s=28, color=colors[5],
               label=r"$1\oplus F_2$")
    bx.set_xlim(-0.03, 1.62); bx.set_ylim(0, 0.56)
    bx.set_xlabel(r"minimal entropic uncertainty $U_*$ (bits)"); bx.set_ylabel(r"order information $\mathcal{J}$ (bits)")
    bx.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, frameon=False)
    bx.text(0.97, 0.04, "(b)", transform=bx.transAxes, ha="right")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("figure written to", out)


def main(argv=None):
    """Run a selected suite; return a nonzero status for any failed check."""
    FAILURES.clear()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="verify",
                        choices=("verify", "lp", "three", "pm", "conjecture", "figure", "all"))
    mode = parser.parse_args(argv).mode
    if mode in ("verify", "all"):
        run_verify()
    if mode in ("lp", "all"):
        run_lp()
        planar_search()
    if mode in ("three", "all"):
        if mode == "three":
            check_three_measurements()
            check_kcbs_pentagon()
            check_necessary_chords()
        pentagon_rank_one_scan()
        pentagon_search()
    if mode in ("pm", "all"):
        schedule_minima_fast()
        run_pm()
    if mode in ("conjecture", "all"):
        conjecture_search_adversarial()
        conjecture_search_random()
    if mode in ("figure", "all"):
        plot_figure(figure_data())
    if FAILURES:
        print(f"\nFAILED: {FAILURES}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
