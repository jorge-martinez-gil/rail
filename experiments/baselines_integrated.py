"""Integration layer wiring strong noise-robust baselines into the main replay loop.

The classical noise-robust gates in :mod:`experiments.baselines` (CoTeaching,
SelfPaced, JointAgreement, DynamicQuantile) already exist as standalone
dataclasses but were not visible to ``rail_paper.replay_once`` because they do
not share the per-event ``score / admit / weight`` interface used by the
publication harness.

This module fixes that. It provides:

* A ``StatefulPolicy`` adapter class whose ``score / admits / weight`` methods
  are statefully driven, suitable for drop-in use in any replay loop.
* Factories for the integrated baselines:

  - ``CoTeachingPolicy``  -- two-view small-loss selection (Han et al., 2018).
  - ``SelfPacedPolicy``   -- curriculum admission (Kumar et al., 2010).
  - ``JointAgreementPolicy`` -- JoCoR-style agreement filter (Wei et al., 2020).
  - ``DynamicQuantilePolicy`` -- streaming quantile of admission score.
  - ``GCEWeightPolicy``   -- Generalized Cross-Entropy down-weighting
    (Zhang & Sabuncu, 2018). Always-admit, weight by ``(1 - p_y^q)/q``.
  - ``SCEWeightPolicy``   -- Symmetric Cross-Entropy (Wang et al., 2019).
  - ``ITLMPolicy``        -- Iterative trimmed-loss minimization (Shen
    & Sanghavi, 2019).

All policies expose the same per-event interface as the existing baselines, so
they can be added to ``run_all_policies_once`` without modifying the core
``replay_once`` loop -- see :func:`make_integrated_policies` for a turn-key
factory and :func:`register_integrated_policies` to monkey-patch the dispatch
in :mod:`experiments.rail_paper` for backwards compatibility.

References
----------
* Han et al., "Co-teaching: Robust Training of Deep Neural Networks with
  Extremely Noisy Labels," NeurIPS 2018.
* Wei et al., "Combating Noisy Labels by Agreement: A Joint Training Method
  with Co-Regularization," CVPR 2020.
* Kumar et al., "Self-Paced Learning for Latent Variable Models," NeurIPS 2010.
* Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural
  Networks with Noisy Labels," NeurIPS 2018.
* Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy
  Labels," ICCV 2019.
* Shen & Sanghavi, "Learning with Bad Training Data via Iterative Trimmed Loss
  Minimization," ICML 2019.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:
    from .baselines import (
        CoTeachingGate,
        DynamicQuantileGate,
        JointAgreementGate,
        SelfPacedGate,
    )
except ImportError:  # pragma: no cover - script-style import fallback
    from baselines import (  # type: ignore[no-redef]
        CoTeachingGate,
        DynamicQuantileGate,
        JointAgreementGate,
        SelfPacedGate,
    )

EPS = 1e-12

__all__ = [
    "INTEGRATED_POLICY_NAMES",
    "CoTeachingPolicy",
    "DynamicQuantilePolicy",
    "GCEWeightPolicy",
    "ITLMPolicy",
    "JointAgreementPolicy",
    "SCEWeightPolicy",
    "SelfPacedPolicy",
    "StatefulPolicy",
    "make_integrated_policies",
]


# ---------------------------------------------------------------------------
# Shared utilities.
# ---------------------------------------------------------------------------


def _cross_entropy(probs: np.ndarray, y: int) -> float:
    p = float(np.clip(np.asarray(probs)[int(y)], EPS, 1.0))
    return -math.log(p)


def _two_view_losses(probs: np.ndarray, y: int, rng: np.random.Generator) -> tuple[float, float]:
    """Cheap two-view loss split for CoTeaching/JointAgreement on a single model.

    We perturb the predicted distribution with two independent Dirichlet
    samples concentrated around ``probs``; the resulting losses behave like
    two correlated-but-distinct critics. This avoids requiring two separately
    trained models in the streaming setting where memory is tight.
    """
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    alpha = 50.0 * probs  # high concentration -> small perturbation
    p_a = rng.dirichlet(alpha)
    p_b = rng.dirichlet(alpha)
    return _cross_entropy(p_a, y), _cross_entropy(p_b, y)


# ---------------------------------------------------------------------------
# Stateful policy contract.
# ---------------------------------------------------------------------------


@dataclass
class StatefulPolicy:
    """Per-event policy with mutable state that survives between events.

    The contract:

    * ``name`` -- unique policy name used in summary tables.
    * ``score(probs, y_human, telemetry) -> float`` -- a real-valued admission
      score (interpretation depends on the policy).
    * ``admits(probs, y_human, telemetry) -> bool`` -- admission decision.
    * ``weight(probs, y_human, telemetry) -> float`` -- non-negative weight
      applied at update time; 0.0 means "discard".

    Stateful policies update their internal counters/quantiles on each call.
    """

    name: str

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        raise NotImplementedError

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        raise NotImplementedError

    def weight(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        return 1.0 if self.admits(probs, y_human, telemetry) else 0.0


# ---------------------------------------------------------------------------
# Adapter policies wrapping the existing gates.
# ---------------------------------------------------------------------------


@dataclass
class CoTeachingPolicy(StatefulPolicy):
    name: str = "co_teaching"
    forget_rate: float = 0.3
    ramp_steps: int = 500
    window: int = 500
    seed: int = 0
    _gate: CoTeachingGate = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gate = CoTeachingGate(
            forget_rate=self.forget_rate,
            ramp_steps=self.ramp_steps,
            window=self.window,
        )
        self._rng = np.random.default_rng(self.seed)

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        return -_cross_entropy(probs, y_human)  # higher = lower loss = more admissible

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        la, lb = _two_view_losses(probs, y_human, self._rng)
        return bool(self._gate.decide(la, lb))


@dataclass
class SelfPacedPolicy(StatefulPolicy):
    name: str = "self_paced"
    lambda_0: float = 1.0
    growth: float = 1.02
    step: int = 100
    cap: float = 25.0
    _gate: SelfPacedGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gate = SelfPacedGate(
            lambda_0=self.lambda_0,
            growth=self.growth,
            step=self.step,
            cap=self.cap,
        )

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        return -_cross_entropy(probs, y_human)

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        return bool(self._gate.decide(_cross_entropy(probs, y_human)))


@dataclass
class JointAgreementPolicy(StatefulPolicy):
    name: str = "joint_agreement"
    keep_fraction: float = 0.7
    window: int = 500
    seed: int = 0
    _gate: JointAgreementGate = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gate = JointAgreementGate(
            keep_fraction=self.keep_fraction,
            window=self.window,
        )
        self._rng = np.random.default_rng(self.seed)

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        return -_cross_entropy(probs, y_human)

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        la, lb = _two_view_losses(probs, y_human, self._rng)
        # two-view predictions: argmax of perturbed distributions
        probs_arr = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
        alpha = 50.0 * probs_arr
        p_a = self._rng.dirichlet(alpha)
        p_b = self._rng.dirichlet(alpha)
        pred_a = int(np.argmax(p_a))
        pred_b = int(np.argmax(p_b))
        return bool(self._gate.decide(pred_a, pred_b, la, lb))


@dataclass
class DynamicQuantilePolicy(StatefulPolicy):
    """Streaming-quantile gate over the RAIL vigilance score (V).

    This is the strongest non-trivial "adaptive threshold" baseline: it uses
    RAIL's V score but lets a P^2 quantile estimator pick the operating point
    online, with no manual ``theta``. Demonstrating that a tuned ``theta``
    still beats this baseline isolates the value of the *contract* on top of
    the score.
    """

    name: str = "dynamic_quantile"
    upper: float = 0.6
    warmup: int = 50
    score_fn: Callable[[np.ndarray, int, object], float] | None = None
    _gate: DynamicQuantileGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gate = DynamicQuantileGate(upper=self.upper, warmup=self.warmup)

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        if self.score_fn is None:
            return float(np.max(np.asarray(probs)))
        return float(self.score_fn(probs, y_human, telemetry))

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        s = self.score(probs, y_human, telemetry)
        return bool(self._gate.decide(s))


# ---------------------------------------------------------------------------
# Robust-loss weighting policies (always admit, but reweight).
# ---------------------------------------------------------------------------


@dataclass
class GCEWeightPolicy(StatefulPolicy):
    """Generalized Cross-Entropy weighting (Zhang & Sabuncu, NeurIPS 2018).

    Loss L_q(p) = (1 - p_y^q) / q with q in (0, 1]. As q -> 0 we recover CE;
    at q = 1 we recover MAE, which is provably noise-tolerant under symmetric
    label noise. We use the *gradient ratio* w(p) = p_y^(q) as the streaming
    sample weight, which is the standard derivation: dL_q/d theta = w(p) *
    dL_CE/d theta. Robust because confident-but-wrong samples get small w.
    """

    name: str = "gce_weight"
    q: float = 0.7

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        p = float(np.clip(np.asarray(probs)[int(y_human)], EPS, 1.0))
        return p ** float(self.q)

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        return True  # always-admit; the *weight* is the robust mechanism

    def weight(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        return self.score(probs, y_human, telemetry)


@dataclass
class SCEWeightPolicy(StatefulPolicy):
    """Symmetric Cross-Entropy (Wang et al., ICCV 2019).

    L_SCE = alpha * CE(p, y) + beta * RCE(p, y) with RCE = -sum_k p_k *
    log q_k where q_k is a clamped one-hot target. We expose the *implicit*
    sample weight that makes the SGDClassifier update equivalent to one step
    of SCE: w_SCE(p) = alpha + beta * (-log p_y * p_y). This is bounded above
    and decays for over-confident wrong predictions.
    """

    name: str = "sce_weight"
    alpha: float = 0.1
    beta: float = 1.0
    clip_log: float = 4.0  # |log q_y| <= clip_log; q_y in [exp(-clip_log), 1]

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        p = float(np.clip(np.asarray(probs)[int(y_human)], EPS, 1.0))
        rce_factor = min(self.clip_log, -math.log(p)) * p
        return float(self.alpha + self.beta * rce_factor)

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        return True

    def weight(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        return self.score(probs, y_human, telemetry)


@dataclass
class ITLMPolicy(StatefulPolicy):
    """Iterative Trimmed Loss Minimization (Shen & Sanghavi, ICML 2019).

    Standard ITLM trains on the smallest-loss alpha-fraction at each epoch.
    The streaming variant we use maintains a P^2 quantile of recent losses
    and admits an event iff its loss is below the alpha-quantile of the
    window. Unlike CoTeaching this is a single-view gate. We default to
    alpha = 0.7 (admit smallest-loss 70%).
    """

    name: str = "itlm"
    alpha: float = 0.7
    warmup: int = 100
    _gate: DynamicQuantileGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # "upper=alpha" admits the top-alpha fraction by negative-loss score
        self._gate = DynamicQuantileGate(upper=self.alpha, warmup=self.warmup)

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        return -_cross_entropy(probs, y_human)

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        return bool(self._gate.decide(self.score(probs, y_human, telemetry)))


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


INTEGRATED_POLICY_NAMES = (
    "co_teaching",
    "self_paced",
    "joint_agreement",
    "dynamic_quantile",
    "gce_weight",
    "sce_weight",
    "itlm",
)


def make_integrated_policies(seed: int = 0) -> list[StatefulPolicy]:
    """Return a default list of integrated baselines for the headline tables."""

    return [
        CoTeachingPolicy(forget_rate=0.3, ramp_steps=500, seed=seed),
        SelfPacedPolicy(lambda_0=1.0, growth=1.02, step=100),
        JointAgreementPolicy(keep_fraction=0.7, seed=seed),
        DynamicQuantilePolicy(upper=0.6, warmup=50),
        GCEWeightPolicy(q=0.7),
        SCEWeightPolicy(alpha=0.1, beta=1.0),
        ITLMPolicy(alpha=0.7, warmup=100),
    ]


# ---------------------------------------------------------------------------
# Optional: in-place registration into rail_paper's policy dispatch.
# ---------------------------------------------------------------------------


def register_integrated_policies_in_rail_paper() -> None:
    """Monkey-patch :mod:`experiments.rail_paper` to dispatch integrated names.

    This keeps the public ``replay_once`` signature untouched while extending
    the policy dispatcher so the integrated baselines can be passed alongside
    the existing ``PolicyConfig`` objects.
    """
    try:
        from . import rail_paper as rp
    except ImportError:
        import rail_paper as rp  # type: ignore[no-redef]

    # Build a name -> StatefulPolicy registry per-call (avoids global mutable
    # state shared between unrelated runs).
    if hasattr(rp, "_integrated_registry"):
        return

    registry: dict[str, StatefulPolicy] = {}

    original_compute = rp.compute_policy_score
    original_admit = rp.policy_admits
    original_weight = rp.policy_weight

    def install(policy: StatefulPolicy) -> None:
        registry[policy.name.lower()] = policy

    def reset() -> None:
        registry.clear()

    def compute_policy_score(policy, probs, y_human, telemetry):  # type: ignore[no-redef]
        name = policy.name.lower()
        if name in registry:
            return registry[name].score(probs, y_human, telemetry)
        return original_compute(policy, probs, y_human, telemetry)

    def policy_admits(policy, score):  # type: ignore[no-redef]
        name = policy.name.lower()
        if name in registry:
            # score was already computed via compute_policy_score; admission
            # is consulted on the underlying state, so we look at the gate
            # decision recomputed via a sentinel call: this requires careful
            # design -- see replay loop note below.
            raise RuntimeError(
                "Integrated policies must be driven by replay_with_integrated; "
                "policy_admits is consulted only for legacy policies."
            )
        return original_admit(policy, score)

    def policy_weight(policy, score):  # type: ignore[no-redef]
        name = policy.name.lower()
        if name in registry:
            raise RuntimeError("Integrated policies must be driven by replay_with_integrated.")
        return original_weight(policy, score)

    rp._integrated_registry = registry  # type: ignore[attr-defined]
    rp._integrated_install = install  # type: ignore[attr-defined]
    rp._integrated_reset = reset  # type: ignore[attr-defined]
    rp.compute_policy_score = compute_policy_score  # type: ignore[assignment]
    rp.policy_admits = policy_admits  # type: ignore[assignment]
    rp.policy_weight = policy_weight  # type: ignore[assignment]
