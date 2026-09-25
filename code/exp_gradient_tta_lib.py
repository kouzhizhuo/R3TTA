"""Track A library: neural slow models + faithful gradient-based TTA baselines.

Everything here operates on the shared protocol module ``common_protocol`` (cp).
Nothing in this file re-derives a panel, a split, or a bootstrap.

Design notes that matter for faithfulness
-----------------------------------------
* The neural slow models emit **2 logits** (not 1), so that softmax-entropy, the
  loss every one of Tent / EATA / SAR / CoTTA / RoID is defined on, is used in
  exactly its published C-class form with C = 2.
* Two backbones are provided so the norm-affine methods are well defined in both
  of their native homes:
    - ``mlp_bn``  : MLP with BatchNorm1d  (Tent's original setting)
    - ``ft_trans``: FT-Transformer-style encoder with LayerNorm (SAR's setting)
* The deployment stream is replayed **bar by bar** in chronological order.  A bar
  is the TTA "batch".  Predictions for bar t are emitted by the model state that
  exists when bar t arrives, exactly as in the official Tent/EATA/SAR/CoTTA
  implementations (`forward_and_adapt` returns the pre-step outputs).  No label
  from bar >= t can influence the prediction at bar t.
"""

from __future__ import annotations

import copy
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# backbones
# --------------------------------------------------------------------------- #

class MLPNorm(nn.Module):
    """MLP with an explicit normalisation layer per block.

    ``norm='batch'`` gives BatchNorm1d -> the affine parameters Tent/EATA/SAR
    were designed to adapt, and batch statistics that Tent recomputes at test
    time.  ``norm='layer'`` gives LayerNorm.
    """

    def __init__(self, d_in: int, width: int = 64, depth: int = 2,
                 norm: str = "batch", p_drop: float = 0.0):
        super().__init__()
        blocks = []
        d = d_in
        for _ in range(depth):
            blocks.append(nn.Linear(d, width))
            blocks.append(nn.BatchNorm1d(width) if norm == "batch" else nn.LayerNorm(width))
            blocks.append(nn.ReLU())
            if p_drop > 0:
                blocks.append(nn.Dropout(p_drop))
            d = width
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        return self.head(self.body(x))


class FTTransformer(nn.Module):
    """Small feature-tokenizer Transformer over tabular features.

    Each scalar feature becomes a token (FT-Transformer style), a CLS token is
    prepended, and a pre-norm TransformerEncoder is applied.  LayerNorm is
    everywhere, so norm-affine TTA is well defined without any batch statistics.
    """

    def __init__(self, d_in: int, d_model: int = 32, nhead: int = 4,
                 nlayers: int = 2, p_drop: float = 0.0):
        super().__init__()
        self.feat_w = nn.Parameter(torch.randn(d_in, d_model) * 0.05)
        self.feat_b = nn.Parameter(torch.zeros(d_in, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.05)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=p_drop, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.feat_w + self.feat_b          # (B, F, D)
        tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
        h = self.enc(tok)
        return self.head(self.norm(h[:, 0]))


def build_backbone(kind: str, d_in: int, **kw) -> nn.Module:
    if kind == "mlp_bn":
        return MLPNorm(d_in, width=kw.get("width", 64), depth=kw.get("depth", 2),
                       norm="batch", p_drop=kw.get("p_drop", 0.0))
    if kind == "mlp_ln":
        return MLPNorm(d_in, width=kw.get("width", 64), depth=kw.get("depth", 2),
                       norm="layer", p_drop=kw.get("p_drop", 0.0))
    if kind == "ft_trans":
        return FTTransformer(d_in, d_model=kw.get("d_model", 32),
                             nhead=kw.get("nhead", 4), nlayers=kw.get("nlayers", 2),
                             p_drop=kw.get("p_drop", 0.0))
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# standardisation + training
# --------------------------------------------------------------------------- #

@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean, std)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def frame_matrix(frame, features) -> np.ndarray:
    x = frame[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float64)


def train_backbone(kind, x_fit, y_fit, *, seed=0, epochs=60, lr=1e-3,
                   weight_decay=1e-4, batch_size=256, arch=None) -> nn.Module:
    """Train one frozen neural slow model.  Deterministic given ``seed``."""
    arch = arch or {}
    set_determinism(seed)
    model = build_backbone(kind, x_fit.shape[1], **arch)
    xt = torch.tensor(x_fit, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(xt)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:           # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_frozen(model: nn.Module, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.tensor(x, dtype=torch.float32)
    for s in range(0, len(xt), batch_size):
        out.append(F.softmax(model(xt[s:s + batch_size]), dim=1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------------------- #
# shared TTA plumbing
# --------------------------------------------------------------------------- #

NORM_TYPES = (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """-sum p log p, the loss used by Tent / EATA / SAR / RoID."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def configure_norm_only(model: nn.Module, reset_bn_stats: bool = True) -> nn.Module:
    """Official Tent `configure_model`: train mode, grads only on norm affines.

    For BatchNorm the running statistics are discarded so the layer uses the
    *test batch* statistics (this is a defining part of Tent).  Dropout modules
    are forced to eval for every method uniformly - see REPORT.md, 'Deviations'.
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            if reset_bn_stats:
                # keep the source statistics so a batch too small for batch
                # statistics can still be scored (see ``eval_forward``)
                m._src_running_mean = (None if m.running_mean is None
                                       else m.running_mean.detach().clone())
                m._src_running_var = (None if m.running_var is None
                                      else m.running_var.detach().clone())
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
        elif isinstance(m, nn.Dropout):
            m.eval()
    return model


def eval_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Score ``x`` with the SOURCE normalisation statistics.

    Tent-style configuration sets ``running_mean = running_var = None``, which
    makes PyTorch use batch statistics *even in eval mode* - and BatchNorm1d
    then raises on a 1-row batch.  Our bars carry 3.5 rows on average and many
    carry one, so this path is hit constantly.  Here the saved source statistics
    are reinstated for the duration of one forward pass, which is the only
    well-defined behaviour for a singleton batch and is applied identically to
    every method.
    """
    touched = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) and getattr(m, "_src_running_mean", None) is not None:
            m.running_mean = m._src_running_mean
            m.running_var = m._src_running_var
            m.track_running_stats = True
            touched.append(m)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(x)
    if was_training:
        model.train()
        for mm in model.modules():
            if isinstance(mm, nn.Dropout):
                mm.eval()
    for m in touched:
        m.running_mean = None
        m.running_var = None
        m.track_running_stats = False
    return out


def collect_norm_params(model: nn.Module):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, NORM_TYPES):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias") and p is not None:
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


def make_optimizer(params, name: str, lr: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(name)


@dataclass
class Cost:
    """Test-time cost accounting, per method, over the whole stream."""
    forward: int = 0
    backward: int = 0
    opt_steps: int = 0
    wall: float = 0.0
    n_bars: int = 0
    n_rows: int = 0
    adapted_params: int = 0
    extra: dict = field(default_factory=dict)


class BaseTTA:
    """Interface: ``__call__(x)`` returns probabilities for the bar and adapts."""
    name = "base"
    uses_labels = False

    def __init__(self, model: nn.Module):
        self.model = model
        self.cost = Cost()

    def n_adapted(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def observe_labels(self, x_mat: torch.Tensor, y_vec: torch.Tensor) -> None:
        """Delivered when labels mature (used only by label-aware methods)."""
        return None

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


class Frozen(BaseTTA):
    name = "frozen"

    def __init__(self, model):
        super().__init__(model)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x):
        self.cost.forward += 1
        return F.softmax(self.model(x), dim=1)[:, 1].numpy()


class BNStatsOnly(BaseTTA):
    """'BN-adapt' / test-time norm: recompute BatchNorm stats on the test batch,
    no gradient at all.  The classic zero-backward reference point."""
    name = "bn_stats"

    def __init__(self, model):
        super().__init__(model)
        configure_norm_only(self.model, reset_bn_stats=True)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x):
        self.cost.forward += 1
        if not _bn_safe(self.model, x):        # BN batch stats need >1 sample
            self.cost.extra["n_singleton_bars"] = self.cost.extra.get("n_singleton_bars", 0) + 1
            out = eval_forward(self.model, x)
        else:
            out = self.model(x)
        return F.softmax(out, dim=1)[:, 1].numpy()


# --------------------------------------------------------------------------- #
# a small helper so BatchNorm never sees a 1-row batch in train mode
# --------------------------------------------------------------------------- #

def _bn_safe(model: nn.Module, x: torch.Tensor):
    """Return True if a train-mode forward is safe for this batch."""
    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    return (not has_bn) or x.size(0) >= 2

# --------------------------------------------------------------------------- #
# 1. Tent  (Wang et al., ICLR 2021)  -- FAITHFUL PORT
# --------------------------------------------------------------------------- #

class Tent(BaseTTA):
    """Entropy minimisation on normalisation-layer affine parameters.

    Faithful to the official `tent.py`:
      * `configure_model`: train mode, grads only on norm affines, BN running
        stats discarded so batch statistics are used.
      * `forward_and_adapt`: forward -> mean softmax entropy -> backward -> step,
        returning the outputs computed *before* the step.
      * `episodic=False`: state persists across the stream.
    """
    name = "tent"

    def __init__(self, model, lr=1e-3, steps=1, optimizer="adam"):
        super().__init__(model)
        configure_norm_only(self.model, reset_bn_stats=True)
        params, _ = collect_norm_params(self.model)
        self.params = params
        self.opt = make_optimizer(params, optimizer, lr)
        self.steps = steps
        self.cost.adapted_params = sum(p.numel() for p in params)

    def __call__(self, x):
        if not _bn_safe(self.model, x):
            self.cost.forward += 1
            self.cost.extra["n_singleton_bars"] = self.cost.extra.get("n_singleton_bars", 0) + 1
            out = eval_forward(self.model, x)
            return F.softmax(out, dim=1)[:, 1].detach().numpy()
        for _ in range(self.steps):
            out = self.model(x)
            self.cost.forward += 1
            loss = softmax_entropy(out).mean(0)
            loss.backward()
            self.cost.backward += 1
            self.opt.step()
            self.opt.zero_grad()
            self.cost.opt_steps += 1
        return F.softmax(out, dim=1)[:, 1].detach().numpy()


# --------------------------------------------------------------------------- #
# 2. EATA  (Niu et al., ICML 2022)  -- FAITHFUL PORT
# --------------------------------------------------------------------------- #

class EATA(BaseTTA):
    """Efficient anti-forgetting test-time adaptation.

    Faithful to the official `eata.py`:
      * sample-efficiency filter  : entropy < e_margin (= 0.4 * ln C)
      * redundancy filter         : |cos(p_i, moving-average p)| < d_margin
      * reweighting               : coeff = 1 / exp(entropy - e_margin)
      * Fisher anti-forgetting    : + fisher_alpha * sum F_i (theta_i - theta_i^0)^2
    The Fisher matrix is estimated on pre-deployment (validation) rows with the
    model's own predicted labels, exactly as the official code does on its
    `fisher_dataset`.
    """
    name = "eata"

    def __init__(self, model, lr=1e-3, steps=1, optimizer="adam",
                 e_margin=None, d_margin=0.05, fisher_alpha=2000.0,
                 fishers=None):
        super().__init__(model)
        configure_norm_only(self.model, reset_bn_stats=True)
        params, names = collect_norm_params(self.model)
        self.params, self.pnames = params, names
        self.opt = make_optimizer(params, optimizer, lr)
        self.steps = steps
        self.e_margin = e_margin if e_margin is not None else 0.4 * math.log(2)
        self.d_margin = d_margin
        self.fisher_alpha = fisher_alpha
        self.fishers = fishers            # {name: (fisher, theta0)}
        self.current_model_probs = None
        self.num_samples_update = 0
        self.cost.adapted_params = sum(p.numel() for p in params)

    @staticmethod
    def update_model_probs(current, new_probs):
        if current is None:
            if new_probs.size(0) == 0:
                return None
            return new_probs.mean(0)
        if new_probs.size(0) == 0:
            return current
        return 0.9 * current + 0.1 * new_probs.mean(0)

    def __call__(self, x):
        if not _bn_safe(self.model, x):
            self.cost.forward += 1
            self.cost.extra["n_singleton_bars"] = self.cost.extra.get("n_singleton_bars", 0) + 1
            out = eval_forward(self.model, x)
            return F.softmax(out, dim=1)[:, 1].detach().numpy()
        for _ in range(self.steps):
            out = self.model(x)
            self.cost.forward += 1
            entropys = softmax_entropy(out)
            filter_1 = torch.where(entropys < self.e_margin)[0]
            entropys = entropys[filter_1]
            probs_1 = out[filter_1].softmax(1)
            if self.current_model_probs is not None and probs_1.size(0) > 0:
                cos = F.cosine_similarity(
                    self.current_model_probs.unsqueeze(0), probs_1, dim=1)
                filter_2 = torch.where(cos.abs() < self.d_margin)[0]
                entropys = entropys[filter_2]
                updated = self.update_model_probs(
                    self.current_model_probs, probs_1[filter_2])
            else:
                updated = self.update_model_probs(self.current_model_probs, probs_1)
            if entropys.size(0) > 0:
                coeff = 1.0 / torch.exp(entropys.clone().detach() - self.e_margin)
                loss = (entropys * coeff).mean(0)
            else:
                loss = None
            if self.fishers is not None:
                ewc = torch.tensor(0.0)
                for nm, p in self.model.named_parameters():
                    if nm in self.fishers:
                        f, t0 = self.fishers[nm]
                        ewc = ewc + (f * (p - t0) ** 2).sum()
                ewc = self.fisher_alpha * ewc
                loss = ewc if loss is None else loss + ewc
            if loss is not None and loss.requires_grad:
                loss.backward()
                self.cost.backward += 1
                self.opt.step()
                self.cost.opt_steps += 1
            self.opt.zero_grad()
            self.current_model_probs = updated
            self.num_samples_update += int(entropys.size(0))
        self.cost.extra["n_samples_used"] = self.num_samples_update
        return F.softmax(out, dim=1)[:, 1].detach().numpy()


def compute_fishers(model, x_fisher, *, lr_dummy=1e-3, batch_size=64):
    """Official EATA Fisher estimation: self-labelled cross-entropy gradients
    squared, averaged over a pre-deployment sample."""
    model = copy.deepcopy(model)
    configure_norm_only(model, reset_bn_stats=True)
    params, names = collect_norm_params(model)
    name_set = set(names)
    fishers = {}
    xt = torch.tensor(x_fisher, dtype=torch.float32)
    nb = 0
    for s in range(0, len(xt), batch_size):
        xb = xt[s:s + batch_size]
        if xb.size(0) < 2:
            continue
        model.zero_grad()
        out = model(xb)
        targets = out.argmax(1).detach()
        F.cross_entropy(out, targets).backward()
        for nm, p in model.named_parameters():
            if nm in name_set and p.grad is not None:
                g2 = p.grad.detach().clone() ** 2
                if nm in fishers:
                    fishers[nm] = (fishers[nm][0] + g2, fishers[nm][1])
                else:
                    fishers[nm] = (g2, p.detach().clone())
        nb += 1
    model.zero_grad()
    if nb > 0:
        fishers = {k: (v[0] / nb, v[1]) for k, v in fishers.items()}
    return fishers


# --------------------------------------------------------------------------- #
# 3. SAR  (Niu et al., ICLR 2023)  -- FAITHFUL PORT
# --------------------------------------------------------------------------- #

class SAM(torch.optim.Optimizer):
    """Sharpness-aware minimisation wrapper, as shipped with SAR."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kw):
        defaults = dict(rho=rho, adaptive=adaptive, **kw)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kw)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)


class SAR(BaseTTA):
    """Sharpness-aware and reliable entropy minimisation.

    Faithful to the official `sar.py`:
      * reliable-sample filter  : entropy < margin_e0 (= 0.4 * ln C)
      * SAM two-step update on norm affines
      * second filter after the ascent step
      * model recovery: reset to the source state when the moving average of the
        second-step loss falls below `reset_constant` (0.2), signalling collapse.
    """
    name = "sar"

    def __init__(self, model, lr=2.5e-4, steps=1, rho=0.05,
                 margin_e0=None, reset_constant=0.2):
        super().__init__(model)
        configure_norm_only(self.model, reset_bn_stats=True)
        params, _ = collect_norm_params(self.model)
        self.params = params
        self.opt = SAM(params, torch.optim.SGD, rho=rho, lr=lr, momentum=0.9)
        self.steps = steps
        self.margin_e0 = margin_e0 if margin_e0 is not None else 0.4 * math.log(2)
        self.reset_constant = reset_constant
        self.ema = None
        self.n_resets = 0
        self.source_state = copy.deepcopy(self.model.state_dict())
        self.opt_state = copy.deepcopy(self.opt.state_dict())
        self.cost.adapted_params = sum(p.numel() for p in params)

    def _reset(self):
        self.model.load_state_dict(self.source_state, strict=True)
        self.opt.load_state_dict(self.opt_state)
        self.ema = None
        self.n_resets += 1

    def __call__(self, x):
        if not _bn_safe(self.model, x):
            self.cost.forward += 1
            self.cost.extra["n_singleton_bars"] = self.cost.extra.get("n_singleton_bars", 0) + 1
            out = eval_forward(self.model, x)
            return F.softmax(out, dim=1)[:, 1].detach().numpy()
        for _ in range(self.steps):
            self.opt.zero_grad()
            out = self.model(x)
            self.cost.forward += 1
            entropys = softmax_entropy(out)
            filter_1 = torch.where(entropys < self.margin_e0)[0]
            if filter_1.numel() == 0:
                self.opt.zero_grad()
                break
            loss = entropys[filter_1].mean(0)
            loss.backward()
            self.cost.backward += 1
            self.opt.first_step(zero_grad=True)

            out2 = self.model(x)
            self.cost.forward += 1
            entropys2 = softmax_entropy(out2)[filter_1]
            filter_2 = torch.where(entropys2 < self.margin_e0)[0]
            if filter_2.numel() == 0:
                self.opt.zero_grad()
                break
            loss2 = entropys2[filter_2].mean(0)
            loss2.backward()
            self.cost.backward += 1
            self.opt.second_step(zero_grad=True)
            self.cost.opt_steps += 1

            l2 = loss2.item()
            if not math.isnan(l2):
                self.ema = l2 if self.ema is None else 0.9 * self.ema + 0.1 * l2
            if self.ema is not None and self.ema < self.reset_constant:
                self._reset()
        self.cost.extra["n_resets"] = self.n_resets
        return F.softmax(out, dim=1)[:, 1].detach().numpy()


# --------------------------------------------------------------------------- #
# 4. CoTTA  (Wang et al., CVPR 2022)  -- ADAPTED PORT (augmentation family)
# --------------------------------------------------------------------------- #

class CoTTA(BaseTTA):
    """Continual test-time adaptation: weight-averaged teacher + augmentation
    averaging + stochastic restore.  All parameters are adapted.

    FAITHFUL: teacher EMA (mt=0.999), the augmentation-averaging trigger on the
    anchor model's mean confidence (ap=0.92), the student/teacher cross-entropy
    loss, and stochastic restore (rst=0.01) toward the source weights.

    ADAPTED: the augmentation family.  CoTTA's image augmentations do not exist
    for a 12-dimensional tabular feature vector, so augmentation is Gaussian
    jitter on the standardised features, x' = x + sigma*eps.  ``sigma`` and the
    number of augmentations are swept on validation.  This substitution is
    declared in REPORT.md.
    """
    name = "cotta"

    def __init__(self, model, lr=1e-3, steps=1, mt=0.999, rst=0.01, ap=0.92,
                 n_aug=32, aug_sigma=0.1, optimizer="adam"):
        super().__init__(model)
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        self.model.requires_grad_(True)
        self.model_ema = copy.deepcopy(self.model)
        self.model_ema.requires_grad_(False)
        self.model_anchor = copy.deepcopy(self.model)
        self.model_anchor.requires_grad_(False)
        self.model_anchor.eval()
        self.source_state = {k: v.detach().clone()
                             for k, v in self.model.state_dict().items()}
        self.opt = make_optimizer(
            [p for p in self.model.parameters() if p.requires_grad], optimizer, lr)
        self.steps, self.mt, self.rst, self.ap = steps, mt, rst, ap
        self.n_aug, self.aug_sigma = n_aug, aug_sigma
        self.gen = torch.Generator().manual_seed(1234)
        self.n_aug_triggered = 0
        self.cost.adapted_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

    def _aug(self, x):
        return x + self.aug_sigma * torch.randn(
            x.shape, generator=self.gen, dtype=x.dtype)

    def __call__(self, x):
        safe = _bn_safe(self.model, x)
        for _ in range(self.steps):
            if not safe:
                break
            out = self.model(x)
            self.cost.forward += 1
            with torch.no_grad():
                anchor_prob = F.softmax(self.model_anchor(x), dim=1).max(1)[0]
                self.cost.forward += 1
                if anchor_prob.mean(0).item() < self.ap:
                    self.n_aug_triggered += 1
                    acc = torch.zeros_like(out)
                    for _a in range(self.n_aug):
                        acc = acc + F.softmax(self.model_ema(self._aug(x)), dim=1)
                        self.cost.forward += 1
                    ema_prob = acc / self.n_aug
                else:
                    ema_prob = F.softmax(self.model_ema(x), dim=1)
                    self.cost.forward += 1
            loss = -(ema_prob * out.log_softmax(1)).sum(1).mean(0)
            loss.backward()
            self.cost.backward += 1
            self.opt.step()
            self.opt.zero_grad()
            self.cost.opt_steps += 1
            # teacher EMA
            with torch.no_grad():
                for pe, ps in zip(self.model_ema.parameters(), self.model.parameters()):
                    pe.mul_(self.mt).add_(ps.detach(), alpha=1.0 - self.mt)
                for be, bs in zip(self.model_ema.buffers(), self.model.buffers()):
                    be.copy_(bs)
            # stochastic restore
            with torch.no_grad():
                for nm, p in self.model.named_parameters():
                    if nm in self.source_state:
                        mask = (torch.rand(p.shape, generator=self.gen) < self.rst).float()
                        p.mul_(1.0 - mask).add_(self.source_state[nm] * mask)
        # CoTTA's deployed prediction is the teacher's
        if safe:
            with torch.no_grad():
                pred = F.softmax(self.model_ema(x), dim=1)[:, 1]
        else:
            self.cost.extra["n_singleton_bars"] = self.cost.extra.get("n_singleton_bars", 0) + 1
            pred = F.softmax(eval_forward(self.model_ema, x), dim=1)[:, 1]
        self.cost.forward += 1
        self.cost.extra["n_aug_triggered"] = self.n_aug_triggered
        return pred.numpy()

# --------------------------------------------------------------------------- #
# 5. ROID  (Marsden, Doebler, Yang; WACV 2024)  -- FAITHFUL PORT (aug adapted)
# --------------------------------------------------------------------------- #

def soft_likelihood_ratio(logits, clip=0.99, eps=1e-5):
    """ROID's base loss (official `SoftLikelihoodRatio`)."""
    probs = logits.softmax(1).clamp(0.0, clip)
    return -(probs * torch.log(probs / (1.0 - probs) + eps)).sum(1)


def symmetric_cross_entropy(x, x_ema, alpha=0.5):
    """Official ROID `symmetric_cross_entropy`; neither branch detached."""
    return -(1 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) \
           - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def _minmax(v):
    lo, hi = v.min(), v.max()
    if (hi - lo) < 1e-12:
        return torch.zeros_like(v)
    return (v - lo) / (hi - lo)


class ROID(BaseTTA):
    """Universal TTA: weight ensembling + diversity weighting + prior correction.

    FAITHFUL: the clipped soft-likelihood-ratio loss (clip 0.99), the min-max
    normalised diversity weight (1 - cos to the EMA of predictions, beta=0.9)
    and certainty weight (-entropy), their multiplicative combination with
    temperature tau=1/3, the hard diversity mask at the batch mean, the
    symmetric-cross-entropy consistency term on an augmented view, weight
    ensembling theta <- 0.99*theta + 0.01*theta_source applied to the adapted
    (norm-affine) parameters after every step, and the adaptively smoothed prior
    correction applied to the emitted prediction only.

    ADAPTED: the augmentation used by the consistency term.  ROID uses colour
    jitter / random affine / horizontal flip, none of which exist for a
    12-dimensional tabular vector; Gaussian jitter on standardised features is
    substituted, with sigma swept on validation.  Declared in REPORT.md.
    """
    name = "roid"

    def __init__(self, model, lr=1e-3, steps=1, optimizer="adam",
                 momentum_src=0.99, momentum_probs=0.9, temperature=1.0 / 3.0,
                 use_weighting=True, use_prior_correction=True,
                 use_consistency=True, aug_sigma=0.1,
                 prior_mode="logits"):
        super().__init__(model)
        configure_norm_only(self.model, reset_bn_stats=True)
        params, names = collect_norm_params(self.model)
        self.params, self.pnames = params, names
        self.opt = make_optimizer(params, optimizer, lr)
        self.steps = steps
        self.momentum_src = momentum_src
        self.momentum_probs = momentum_probs
        self.temperature = temperature
        self.use_weighting = use_weighting
        self.use_prior_correction = use_prior_correction
        self.use_consistency = use_consistency
        self.aug_sigma = aug_sigma
        self.prior_mode = prior_mode
        self.n_classes = 2
        self.class_probs_ema = torch.full((2,), 1.0 / 2.0)   # uniform init
        self.src_params = [p.detach().clone() for p in params]
        self.gen = torch.Generator().manual_seed(4321)
        self.cost.adapted_params = sum(p.numel() for p in params)

    def _aug(self, x):
        return x + self.aug_sigma * torch.randn(
            x.shape, generator=self.gen, dtype=x.dtype)

    def _emit(self, logits):
        if not self.use_prior_correction:
            return F.softmax(logits, dim=1)[:, 1].detach().numpy()
        prior = logits.softmax(1).mean(0)
        smooth = max(1.0 / logits.shape[0], 1.0 / self.n_classes) / torch.max(prior)
        sm_prior = (prior + smooth) / (1.0 + smooth * self.n_classes)
        if self.prior_mode == "logits":            # released-code behaviour
            out = logits * sm_prior
            return F.softmax(out, dim=1)[:, 1].detach().numpy()
        post = logits.softmax(1) * sm_prior        # paper derivation
        post = post / post.sum(1, keepdim=True)
        return post[:, 1].detach().numpy()

    def __call__(self, x):
        if not _bn_safe(self.model, x):
            self.cost.forward += 1
            self.cost.extra["n_singleton_bars"] = self.cost.extra.get("n_singleton_bars", 0) + 1
            out = eval_forward(self.model, x)
            return self._emit(out)
        for _ in range(self.steps):
            outputs = self.model(x)
            self.cost.forward += 1
            if self.use_weighting:
                with torch.no_grad():
                    probs = outputs.softmax(1)
                    w_div = 1.0 - F.cosine_similarity(
                        self.class_probs_ema.unsqueeze(0), probs, dim=1)
                    w_div = _minmax(w_div)
                    mask = w_div < w_div.mean()
                    ent = -(probs * torch.log(probs + 1e-12)).sum(1)
                    w_cert = _minmax(-ent)
                    w = torch.exp(w_div * w_cert / self.temperature)
                    w[mask] = 0.0
                    self.class_probs_ema = (
                        self.momentum_probs * self.class_probs_ema
                        + (1 - self.momentum_probs) * probs.mean(0))
            else:
                w = torch.ones(x.size(0))
                mask = torch.zeros(x.size(0), dtype=torch.bool)
            nb = x.size(0)
            keep = ~mask
            if keep.sum() == 0:
                self.opt.zero_grad()
                break
            loss = (soft_likelihood_ratio(outputs) * w)[keep].sum() / nb
            if self.use_consistency and keep.sum() > 0:
                x_aug = self._aug(x[keep])
                if _bn_safe(self.model, x_aug):
                    out_aug = self.model(x_aug)
                    self.cost.forward += 1
                    loss = loss + (symmetric_cross_entropy(out_aug, outputs[keep])
                                   * w[keep]).sum() / nb
            loss.backward()
            self.cost.backward += 1
            self.opt.step()
            self.opt.zero_grad()
            self.cost.opt_steps += 1
            # weight ensembling toward the frozen source weights
            with torch.no_grad():
                for p, p0 in zip(self.params, self.src_params):
                    p.mul_(self.momentum_src).add_(p0, alpha=1.0 - self.momentum_src)
        return self._emit(outputs)


# --------------------------------------------------------------------------- #
# 6. TAFAS  (Kim, Kim, Mok, Yoon; AAAI 2025, arXiv:2501.04970)
#    -- ADAPTED PORT.  See REPORT.md 'Faithfulness' for the full account.
# --------------------------------------------------------------------------- #

class GCM(nn.Module):
    """Gated Calibration Module, TAFAS Eq. 3.

    ``out = x + tanh(alpha) * (W x + b)`` with ``W`` and ``b`` zero-initialised
    (so the module is exactly the identity at deployment time) and ``alpha``
    initialised to ``gating_init`` (0.01 in the release).

    TAFAS mixes over the *time* axis of a look-back window with one W per
    variable.  Our event rows are not windows, so the mixing matrix acts over
    the feature axis instead - the same dense residual gated linear map, over
    the axis that exists in this problem.
    """

    def __init__(self, dim: int, gating_init: float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.gating = nn.Parameter(gating_init * torch.ones(dim))

    def forward(self, x):
        return x + torch.tanh(self.gating) * (x @ self.weight + self.bias)


class TAFAS(BaseTTA):
    """Test-time adaptation driven by DELAYED ground truth.

    Kept from TAFAS
    ---------------
    * The backbone is **completely frozen**; the only adapted parameters are
      those of an input GCM and an output GCM (paper Table 5 shows adapting the
      backbone is worse than doing nothing).
    * GCMs are residual, zero-initialised, per-dimension ``tanh`` gated.
    * Adaptation is supervised by **real, matured ground truth** - never by an
      entropy surrogate.  This is the whole point of the method and the reason
      it is the right comparator for our label-delayed setting.
    * Periodicity-aware scheduling (PAAS): the adaptation batch size is derived
      by an FFT on the recent stream, using the released code's
      ``period = L // argmax|FFT|`` (floor, per the release, not the paper's
      ceil), with the release's fallback when the argmax lands on DC.
    * One gradient step per adaptation event (``STEPS=1``), Adam, lr 1e-3,
      weight decay 0, gate init 0.01 - the released ``run.sh`` values.

    Changed, and why
    ----------------
    * **Task.**  TAFAS is multivariate forecasting with an ``L``-step look-back
      and an ``H``-step horizon; here the model is a binary classifier over a
      12-dimensional event feature vector.  The MSE objective is therefore
      replaced by cross-entropy on the matured binary label.
    * **POGT.**  TAFAS exploits the fact that the first ``p`` of ``H`` horizon
      steps are observable before the rest.  Our label is a single scalar that
      matures atomically after ``LABEL_DELAY`` (1 h).  There is no *partial*
      ground truth, only *delayed* ground truth, so ``L_partial`` and ``L_full``
      collapse into one supervised term over whatever has matured by the current
      bar.  This is the one structural component of TAFAS that our setting
      cannot support.
    * **Prediction Adjustment** is dropped: PA rewrites the not-yet-realised
      tail of a multi-step forecast, and a one-step-ahead binary prediction has
      no tail to rewrite.
    * The input GCM mixes over features rather than over look-back time.

    Because it consumes matured labels, TAFAS is the only method here that sees
    **more** information than R3-TTT, which never ingests a deployment outcome.
    """
    name = "tafas"
    uses_labels = True

    def __init__(self, model, d_in, lr=1e-3, steps=1, gating_init=0.01,
                 weight_decay=0.0, use_paas=True, fallback_batch=64,
                 period_n=1, fft_window=64, input_gcm=True, output_gcm=True,
                 min_batch=8):
        super().__init__(model)
        self.model.eval()
        self.model.requires_grad_(False)          # backbone frozen, always
        self.gcm_in = GCM(d_in, gating_init) if input_gcm else None
        self.gcm_out = GCM(2, gating_init) if output_gcm else None
        params = []
        for g in (self.gcm_in, self.gcm_out):
            if g is not None:
                params += list(g.parameters())
        self.opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.steps = steps
        self.use_paas = use_paas
        self.fallback_batch = fallback_batch
        self.period_n = period_n
        self.fft_window = fft_window
        self.min_batch = min_batch
        self.pending_x, self.pending_y = [], []
        self.recent = deque(maxlen=fft_window)
        self.n_adapt_events = 0
        self.periods = []
        self.cost.adapted_params = sum(p.numel() for p in params)

    def _forward(self, x):
        z = self.gcm_in(x) if self.gcm_in is not None else x
        z = self.model(z)
        if self.gcm_out is not None:
            z = self.gcm_out(z)
        return z

    def _period(self):
        """PAAS, following the released `_calculate_period_and_batch_size`."""
        if not self.use_paas or len(self.recent) < 8:
            return self.fallback_batch - 1
        arr = torch.tensor(np.stack(list(self.recent)), dtype=torch.float32)
        arr = arr - arr.mean(dim=0, keepdim=True)
        amp = torch.fft.rfft(arr, dim=0).abs()            # [F_freq, C]
        power = (amp ** 2).mean(dim=0)                    # [C]
        c_star = int(power.argmax())
        f_star = int(amp[:, c_star].argmax())
        L = arr.shape[0]
        period = (L // f_star) if f_star > 0 else 24
        return max(int(period * self.period_n), 1)

    def observe_labels(self, x_mat, y_vec):
        """Matured ground truth arrives; buffer it and adapt when the
        periodicity-aware batch is full."""
        for i in range(x_mat.shape[0]):
            self.pending_x.append(x_mat[i])
            self.pending_y.append(y_vec[i])
            self.recent.append(x_mat[i].numpy())
        target = max(self._period() + 1, self.min_batch)
        while len(self.pending_x) >= target:
            xb = torch.stack(self.pending_x[:target])
            yb = torch.stack(self.pending_y[:target])
            self.pending_x = self.pending_x[target:]
            self.pending_y = self.pending_y[target:]
            self.periods.append(target)
            for _ in range(self.steps):
                self.opt.zero_grad()
                out = self._forward(xb)
                self.cost.forward += 1
                loss = F.cross_entropy(out, yb)
                loss.backward()
                self.cost.backward += 1
                self.opt.step()
                self.cost.opt_steps += 1
            self.n_adapt_events += 1
            target = max(self._period() + 1, self.min_batch)

    @torch.no_grad()
    def __call__(self, x):
        self.cost.forward += 1
        self.cost.extra["n_adapt_events"] = self.n_adapt_events
        self.cost.extra["mean_paas_batch"] = (
            float(np.mean(self.periods)) if self.periods else 0.0)
        return F.softmax(self._forward(x), dim=1)[:, 1].numpy()

# --------------------------------------------------------------------------- #
# streaming harness
# --------------------------------------------------------------------------- #




def run_stream(method: BaseTTA, x_stream: np.ndarray, groups, *,
               buffer_rows: int = 0, bar_times=None, available_times=None,
               y_stream=None, buffer_init: np.ndarray | None = None):
    """Replay the deployment stream bar by bar.

    ``buffer_rows``  : if > 0, the adaptation batch is the current bar's rows
                       followed by up to ``buffer_rows`` most recent *past* rows.
                       Predictions are always read off the current bar's rows
                       only.  Strictly causal.  Declared as a deviation.
    ``available_times``/``y_stream``: enable delayed-label delivery.  Before
                       predicting bar t, every row whose label has matured
                       (available_time <= bar_time[t]) is handed to
                       ``method.observe_labels``.  Only label-aware methods
                       (TAFAS) consume this.
    """
    n = len(x_stream)
    p = np.zeros(n, dtype=float)
    buf = deque(maxlen=max(buffer_rows, 1))
    # Warm-start the buffer with the tail of the PRE-DEPLOYMENT matrix.  At the
    # moment deployment begins these rows are already in hand, so this is
    # causal; without it the first bars of the stream would have to adapt on a
    # 1-3 row batch, which BatchNorm cannot even evaluate.
    if buffer_rows > 0 and buffer_init is not None and len(buffer_init):
        for row in buffer_init[-buffer_rows:]:
            buf.append(row)
    label_aware = (getattr(method, "uses_labels", False)
                   and available_times is not None and bar_times is not None
                   and y_stream is not None)
    if label_aware:
        order = np.argsort(available_times, kind="stable")
    cursor = 0

    method.cost.n_bars = len(groups)
    method.cost.n_rows = n
    t_start = time.perf_counter()

    for gi, locs in enumerate(groups):
        # ---- deliver matured labels (label-aware methods only) --------------
        if label_aware:
            now = bar_times[gi]
            fresh = []
            while cursor < n and available_times[order[cursor]] <= now:
                fresh.append(order[cursor])
                cursor += 1
            if fresh:
                fresh = np.asarray(fresh, dtype=int)
                t0 = time.perf_counter()
                method.observe_labels(
                    torch.tensor(x_stream[fresh], dtype=torch.float32),
                    torch.tensor(y_stream[fresh], dtype=torch.long))
                method.cost.wall += time.perf_counter() - t0

        # ---- assemble the adaptation batch ----------------------------------
        cur = x_stream[locs]
        if buffer_rows > 0 and len(buf) > 0:
            past = np.stack(list(buf)[-buffer_rows:])
            xa = np.concatenate([cur, past], axis=0)
        else:
            xa = cur
        xt = torch.tensor(xa, dtype=torch.float32)

        t0 = time.perf_counter()
        out = method(xt)
        method.cost.wall += time.perf_counter() - t0
        p[locs] = out[:len(locs)]

        if buffer_rows > 0:
            for row in cur:
                buf.append(row)
    method.cost.extra["wall_total_s"] = time.perf_counter() - t_start
    return p


def cost_row(method_name: str, backbone: str, cost: Cost) -> dict:
    nb = max(cost.n_bars, 1)
    return {
        "backbone": backbone,
        "method": method_name,
        "n_bars": cost.n_bars,
        "n_rows": cost.n_rows,
        "forward_passes_total": cost.forward,
        "backward_passes_total": cost.backward,
        "optimizer_steps_total": cost.opt_steps,
        "forward_per_bar": cost.forward / nb,
        "backward_per_bar": cost.backward / nb,
        "adapted_parameters": cost.adapted_params,
        "wall_clock_total_s": round(cost.wall, 4),
        "wall_clock_ms_per_bar": round(1000.0 * cost.wall / nb, 4),
        **{f"extra_{k}": v for k, v in cost.extra.items()},
    }
