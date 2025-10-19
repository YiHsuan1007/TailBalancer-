"""Runtime control helpers for percentile clipping."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from torch.utils.data import DataLoader

from cobra_test.integration.hooks import attach_percentile_hooks, remove_handles
from cobra_test.quantize.calibrate import calibrate_model, load_stats
from cobra_test.quantize.config import QuantConfig
from cobra_test.quantize.observers import PercentileObserver

_HANDLE_ATTR = "_quant_pct_handles"
_OBSERVER_ATTR = "_quant_pct_observers"

ActivationCallback = Callable[[str, str, object], None]


def calibrate(model, dataloader: DataLoader, cfg: QuantConfig) -> dict:
    """Run calibration and persist observer statistics."""
    return calibrate_model(model, dataloader, cfg)


def _build_observer(state: dict, cfg: QuantConfig) -> PercentileObserver:
    observer = PercentileObserver(cfg.p_max, cfg.mode, cfg.max_samples)
    observer.load_state_dict(state)
    return observer


def _collect_observers(
    stats: dict,
    cfg: QuantConfig,
) -> Tuple[PercentileObserver, PercentileObserver, PercentileObserver]:
    observers = stats.get("observers", {})
    dino_state = observers.get("dino")
    siglip_state = observers.get("siglip")
    fused_state = observers.get("fused")
    if not (dino_state and siglip_state and fused_state):
        raise KeyError("Calibration statistics are missing required observer entries.")

    return (
        _build_observer(dino_state, cfg),
        _build_observer(siglip_state, cfg),
        _build_observer(fused_state, cfg),
    )


def enable(
    model,
    cfg: QuantConfig,
    *,
    mode: str = "apply",
    dumper: Optional[ActivationCallback] = None,
) -> None:
    """Enable percentile clipping by registering forward hooks on ``model``.

    Parameters
    ----------
    model:
        Cobra model instance.
    cfg:
        Percentile configuration describing calibration statistics.
    mode:
        ``"apply"`` clamps activations using saved thresholds.
        ``"collect"`` only updates observers without clamping.
        ``"off"`` removes any existing percentile hooks.
    dumper:
        Optional callback invoked with ``(tag, phase, tensor)`` while hooks fire.
    """

    normalized_mode = mode.lower()
    if normalized_mode not in {"apply", "collect", "off"}:
        raise ValueError("`mode` must be one of {\"apply\", \"collect\", \"off\"}.")

    if normalized_mode == "off":
        disable(model)
        return

    # Always start from a clean state before attaching new hooks.
    disable(model)

    if normalized_mode == "collect":
        observer_dino = PercentileObserver(cfg.p_max, cfg.mode, cfg.max_samples)
        observer_siglip = PercentileObserver(cfg.p_max, cfg.mode, cfg.max_samples)
        observer_fused = PercentileObserver(cfg.p_max, cfg.mode, cfg.max_samples)
        handles = attach_percentile_hooks(
            model,
            observer_dino=observer_dino,
            observer_siglip=observer_siglip,
            observer_fused=observer_fused,
            apply_clipping=False,
            dumper=dumper,
        )
    else:  # apply
        stats = load_stats(cfg.stats_path)
        cfg_from_stats = stats.get("config", {})
        cfg.p_max = cfg_from_stats.get("p_max", cfg.p_max)
        cfg.mode = cfg_from_stats.get("mode", cfg.mode)
        cfg.max_samples = cfg_from_stats.get("max_samples", cfg.max_samples)

        observer_dino, observer_siglip, observer_fused = _collect_observers(stats, cfg)
        handles = attach_percentile_hooks(
            model,
            observer_dino=observer_dino,
            observer_siglip=observer_siglip,
            observer_fused=observer_fused,
            apply_clipping=True,
            dumper=dumper,
        )

    setattr(model, _HANDLE_ATTR, handles)
    setattr(model, _OBSERVER_ATTR, (observer_dino, observer_siglip, observer_fused))


def disable(model) -> None:
    """Disable percentile clipping hooks if they are registered."""
    handles = getattr(model, _HANDLE_ATTR, None)
    if handles is not None:
        remove_handles(handles)
        delattr(model, _HANDLE_ATTR)
    if hasattr(model, _OBSERVER_ATTR):
        delattr(model, _OBSERVER_ATTR)
