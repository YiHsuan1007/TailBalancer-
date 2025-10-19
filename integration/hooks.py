"""Hook registration helpers for percentile-based clipping."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch.utils.hooks import RemovableHandle

from ..quantize import clipping
from ..quantize.observers import PercentileObserver

ActivationCallback = Callable[[str, str, torch.Tensor], None]


def _make_activation_hook(
    observer: PercentileObserver,
    apply_clipping: bool,
    tag: str,
    dumper: Optional[ActivationCallback],
):
    @torch.no_grad()
    def hook(_module, _inputs, output):
        if dumper is not None:
            dumper(tag, "pre", output)

        if apply_clipping:
            clip = observer.get_clip_value()
            if clip is not None:
                clamped = clipping.clamp_tensor_(output, clip)
                if dumper is not None:
                    dumper(tag, "post", clamped)
                return clamped
            if dumper is not None:
                dumper(tag, "post", output)
            return output

        observer.update(output)
        if dumper is not None:
            dumper(tag, "post", output)
        return output

    return hook


def _make_fused_pre_hook(
    observer: PercentileObserver,
    apply_clipping: bool,
    tag: str,
    dumper: Optional[ActivationCallback],
):
    @torch.no_grad()
    def hook(_module, args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            return None

        if dumper is not None:
            dumper(tag, "pre", embeds)

        if apply_clipping:
            clip = observer.get_clip_value()
            if clip is not None:
                kwargs["inputs_embeds"] = clipping.clamp_tensor_(embeds, clip)
                if dumper is not None:
                    dumper(tag, "post", kwargs["inputs_embeds"])
                return None
            if dumper is not None:
                dumper(tag, "post", embeds)
            return None

        observer.update(embeds)
        if dumper is not None:
            dumper(tag, "post", embeds)
        return None

    return hook


def _get_dino_module(model) -> torch.nn.Module:
    vision = getattr(model, "vision_backbone", None)
    if vision is None or not hasattr(vision, "dino_featurizer"):
        raise AttributeError(f"Model does not expose a `dino_featurizer` module for hooking.")
    return vision.dino_featurizer


def _get_siglip_module(model) -> torch.nn.Module:
    vision = getattr(model, "vision_backbone", None)
    if vision is None or not hasattr(vision, "siglip_featurizer"):
        raise AttributeError(f"Model does not expose a `siglip_featurizer` module for hooking.")
    return vision.siglip_featurizer


def _get_fused_module(model) -> torch.nn.Module:
    if not hasattr(model, "llm_backbone"):
        raise AttributeError(f"Model does not expose an `llm_backbone` module for hooking.")
    return model.llm_backbone


def attach_percentile_hooks(
    model,
    *,
    observer_dino: PercentileObserver,
    observer_siglip: PercentileObserver,
    observer_fused: PercentileObserver,
    apply_clipping: bool,
    dumper: Optional[ActivationCallback] = None,
) -> List[RemovableHandle]:
    """Attach percentile observers to the Cobra pipeline.

    Parameters
    ----------
    model:
        Cobra VLM instance.
    observer_dino / observer_siglip / observer_fused:
        Observer objects that either collect statistics or apply pre-computed clipping.
    apply_clipping:
        When ``True`` the observers clamp activations instead of updating statistics.
    dumper:
        Optional callback invoked with ``(tag, phase, tensor)`` to record pre/post activations.
    """

    handles: List[RemovableHandle] = []

    dino_module = _get_dino_module(model)
    siglip_module = _get_siglip_module(model)
    fused_module = _get_fused_module(model)

    handles.append(
        dino_module.register_forward_hook(
            _make_activation_hook(observer_dino, apply_clipping, "dino", dumper)
        )
    )
    handles.append(
        siglip_module.register_forward_hook(
            _make_activation_hook(observer_siglip, apply_clipping, "siglip", dumper)
        )
    )
    handles.append(
        fused_module.register_forward_pre_hook(
            _make_fused_pre_hook(observer_fused, apply_clipping, "fused", dumper),
            with_kwargs=True,
        )
    )

    return handles


def remove_handles(handles: Iterable[RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()
