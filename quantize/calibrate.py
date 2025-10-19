"""Calibration utilities for percentile-based clipping."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .config import QuantConfig
from .observers import PercentileObserver



def _move_to_device(payload, device: torch.device):
    if isinstance(payload, dict):
        return {k: _move_to_device(v, device) for k, v in payload.items()}
    if isinstance(payload, torch.Tensor):
        return payload.to(device)
    raise TypeError(f"Unsupported payload type `{type(payload)}` for device transfer.")


def _ensure_text_batch(model, batch_size: int, cfg: QuantConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    tokenizer = model.llm_backbone.tokenizer
    encoded = tokenizer(
        [cfg.prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {
        "input_ids": encoded.input_ids.to(device),
        "attention_mask": encoded.attention_mask.to(device),
    }


def _extract_text_inputs(batch: Dict, model, cfg: QuantConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    if "input_ids" in batch and batch["input_ids"] is not None:
        inputs = {"input_ids": batch["input_ids"].to(device)}
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            inputs["attention_mask"] = batch["attention_mask"].to(device)
        return inputs
    batch_size = next(iter(batch.values())).shape[0]
    return _ensure_text_batch(model, batch_size, cfg, device)


def calibrate_model(
    model,
    dataloader: DataLoader,
    cfg: QuantConfig,
    observer_factory = PercentileObserver,
) -> Dict[str, Dict]:
    device = torch.device(cfg.device) if cfg.device else next(model.parameters()).device
    model.eval()

    observer_dino = observer_factory(cfg.p_max, cfg.mode, cfg.max_samples)
    observer_siglip = observer_factory(cfg.p_max, cfg.mode, cfg.max_samples)
    observer_fused = observer_factory(cfg.p_max, cfg.mode, cfg.max_samples)

    from cobra_test.integration.hooks import attach_percentile_hooks

    handles = attach_percentile_hooks(
        model,
        observer_dino=observer_dino,
        observer_siglip=observer_siglip,
        observer_fused=observer_fused,
        apply_clipping=False,
    )

    try:
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if cfg.num_batches is not None and step >= cfg.num_batches:
                    break
                if not isinstance(batch, dict):
                    raise TypeError("Calibration dataloader must yield dictionaries.")

                pixel_values = batch.get("pixel_values")
                if pixel_values is None:
                    raise KeyError("Batch is missing `pixel_values`.")
                pixel_values = _move_to_device(pixel_values, device)
                text_inputs = _extract_text_inputs(batch, model, cfg, device)

                model(
                    input_ids=text_inputs.get("input_ids"),
                    attention_mask=text_inputs.get("attention_mask"),
                    pixel_values=pixel_values,
                    use_cache=False,
                )
    finally:
        for handle in handles:
            handle.remove()

    stats = {
        "config": cfg.to_dict(),
        "observers": {
            "dino": observer_dino.state_dict(),
            "siglip": observer_siglip.state_dict(),
            "fused": observer_fused.state_dict(),
        },
    }
    save_stats(stats, cfg.stats_path)
    return stats


def save_stats(stats: Dict, path: Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_path)


def load_stats(path: Path) -> Dict:
    return torch.load(Path(path), map_location="cpu")
