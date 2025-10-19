"""CLI entrypoint to enable percentile clipping and run inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import torch
from PIL import Image

from cobra import load as load_model

from cobra_test.quantize.config import QuantConfig
from cobra_test.switches import quant_pct

_DEFAULT_DUMP_POINTS = {"dino", "siglip"}
_PERCENTILES: Sequence[float] = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)


class ActivationDumper:
    """Utility that stores pre/post activations and emits debug summaries."""

    def __init__(self, targets: Set[str]) -> None:
        self.targets = targets
        self.records: Dict[str, Dict[str, torch.Tensor]] = {}

    def capture(self, name: str, phase: str, tensor: torch.Tensor) -> None:
        if name not in self.targets:
            return
        self.records.setdefault(name, {})[phase] = tensor.detach().cpu()

    def save(
        self,
        out_dir: Path,
        clip_values: Optional[Dict[str, object]] = None,
        mode: Optional[str] = None,
    ) -> None:
        if not self.records:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Dict[str, object]] = {}

        for name, phases in self.records.items():
            pre = phases.get("pre")
            if pre is None:
                continue
            post = phases.get("post") or pre

            torch.save(pre, out_dir / f"{name}_pre.pt")
            torch.save(post, out_dir / f"{name}_post.pt")

            entry: Dict[str, object] = {
                "pre": _tensor_stats(pre),
                "post": _tensor_stats(post),
                "clamped_count": _clamped_count(pre, post),
            }
            if clip_values and name in clip_values:
                entry["clip_value"] = clip_values[name]
            if mode is not None:
                entry["mode"] = mode
            summary[name] = entry

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    if tensor.numel() == 0:
        return {"min": 0.0, "max": 0.0, "percentiles": {}}
    flat = tensor.detach().float().view(-1)
    stats = {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }
    percentiles = torch.quantile(flat, torch.tensor(_PERCENTILES, dtype=torch.float32))
    stats["percentiles"] = {
        f"p{int(p * 1000) / 10:.1f}": float(val)
        for p, val in zip(_PERCENTILES, percentiles)
    }
    return stats


def _clamped_count(pre: torch.Tensor, post: torch.Tensor) -> int:
    if pre.shape != post.shape:
        post = post.view_as(pre)
    diff = ~torch.isclose(post, pre, atol=1e-6, rtol=1e-6)
    return int(diff.sum().item())


def _parse_dump_targets(raw: str) -> Set[str]:
    if not raw:
        return set()
    parts = {part.strip().lower() for part in raw.split(',') if part.strip()}
    if not parts or "all" in parts:
        return {"dino", "siglip", "fused"}
    allowed = {"dino", "siglip", "fused"}
    invalid = parts - allowed
    if invalid:
        raise ValueError("Unknown dump targets: " + ", ".join(sorted(invalid)))
    return parts


def _extract_clip_values(model) -> Dict[str, object]:
    observers = getattr(model, "_quant_pct_observers", None)
    if not observers:
        return {}
    names = ["dino", "siglip", "fused"]
    result: Dict[str, object] = {}
    for name, observer in zip(names, observers):
        clip = observer.get_clip_value()
        if clip is None:
            continue
        if isinstance(clip, torch.Tensor):
            clip_cpu = clip.detach().cpu()
            result[name] = float(clip_cpu.item()) if clip_cpu.numel() == 1 else clip_cpu.tolist()
        else:
            result[name] = float(clip)
    return result


def _register_passthrough_hooks(model, dumper: ActivationDumper, targets: Set[str]):
    handles = []

    def make(tag: str, *, is_pre: bool = False):
        if is_pre:
            @torch.no_grad()
            def pre_hook(_module, _args, kwargs):
                embeds = kwargs.get("inputs_embeds")
                if embeds is None:
                    return None
                dumper.capture(tag, "pre", embeds)
                dumper.capture(tag, "post", embeds)
                return None

            return pre_hook

        @torch.no_grad()
        def hook(_module, _inputs, output):
            dumper.capture(tag, "pre", output)
            dumper.capture(tag, "post", output)
            return output

        return hook

    vision = getattr(model, "vision_backbone", None)
    if vision is not None:
        if "dino" in targets and hasattr(vision, "dino_featurizer"):
            handles.append(vision.dino_featurizer.register_forward_hook(make("dino")))
        if "siglip" in targets and hasattr(vision, "siglip_featurizer"):
            handles.append(vision.siglip_featurizer.register_forward_hook(make("siglip")))
    if "fused" in targets and hasattr(model, "llm_backbone"):
        handles.append(model.llm_backbone.register_forward_pre_hook(make("fused", is_pre=True), with_kwargs=True))
    return handles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enable percentile clipping and run a single inference")
    parser.add_argument("--ckpt", required=True, help="Model identifier or local checkpoint directory.")
    parser.add_argument("--cfg", required=True, help="YAML configuration with percentile statistics path.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", required=True, help="Prompt or question for the model.")
    parser.add_argument("--hf-token", default=None, help="Optional HuggingFace token for gated models.")
    parser.add_argument(
        "--mode",
        default="apply",
        choices=["apply", "collect", "off"],
        help="Select clipping behaviour: apply thresholds, collect stats, or run unclipped.",
    )
    parser.add_argument(
        "--dump-activations",
        action="store_true",
        help="Dump pre/post activations and emit a JSON summary.",
    )
    parser.add_argument(
        "--dump-where",
        default="dino,siglip",
        help="Comma-separated list of stages to dump (dino,siglip,fused,all).",
    )
    parser.add_argument(
        "--out",
        default="outputs/debug_pct",
        help="Directory used for activation dumps when --dump-activations is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = QuantConfig.from_file(args.cfg)

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = load_model(args.ckpt, hf_token=args.hf_token)
    model.to(device, dtype=dtype)

    dump_targets = _parse_dump_targets(args.dump_where) if args.dump_activations else set()
    dumper = ActivationDumper(dump_targets) if args.dump_activations else None
    dump_handles: List = []

    if args.mode == "off":
        quant_pct.disable(model)
        if dumper is not None:
            targets = dump_targets or _DEFAULT_DUMP_POINTS
            dump_handles = _register_passthrough_hooks(model, dumper, targets)
    else:
        quant_pct.enable(
            model,
            cfg,
            mode=args.mode,
            dumper=dumper.capture if dumper is not None else None,
        )

    try:
        image = Image.open(Path(args.image)).convert("RGB")
        prompt_builder = model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=args.question)
        prompt_text = prompt_builder.get_prompt()

        response = model.generate(image, prompt_text)
        print(response)
    finally:
        clip_values: Dict[str, object] = {}
        if dumper is not None:
            clip_values = _extract_clip_values(model)

        if args.mode == "off":
            for handle in dump_handles:
                handle.remove()
        else:
            quant_pct.disable(model)

        if dumper is not None:
            out_dir = Path(args.out)
            dumper.save(out_dir, clip_values=clip_values, mode=args.mode)


if __name__ == "__main__":
    main()
