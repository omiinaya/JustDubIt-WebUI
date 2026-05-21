import argparse
import gc
from pathlib import Path

import torch

from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.upsampler import LatentUpsampler
from ltx_core.model.video_vae import Encoder as VideoEncoder
from ltx_pipelines.constants import (
    DEFAULT_CFG_GUIDANCE_SCALE,
    DEFAULT_FRAME_RATE,
    DEFAULT_HEIGHT,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
)
from ltx_pipelines.media_io import load_image_conditioning


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def resolve_path(path: str) -> str:
    return str(Path(path).expanduser().resolve().as_posix())


class VideoConditioningAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        path, strength_str = values
        strength = float(strength_str)
        current = getattr(namespace, self.dest) or []
        current.append((path, strength))
        setattr(namespace, self.dest, current)


class ImageAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        path, frame_idx, strength_str = values
        frame_idx = int(frame_idx)
        strength = float(strength_str)
        current = getattr(namespace, self.dest) or []
        current.append((path, frame_idx, strength))
        setattr(namespace, self.dest, current)


class LoraAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        path, strength_str = values
        strength = float(strength_str)
        current = getattr(namespace, self.dest) or []
        current.append(
            LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP)
        )
        setattr(namespace, self.dest, current)


def image_conditionings_by_replacing_latent(
    images: list[tuple[str, int, float]],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for image_path, frame_idx, strength in images:
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=strength,
                latent_idx=frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[tuple[str, int, float]],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for image_path, frame_idx, strength in images:
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByKeyframeIndex(
                keyframes=encoded_image, frame_idx=frame_idx, strength=strength
            )
        )
    return conditionings


def upsample_video(
    latent: torch.Tensor, video_encoder: VideoEncoder, upsampler: LatentUpsampler
) -> torch.Tensor:
    latent = video_encoder.per_channel_statistics.un_normalize(latent)
    latent = upsampler(latent)
    latent = video_encoder.per_channel_statistics.normalize(latent)
    return latent


def basic_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=resolve_path, required=True)
    parser.add_argument("--gemma_root", type=resolve_path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=resolve_path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--frame_rate", type=float, default=DEFAULT_FRAME_RATE)
    parser.add_argument(
        "--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS
    )
    parser.add_argument(
        "--image",
        dest="images",
        action=ImageAction,
        nargs=3,
        metavar=("PATH", "FRAME_IDX", "STRENGTH"),
        default=[],
    )
    parser.add_argument("--lora", type=resolve_path, action="append", default=[])
    parser.add_argument("--lora_strength", type=float, action="append", default=[])
    parser.add_argument("--enable_fp8", action="store_true")
    return parser


def default_1_stage_arg_parser() -> argparse.ArgumentParser:
    parser = basic_arg_parser()
    parser.add_argument(
        "--cfg_guidance_scale", type=float, default=DEFAULT_CFG_GUIDANCE_SCALE
    )
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)

    return parser


def default_2_stage_arg_parser() -> argparse.ArgumentParser:
    parser = default_1_stage_arg_parser()
    parser.add_argument("--distilled_lora_path", type=resolve_path, required=True)
    parser.add_argument(
        "--distilled_lora_strength", type=float, default=DEFAULT_LORA_STRENGTH
    )
    parser.add_argument("--spatial_upsampler_path", type=resolve_path, required=True)
    return parser


def default_2_stage_distilled_arg_parser() -> argparse.ArgumentParser:
    parser = basic_arg_parser()
    parser.add_argument("--spatial_upsampler_path", type=resolve_path, required=True)
    return parser
