from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines import utils
from ltx_pipelines.constants import AUDIO_SAMPLE_RATE, DEFAULT_LORA_STRENGTH
from ltx_pipelines.media_io import encode_video
from ltx_pipelines.model_ledger import ModelLedger
from ltx_pipelines.pipeline_utils import (
    PipelineComponents,
    denoise_audio_video,
    encode_text,
    euler_denoising_loop,
    guider_denoising_func,
)
from ltx_pipelines.pipeline_utils import decode_audio as vae_decode_audio
from ltx_pipelines.pipeline_utils import decode_video as vae_decode_video
from ltx_pipelines.utils import image_conditionings_by_replacing_latent

device = utils.get_device()


class TI2VidOneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.

    Generates video at the target resolution in a single diffusion pass with
    classifier-free guidance (CFG). Supports optional image conditioning via
    the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        fp8transformer: bool = False,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        text_encoder = self.model_ledger.text_encoder()
        context_p, context_n = encode_text(
            text_encoder, prompts=[prompt, negative_prompt]
        )
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
        del text_encoder
        utils.cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        sigmas = (
            LTX2Scheduler()
            .execute(steps=num_inference_steps)
            .to(dtype=torch.float32, device=self.device)
        )

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        torch.cuda.synchronize()
        del transformer
        utils.cleanup_memory()

        decoded_video = vae_decode_video(video_state, self.model_ledger.video_decoder())
        decoded_audio = vae_decode_audio(
            audio_state, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )

        return decoded_video, decoded_audio


def main() -> None:
    parser = utils.default_1_stage_arg_parser()
    args = parser.parse_args()
    lora_strengths = (args.lora_strength + [DEFAULT_LORA_STRENGTH] * len(args.lora))[
        : len(args.lora)
    ]
    loras = [
        LoraPathStrengthAndSDOps(lora, strength, LTXV_LORA_COMFY_RENAMING_MAP)
        for lora, strength in zip(args.lora, lora_strengths, strict=True)
    ]
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=loras,
        fp8transformer=args.enable_fp8,
    )
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
