import os
import tempfile
from collections.abc import Iterator
from dataclasses import replace

import cv2
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem, VideoConditionByKeyframeIndex
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import Encoder as AudioEncoder
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.model.video_vae import Encoder as VideoEncoder
from ltx_core.model.video_vae import TilingConfig
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines import utils
from ltx_pipelines.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_LORA_STRENGTH,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.media_io import (
    decode_audio_from_file,
    encode_video,
    load_video_conditioning,
)
from ltx_pipelines.model_ledger import ModelLedger
from ltx_pipelines.pipeline_utils import (
    PipelineComponents,
    denoise_audio_video,
    encode_text,
    euler_denoising_loop,
    guider_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.pipeline_utils import decode_audio as vae_decode_audio
from ltx_pipelines.pipeline_utils import decode_video as vae_decode_video


def extract_first_frame(video_path: str, output_path: str | None = None) -> str:
    """Extract the first frame from a video file and save it as an image."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read first frame from: {video_path}")

    if output_path is None:
        temp_fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)

    cv2.imwrite(output_path, frame)
    return output_path


class AudioConditionByKeyframeIndex(ConditioningItem):
    """Conditions audio generation on keyframe latents at a specific frame index."""

    def __init__(self, keyframes: torch.Tensor, frame_idx: int, strength: float):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: AudioLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.keyframes)
        positions = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape.from_torch_shape(self.keyframes.shape),
            device=self.keyframes.device,
        )
        if self.frame_idx != 0:
            raise NotImplementedError(
                "AudioConditionByKeyframeIndex does not support frame_idx != 0"
            )

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.keyframes.device,
            dtype=self.keyframes.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
        )


class JustDubitPipeline:
    """
    Two-stage audio-video generation pipeline with video conditioning.

    Stage 1 generates video and audio at target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher quality output.
    Audio is automatically extracted from video conditioning sources.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora_path: str,
        distilled_lora_strength: float,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = utils.get_device(),
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=[
                LoraPathStrengthAndSDOps(
                    path=distilled_lora_path,
                    strength=distilled_lora_strength,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ],
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
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
        video_conditioning: list[tuple[str, float]],
        tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        text_encoder = self.stage_1_model_ledger.text_encoder()
        context_p, context_n = encode_text(
            text_encoder, prompts=[prompt, negative_prompt]
        )
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        torch.cuda.synchronize()
        del text_encoder
        utils.cleanup_memory()

        # Stage 1: Initial video and audio generation
        video_encoder = self.stage_1_model_ledger.video_encoder()
        audio_encoder = self.stage_1_model_ledger.audio_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        sigmas = (
            LTX2Scheduler()
            .execute(steps=num_inference_steps)
            .to(dtype=torch.float32, device=self.device)
        )

        # Get conditioning and determine final num_frames from source video
        stage_1_video_conditionings, determined_num_frames = (
            self._create_video_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=height,
                width=width,
                video_encoder=video_encoder,
                tiling_config=tiling_config,
            )
        )

        # Use determined_num_frames if conditioning was present, otherwise fallback to input
        num_frames = determined_num_frames if determined_num_frames > 0 else num_frames

        stage_1_audio_conditionings, audio_latent_shape = (
            self._create_audio_conditionings(
                video_conditioning=video_conditioning,
                audio_encoder=audio_encoder,
                target_num_frames=num_frames,
                frame_rate=frame_rate,
            )
        )

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
        )

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            v2a_cross_attention_mask, a2v_cross_attention_mask = (
                self._prepare_cross_attention_mask(
                    video_state, audio_state, dtype, self.device
                )
            )

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
                    v2a_cross_attention_mask=v2a_cross_attention_mask,
                    a2v_cross_attention_mask=a2v_cross_attention_mask,
                ),
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_video_conditionings,
            audio_conditionings=stage_1_audio_conditionings,
            audio_latent_shape=audio_latent_shape,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        torch.cuda.synchronize()
        del stage_1_audio_conditionings
        del stage_1_video_conditionings
        del transformer
        del audio_encoder
        utils.cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LoRA
        upscaled_video_latent = utils.upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        torch.cuda.synchronize()
        utils.cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)[
            1:
        ]  # remove first sigma

        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            v2a_cross_attention_mask, a2v_cross_attention_mask = (
                self._prepare_cross_attention_mask(
                    video_state, audio_state, dtype, self.device
                )
            )

            # Use clean audio latent as condition (no denoising in Stage 2)
            audio_state = replace(audio_state, latent=audio_state.clean_latent)
            audio_state = replace(
                audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask)
            )

            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                    v2a_cross_attention_mask=v2a_cross_attention_mask,
                    a2v_cross_attention_mask=a2v_cross_attention_mask,
                ),
            )

        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width * 2,
            height=height * 2,
            fps=frame_rate,
        )

        stage_2_conditionings, _ = self._create_video_conditionings(
            images=images,
            video_conditioning=video_conditioning,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            tiling_config=tiling_config,
        )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            audio_latent_shape=audio_latent_shape,
        )

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        utils.cleanup_memory()

        decoded_video = vae_decode_video(
            video_state, self.stage_2_model_ledger.video_decoder(), tiling_config
        )
        decoded_audio = vae_decode_audio(
            audio_state,
            self.stage_2_model_ledger.audio_decoder(),
            self.stage_2_model_ledger.vocoder(),
        )

        return decoded_video, decoded_audio

    def _create_video_conditionings(
        self,
        images: list[tuple[str, int, float]],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        video_encoder: VideoEncoder,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[list[ConditioningItem], int]:
        """Create video conditioning items from images and video sources."""
        conditionings = utils.image_conditionings_by_replacing_latent(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )

        num_frames = 0
        for video_path, strength in video_conditioning:
            video = load_video_conditioning(
                video_path=video_path,
                height=height,
                width=width,
                frame_cap=9999,  # Load all frames
                dtype=self.dtype,
                device=self.device,
            )
            # Ensure valid frame count for encoder (k*8 + 1)
            num_video_frames = video.shape[2]
            if (num_video_frames - 1) % 8 != 0:
                valid_video_frames = (num_video_frames - 1) // 8 * 8 + 1
                video = video[:, :, :valid_video_frames]

            if num_frames == 0:
                num_frames = video.shape[2]

            if tiling_config is None:
                encoded_video = video_encoder(video)
            else:
                encoded_video = video_encoder.tiled_encode(
                    video, tiling_config=tiling_config
                )
            conditionings.append(
                VideoConditionByKeyframeIndex(
                    keyframes=encoded_video, frame_idx=0, strength=strength
                )
            )

        return conditionings, num_frames

    def _create_audio_conditionings(
        self,
        video_conditioning: list[tuple[str, float]],
        audio_encoder: AudioEncoder,
        target_num_frames: int,
        frame_rate: float,
    ) -> tuple[list[ConditioningItem], AudioLatentShape | None]:
        """Create audio conditioning items by extracting audio from video sources."""
        conditionings = []
        processor = AudioProcessor(
            sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        ).to(device=self.device, dtype=torch.float32)

        target_duration = target_num_frames / frame_rate
        final_latent_shape = None

        for video_path, strength in video_conditioning:
            waveform, sample_rate = decode_audio_from_file(
                path=video_path, device=self.device
            )
            if waveform is None:
                raise ValueError(f"Could not load audio from {video_path}")

            # Trim or pad waveform to match target duration
            current_samples = waveform.shape[-1]
            valid_samples = int(target_duration * sample_rate)
            if current_samples > valid_samples:
                waveform = waveform[..., :valid_samples]
            elif current_samples < valid_samples:
                padding = valid_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            mel = processor.waveform_to_mel(waveform, sample_rate)
            mel = mel.to(dtype=self.dtype)

            if mel.shape[1] == 1:
                mel = torch.cat([mel, mel], dim=1)

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                encoded_audio = audio_encoder(mel)

            if final_latent_shape is None:
                final_latent_shape = AudioLatentShape.from_torch_shape(
                    encoded_audio.shape
                )

            conditionings.append(
                AudioConditionByKeyframeIndex(
                    keyframes=encoded_audio,
                    strength=strength,
                    frame_idx=0,
                )
            )

        return conditionings, final_latent_shape

    def _prepare_cross_attention_mask(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross-attention masks for video and audio modalities."""
        video_mask = video_state.denoise_mask.squeeze()
        audio_mask = audio_state.denoise_mask.squeeze()

        # v2a mask: Rows are audio, columns are video.
        # Mask is 1 where BOTH are target tokens (denoise_mask == 1).
        v2a_cross_attention_mask = audio_mask.unsqueeze(-1) * video_mask.unsqueeze(0)
        a2v_cross_attention_mask = v2a_cross_attention_mask.T

        v2a_cross_attention_mask = torch.where(
            v2a_cross_attention_mask == 1,
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(float("-inf"), dtype=dtype, device=device),
        )

        a2v_cross_attention_mask = torch.where(
            a2v_cross_attention_mask == 1,
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(float("-inf"), dtype=dtype, device=device),
        )

        return v2a_cross_attention_mask, a2v_cross_attention_mask


def main() -> None:
    parser = utils.default_2_stage_arg_parser()
    parser.add_argument(
        "--video_conditioning",
        dest="video_conditioning",
        action=utils.VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        default=[],
    )
    args = parser.parse_args()
    lora_strengths = (args.lora_strength + [DEFAULT_LORA_STRENGTH] * len(args.lora))[
        : len(args.lora)
    ]
    loras = [
        LoraPathStrengthAndSDOps(lora, strength, LTXV_LORA_COMFY_RENAMING_MAP)
        for lora, strength in zip(args.lora, lora_strengths, strict=True)
    ]

    # Extract first frame from video conditioning for image conditioning
    images = []
    if args.video_conditioning:
        first_video_path, strength = args.video_conditioning[0]
        first_frame_path = extract_first_frame(first_video_path)
        images.append((first_frame_path, 0, strength))

    pipeline = JustDubitPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora_path=args.distilled_lora_path,
        distilled_lora_strength=args.distilled_lora_strength,
        spatial_upsampler_path=args.spatial_upsampler_path,
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
        images=images,
        video_conditioning=args.video_conditioning,
        tiling_config=TilingConfig.default(),
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
