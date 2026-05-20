"""JustDubIt training strategy for video dubbing and lip-sync.

This strategy implements audio-video generation training for dubbing where:
- Reference video and audio latents condition the generation
- Target video and audio are generated with synchronized lip movements
- Supports first frame conditioning and masked loss computation
- Joint audio-video training with cross-attention masking support
"""

from dataclasses import replace
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class JustDubItConfig(TrainingStrategyConfigBase):
    """Configuration for JustDubIt video dubbing and lip-sync training strategy."""

    name: Literal["justdubit"] = "justdubit"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=True,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )

    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Directory name for latents of reference videos",
    )

    reference_audio_latents_dir: str = Field(
        default="reference_audio_latents",
        description="Directory name for audio latents of reference videos",
    )

    enable_cross_attention_masking: bool = Field(
        default=False,
        description="Whether to enable cross-attention masking between video target and audio reference",
    )

    # -----------------------------------------------------------------------------
    # Masked Training Configuration
    # -----------------------------------------------------------------------------
    mask_config: dict = Field(
        default_factory=lambda: {
            "mask_dir": "masks",
            "use_masked_loss": False,
            "mask_loss_weight": 1.0,
            "background_loss_weight": 0.0,
            "mask_threshold": 0.5,
            "use_soft_masks": False,
        },
        description="Dictionary containing all mask-related configuration options",
    )


class JustDubItStrategy(TrainingStrategy):
    """JustDubIt training strategy for video dubbing and lip-sync.

    This strategy implements audio-video generation training for dubbing where:
    - Reference video and audio latents condition the generation
    - Target video and audio are generated with synchronized lip movements
    - Supports first frame conditioning and masked loss computation
    - Joint audio-video training with cross-attention masking support
    """

    config: JustDubItConfig

    def __init__(self, config: JustDubItConfig):
        """Initialize strategy with configuration.

        Args:
            config: AV-to-AV configuration
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components. Always True for AV-to-AV training."""
        return True

    def get_data_sources(self) -> dict[str, str]:
        """
        AV-to-AV training requires latents, audio latents, reference latents, and reference audio latents.
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
            self.config.audio_latents_dir: "audio_latents",
            self.config.reference_latents_dir: "ref_latents",
            self.config.reference_audio_latents_dir: "ref_audio_latents",
        }
        if self.config.mask_config["use_masked_loss"] and self.config.mask_config["mask_dir"] is not None:
            sources[self.config.mask_config["mask_dir"]] = "masks"

        return sources

    @staticmethod
    def _prepare_cross_attention_mask(
        ref_video_len: int,
        ref_audio_len: int,
        target_video_len: int,
        target_audio_len: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Prepare cross-attention mask for video and audio, return cross-attention masks"""
        video_mask = torch.cat(
            [
                torch.zeros(ref_video_len, dtype=torch.bool, device=device),
                torch.ones(target_video_len, dtype=torch.bool, device=device),
            ],
            dim=0,
        )
        audio_mask = torch.cat(
            [
                torch.zeros(ref_audio_len, dtype=torch.bool, device=device),
                torch.ones(target_audio_len, dtype=torch.bool, device=device),
            ],
            dim=0,
        )

        v2a_cross_attention_mask = audio_mask.unsqueeze(-1) * video_mask.unsqueeze(0)
        a2v_cross_attention_mask = v2a_cross_attention_mask.T

        v2a_cross_attention_mask = torch.where(
            v2a_cross_attention_mask == 1,
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(float("-inf"), dtype=torch.float32),
        ).to(device)

        a2v_cross_attention_mask = torch.where(
            a2v_cross_attention_mask == 1,
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(float("-inf"), dtype=torch.float32),
        ).to(device)

        return v2a_cross_attention_mask, a2v_cross_attention_mask

    def _prepare_masks_for_target(
        self, mask_data: dict[str, Tensor], target_shape: torch.Size, device: torch.device
    ) -> Tensor:
        """
        Process masks for the target sequence.

        Args:
            mask_data: Dictionary containing mask tensors
            target_shape: Shape of the target latents [B, seq_len, latent_dim]
            device: Target device for masks

        Returns:
            Processed foreground masks for target sequence
        """
        masks = mask_data["masks"].to(device)  # [B, T, H, W]
        target_seq_len = target_shape[1]

        # Convert to binary masks if needed
        if not self.config.mask_config["use_soft_masks"] and masks.dtype != torch.bool:
            masks = masks > self.config.mask_config["mask_threshold"]

        # Map to target sequence tokens - flatten spatial dimensions and truncate to target length
        sequence_masks = masks.flatten(1)  # [B, T*H*W]

        return sequence_masks[:, :target_seq_len]

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for AV-to-AV training."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        target_video_latents = latents["latents"]
        ref_video_latents = batch["ref_latents"]["latents"]

        # Get dimensions
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        ref_latents_info = batch["ref_latents"]
        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        target_video_latents = self._video_patchifier.patchify(target_video_latents)
        ref_video_latents = self._video_patchifier.patchify(ref_video_latents)

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_video_latents.shape[0]
        ref_video_seq_len = ref_video_latents.shape[1]
        target_video_seq_len = target_video_latents.shape[1]
        device = target_video_latents.device
        dtype = target_video_latents.dtype

        # Create conditioning mask
        ref_video_conditioning_mask = torch.ones(batch_size, ref_video_seq_len, dtype=torch.bool, device=device)

        # Target tokens: check for first frame conditioning
        target_video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Combined conditioning mask
        video_conditioning_mask = torch.cat([ref_video_conditioning_mask, target_video_conditioning_mask], dim=1)

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(target_video_latents)
        video_noise = torch.randn_like(target_video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * target_video_latents + sigmas_expanded * video_noise

        # For conditioning tokens, use clean latents
        target_video_conditioning_mask_expanded = target_video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(target_video_conditioning_mask_expanded, target_video_latents, noisy_video)

        # Concatenate reference and noisy input video latents
        video_latents = torch.cat([ref_video_latents, noisy_video], dim=1)

        # Compute video targets (velocity prediction)
        video_targets = video_noise - target_video_latents

        # Create per-token timesteps
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        # Generate positions for reference and target separately, then concatenate
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Concatenate positions along sequence dimension
        video_positions = torch.cat([ref_positions, target_positions], dim=2)

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=video_latents,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            cross_attention_mask=None,
        )

        # Video loss mask: True for tokens we want to compute loss on (non-conditioning tokens)
        video_loss_mask = ~video_conditioning_mask

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        audio_modality, audio_targets, audio_loss_mask, ref_audio_seq_len = self._prepare_audio_inputs(
            batch=batch,
            sigmas=sigmas,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        if self.config.enable_cross_attention_masking:
            v2a_cross_attention_mask, a2v_cross_attention_mask = self._prepare_cross_attention_mask(
                ref_video_seq_len,
                ref_audio_seq_len,
                video_modality.latent.shape[1] - ref_video_seq_len,
                audio_modality.latent.shape[1] - ref_audio_seq_len,
                device,
            )
            video_modality = replace(video_modality, cross_attention_mask=a2v_cross_attention_mask)
            audio_modality = replace(audio_modality, cross_attention_mask=v2a_cross_attention_mask)

        if self.config.mask_config["use_masked_loss"]:
            foreground_masks = self._prepare_masks_for_target(
                mask_data=batch["masks"],
                target_shape=video_modality.latent.shape,
                device=device,
            )
        else:
            foreground_masks = None

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            ref_video_seq_len=ref_video_seq_len,
            ref_audio_seq_len=ref_audio_seq_len,
            foreground_masks=foreground_masks,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor, int]:
        """Prepare audio inputs for joint audio-video training.

        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as video)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype

        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Get audio latents - dataset provides uniform non-patchified format [B, C, T, F]
        target_audio_data = batch["audio_latents"]
        target_audio_latents = target_audio_data["latents"]
        ref_audio_latents = batch["ref_audio_latents"]["latents"]

        # Patchify audio latents: [B, C, T, F] -> [B, T, C*F]
        target_audio_latents = self._audio_patchifier.patchify(target_audio_latents)
        ref_audio_latents = self._audio_patchifier.patchify(ref_audio_latents)

        target_audio_seq_len = target_audio_latents.shape[1]
        ref_audio_seq_len = ref_audio_latents.shape[1]

        # create conditioning mask
        ref_audio_conditioning_mask = torch.ones(batch_size, ref_audio_seq_len, dtype=torch.bool, device=device)
        target_audio_conditioning_mask = torch.zeros(batch_size, target_audio_seq_len, dtype=torch.bool, device=device)
        audio_conditioning_mask = torch.cat([ref_audio_conditioning_mask, target_audio_conditioning_mask], dim=1)

        # Sample audio noise
        audio_noise = torch.randn_like(target_audio_latents)

        # Apply noise to audio (same sigma as video)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * target_audio_latents + sigmas_expanded * audio_noise

        # Concatenate reference and noisy input audio latents
        audio_latents = torch.cat([ref_audio_latents, noisy_audio], dim=1)

        # Compute audio targets
        audio_targets = audio_noise - target_audio_latents

        # Create per-token timesteps
        audio_timesteps = self._create_per_token_timesteps(audio_conditioning_mask, sigmas.squeeze())

        # Generate audio positions
        ref_audio_positions = self._get_audio_positions(
            num_time_steps=ref_audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        target_audio_positions = self._get_audio_positions(
            num_time_steps=target_audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Concatenate positions along sequence dimension
        audio_positions = torch.cat([ref_audio_positions, target_audio_positions], dim=2)

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            latent=audio_latents,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
            cross_attention_mask=None,
        )

        # Audio loss mask, True for tokens we want to compute loss on (non-conditioning tokens)
        audio_loss_mask = ~audio_conditioning_mask

        return audio_modality, audio_targets, audio_loss_mask, ref_audio_seq_len

    def _compute_masked_loss(
        self,
        model_pred: Tensor,
        targets: Tensor,
        target_conditioning_mask: Tensor,
        foreground_masks: Tensor,
    ) -> Tensor:
        """
        Compute masked loss with foreground/background weighting.

        Args:
            model_pred: Model predictions
            targets: Ground truth targets
            target_conditioning_mask: Bool mask for target conditioning (True means condition token, False means target)
            foreground_masks: Foreground mask for tokens (usually bool or float, shape [B, seq_len])

        Returns:
            Computed masked loss
        """
        base_loss = (model_pred - targets).pow(2)

        conditioning_loss_mask = target_conditioning_mask.unsqueeze(-1).float()

        # Foreground mask
        foreground_loss_mask = foreground_masks.unsqueeze(-1).float()

        # Combine masks with weights
        if self.config.mask_config["background_loss_weight"] > 0:
            # Apply different weights to foreground vs background
            mask_thresh = (
                self.config.mask_config["mask_threshold"] if not self.config.mask_config["use_soft_masks"] else 0.5
            )
            mask_weights = torch.where(
                foreground_loss_mask > mask_thresh,
                torch.tensor(self.config.mask_config["mask_loss_weight"], device=foreground_loss_mask.device),
                torch.tensor(self.config.mask_config["background_loss_weight"], device=foreground_loss_mask.device),
            )
            combined_mask = conditioning_loss_mask * mask_weights
        else:
            # Only foreground loss
            combined_mask = conditioning_loss_mask * foreground_loss_mask * self.config.mask_config["mask_loss_weight"]

        # Apply mask and normalize
        effective_tokens = combined_mask.sum()
        if effective_tokens > 0:
            masked_loss = base_loss.mul(combined_mask).div(combined_mask.mean())
            return masked_loss.mean()
        else:
            logger.warning("No foreground tokens, using standard loss")
            loss = (
                base_loss.mul(conditioning_loss_mask).div(conditioning_loss_mask.mean())
                * self.config.mask_config["background_loss_weight"]
            )
            return loss.mean()

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and audio."""
        # Extract target portion of prediction
        ref_video_seq_len = inputs.ref_video_seq_len
        target_video_pred = video_pred[:, ref_video_seq_len:, :]

        # Get target portion of loss mask
        target_video_loss_mask = inputs.video_loss_mask[:, ref_video_seq_len:]

        # Video loss
        if self.config.mask_config["use_masked_loss"]:
            video_loss = self._compute_masked_loss(
                model_pred=target_video_pred,
                targets=inputs.video_targets,
                target_conditioning_mask=target_video_loss_mask,
                foreground_masks=inputs.foreground_masks,
            )
        else:
            video_loss = (target_video_pred - inputs.video_targets).pow(2)
            video_loss_mask = target_video_loss_mask.unsqueeze(-1).float()
            video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
            video_loss = video_loss.mean()

        # Extract target portion of audio prediction
        ref_audio_seq_len = inputs.ref_audio_seq_len
        target_audio_pred = audio_pred[:, ref_audio_seq_len:, :]

        # Get target portion of audio loss mask
        target_audio_loss_mask = inputs.audio_loss_mask[:, ref_audio_seq_len:]

        # Audio loss
        audio_loss = (target_audio_pred - inputs.audio_targets).pow(2)
        audio_loss_mask = target_audio_loss_mask.unsqueeze(-1).float()
        audio_loss = audio_loss.mul(audio_loss_mask).div(audio_loss_mask.mean())
        audio_loss = audio_loss.mean()

        # Combined loss
        return video_loss + audio_loss
