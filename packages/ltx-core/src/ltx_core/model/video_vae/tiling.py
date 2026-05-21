import itertools
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Tuple

import torch


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    """
    Generate a 1D trapezoidal blending mask with linear ramps.

    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.
    Returns:
        A 1D tensor of shape `(length,)` with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap.

    Args:
        tile_size_in_pixels (int): Size of each tile in pixels. Must be at least 64 and divisible by 32.
        tile_overlap_in_pixels (int, optional): Overlap between tiles in pixels. Must be divisible by 32. Defaults to 0.
    """

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(
                f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}"
            )
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(
                f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}"
            )
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(
                f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles (chunks of frames) with optional overlap.

    Args:
        tile_size_in_frames (int): Number of frames in each tile. Must be at least 16 and divisible by 8.
        tile_overlap_in_frames (int, optional): Number of overlapping frames between consecutive tiles.
            Must be divisible by 8. Defaults to 0.
    """

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(
                f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}"
            )
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(
                f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}"
            )
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(
                f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap.

    Attributes:
        spatial_config: Configuration for splitting spatial dimensions into tiles.
        temporal_config: Configuration for splitting temporal dimension into tiles.
    """

    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=512, tile_overlap_in_pixels=64
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=64, tile_overlap_in_frames=24
            ),
        )


@dataclass(frozen=True)
class LatentIntervals:
    original_shape: torch.Size
    starts_per_dimension: Tuple[List[int], ...]
    ends_per_dimension: Tuple[List[int], ...]
    left_ramps_per_dimension: Tuple[List[int], ...]
    right_ramps_per_dimension: Tuple[List[int], ...]


class Tile(NamedTuple):
    """
    Represents a single tile.

    Attributes:
        in_coords:
            Tuple of slices specifying where to cut the tile from the INPUT tensor.

        out_coords:
            Tuple of slices specifying where this tile's OUTPUT should be placed in the reconstructed OUTPUT tensor.

        masks_1d:
            Per-dimension masks in OUTPUT units.
            These are used to create all-dimensional blending mask.

    Methods:
        blend_mask:
            Create a single N-D mask from the per-dimension masks.
    """

    in_coords: Tuple[slice, ...]
    out_coords: Tuple[slice, ...]
    masks_1d: Tuple[Tuple[torch.Tensor, ...]]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: List[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                # Broadcast mask along this dimension (length 1).
                one = torch.ones(1)

                view_shape[dim_idx] = 1
                per_dimension_masks.append(one.view(*view_shape))
                continue

            # Reshape (L,) -> (1, ..., L, ..., 1) so masks across dimensions broadcast-multiply.
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        # Multiply per-dimension masks to form the full N-D mask (separable blending window).
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def create_tiles_from_tile_sizes(
    vae: Any,  # noqa: ANN401
    latent_shape: torch.Size,
    spatial_tile_size: int,
    temporal_tile_size: int,
    spatial_overlap: int = 0,
    temporal_overlap: int = 0,
    spatial_axes_indices: Tuple[int, ...] = (3, 4),
    temporal_axes_indices: Tuple[int] = (2,),
) -> List[Tile]:
    latent_intervals = _create_intervals_from_tile_sizes(
        latent_shape=latent_shape,
        spatial_tile_size=spatial_tile_size,
        spatial_overlap=spatial_overlap,
        temporal_tile_size=temporal_tile_size,
        temporal_overlap=temporal_overlap,
        spatial_axes_indices=spatial_axes_indices,
        temporal_axes_indices=temporal_axes_indices,
    )
    return create_tiles_from_latent_intervals(
        vae, latent_intervals, temporal_axes_indices, spatial_axes_indices
    )


def create_tiles_from_tiles_amount(
    vae: Any,  # noqa: ANN401
    latent_shape: torch.Size,
    spatial_tiles_amount: int,
    temporal_tile_size: int,
    spatial_overlap: int = 0,
    temporal_overlap: int = 0,
    temporal_axes_indices: Tuple[int] = (2,),
    spatial_axes_indices: Tuple[int, ...] = (3, 4),
) -> List[Tile]:
    latent_intervals = _create_intervals_from_tiles_amount(
        latent_shape,
        spatial_tiles_amount,
        temporal_tile_size,
        temporal_overlap,
        spatial_overlap,
        temporal_axes_indices,
        spatial_axes_indices,
    )
    return create_tiles_from_latent_intervals(
        vae, latent_intervals, temporal_axes_indices, spatial_axes_indices
    )


def _create_intervals_from_tile_sizes(
    latent_shape: torch.Size,
    spatial_tile_size: int,
    temporal_tile_size: int,
    spatial_overlap: int = 0,
    temporal_overlap: int = 0,
    spatial_axes_indices: Tuple[int, ...] = (3, 4),
    temporal_axes_indices: Tuple[int] = (2,),
) -> LatentIntervals:
    starts_per_dimension = []
    ends_per_dimension = []
    left_ramps_per_dimension = []
    right_ramps_per_dimension = []
    for axis_index in range(len(latent_shape)):
        dimension_size = latent_shape[axis_index]
        size = dimension_size
        overlap = 0
        amount = 1
        if axis_index in temporal_axes_indices or axis_index in spatial_axes_indices:
            size = (
                temporal_tile_size
                if axis_index in temporal_axes_indices
                else spatial_tile_size
            )
            overlap = (
                temporal_overlap
                if axis_index in temporal_axes_indices
                else spatial_overlap
            )
            amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        if axis_index in temporal_axes_indices:
            # each temporal tile is causal / grabs a latent frame before the start
            starts[1:] = [s - 1 for s in starts[1:]]
            left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        starts_per_dimension.append(starts)
        ends_per_dimension.append(ends)
        left_ramps_per_dimension.append(left_ramps)
        right_ramps_per_dimension.append(right_ramps)

    return LatentIntervals(
        original_shape=latent_shape,
        starts_per_dimension=tuple(starts_per_dimension),
        ends_per_dimension=tuple(ends_per_dimension),
        left_ramps_per_dimension=tuple(left_ramps_per_dimension),
        right_ramps_per_dimension=tuple(right_ramps_per_dimension),
    )


def _create_intervals_from_tiles_amount(
    latent_shape: torch.Size,
    spatial_tiles_amount: int,
    temporal_tile_size: int,
    temporal_overlap: int = 0,
    spatial_overlap: int = 0,
    temporal_axes_indices: Tuple[int] = (2,),
    spatial_axes_indices: Tuple[int, ...] = (3, 4),
) -> LatentIntervals:
    starts_per_dimension = []
    ends_per_dimension = []
    left_ramps_per_dimension = []
    right_ramps_per_dimension = []
    for axis_index in range(len(latent_shape)):
        dimension_size = latent_shape[axis_index]
        size = dimension_size
        overlap = 0
        amount = 1
        if axis_index in temporal_axes_indices:
            size = temporal_tile_size
            overlap = temporal_overlap
            amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        elif axis_index in spatial_axes_indices:
            amount = spatial_tiles_amount
            overlap = spatial_overlap
            size = (
                dimension_size + (spatial_tiles_amount - 1) * overlap
            ) // spatial_tiles_amount
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        if axis_index in temporal_axes_indices:
            # each temporal tile is causal / grabs a latent frame before the start
            starts[1:] = [s - 1 for s in starts[1:]]
            left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        starts_per_dimension.append(starts)
        ends_per_dimension.append(ends)
        left_ramps_per_dimension.append(left_ramps)
        right_ramps_per_dimension.append(right_ramps)

    return LatentIntervals(
        original_shape=latent_shape,
        starts_per_dimension=tuple(starts_per_dimension),
        ends_per_dimension=tuple(ends_per_dimension),
        left_ramps_per_dimension=tuple(left_ramps_per_dimension),
        right_ramps_per_dimension=tuple(right_ramps_per_dimension),
    )


def create_tiles_from_latent_intervals(
    vae: Any,  # noqa: ANN401
    latent_intervals: LatentIntervals,
    temporal_axes_indices: Tuple[int] = (2,),
    spatial_axes_indices: Tuple[int, ...] = (3, 4),
) -> List[Tile]:
    full_dim_input_slices = []
    full_dim_output_slices = []
    full_dim_masks_1d = []
    for axis_index in range(len(latent_intervals.original_shape)):
        starts = latent_intervals.starts_per_dimension[axis_index]
        ends = latent_intervals.ends_per_dimension[axis_index]
        left_ramps = latent_intervals.left_ramps_per_dimension[axis_index]
        right_ramps = latent_intervals.right_ramps_per_dimension[axis_index]
        input_slices = [slice(s, e) for s, e in zip(starts, ends, strict=True)]
        output_slices = [slice(0, None) for _ in input_slices]
        masks_1d = [None] * len(input_slices)
        if axis_index in temporal_axes_indices:
            output_slices = []
            masks_1d = []
            for s, e, lr, rr in zip(starts, ends, left_ramps, right_ramps, strict=True):
                output_slice, mask_1d = vae.map_temporal_slice(s, e, lr, rr)
                output_slices.append(output_slice)
                masks_1d.append(mask_1d)
        elif axis_index in spatial_axes_indices:
            output_slices = []
            masks_1d = []
            for s, e, lr, rr in zip(starts, ends, left_ramps, right_ramps, strict=True):
                output_slice, mask_1d = vae.map_spatial_slice(s, e, lr, rr)
                output_slices.append(output_slice)
                masks_1d.append(mask_1d)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    tiles = []
    tile_in_coords = list(itertools.product(*full_dim_input_slices))
    tile_out_coords = list(itertools.product(*full_dim_output_slices))
    tile_mask_1ds = list(itertools.product(*full_dim_masks_1d))
    for in_coord, out_coord, mask_1d in zip(
        tile_in_coords, tile_out_coords, tile_mask_1ds, strict=True
    ):
        tiles.append(
            Tile(
                in_coords=in_coord,
                out_coords=out_coord,
                masks_1d=mask_1d,
            )
        )
    return tiles
