"""
This module contains transforms that implement physics-based forward
models of MR image formation.
"""
__all__ = [
    'RandomSusceptibilityMixtureTransform',
    'SusceptibilityToFieldmapTransform',
    'ShimTransform',
    'OptimalShimTransform',
    'RandomShimTransform',
    'HertzToPhaseTransform',
    'HertzToVoxelShiftTransform',
    'ApplyB0DistortionTransform',
    'B0DistortionTransform',
    'RandomB0DistortionTransform',
    'GradientEchoTransform',
    'ReturnGradientEchoParameters',
    'GMMGradientEchoTransform',
    'RandomGMMGradientEchoTransform',
]

# stdlib
import math
from math import inf
from collections import namedtuple

# dependencies
import torch
import interpol
import typing_extensions as tx
from torch import Tensor

# internal
from .base import FinalTransform, NonFinalTransform, Transform
from .baseutils import Arguments, NoArguments, Returned, nested_get, prepare_output, return_requires, returns_find, returns_update
from .labels import GaussianMixtureFinalTransform, GaussianMixtureTransform
from .intensity import (
    RandomMulFieldTransform, AddValueTransform, MulValueTransform,
    ReturnValueTransform
)
from .random import Sampler, Uniform, RandInt, make_range
from .utils.py import cast_like, ensure_list, make_vector
from .utils.smart_inplace import exp_, div_
from .utils.conv import smoothnd
from .utils import warps
from .utils import b0
from . import typing as cct


class RandomSusceptibilityMixtureTransform(NonFinalTransform):
    """
    A RandomGaussianMixtureTransform tailored to susceptibility maps.

    This transform returns a delta susceptibility map (with respect to
    air), in ppm.
    """

    Next = GaussianMixtureTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = GaussianMixtureFinalTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        mu_tissue: cct.SamplerOrBound[float] = Uniform(9, 10),
        sigma_tissue: cct.SamplerOrBound[float] = 0.01,
        mu_bone: cct.SamplerOrBound[float] = Uniform(12, 13),
        sigma_bone: cct.SamplerOrBound[float] = 0.1,
        fwhm: cct.SamplerOrBound[float] = 2,
        label_air: cct.NumberOrSequence[int] = 0,
        label_bone: tx.Optional[cct.NumberOrSequence[int]] = None,
        dtype: tx.Optional[torch.dtype] = None,
        *,
        shared: cct.SharedT = 'channels',
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        mu_tissue : Sampler | float
            Distribution of negative susceptibility offsets (in ppm)
            of soft tissues with respect to air.
            Will be negated (air susceptibility is larger than all tissues).
            If float: upper bound.
        sigma_tissue : Sampler | float
            Standard deviation of susceptibility offsets, within class.
            If float: uper bound.
        mu_bone : Sampler | float
            Distribution of negative susceptibility offsets (in ppm)
            of hard tissues with respect to air.
            Will be negated (air susceptibility is larger than all tissues).
            If float: upper bound.
        sigma_bone: Sampler | float
            Standard deviation of susceptibility offsets, within class.
            If float: upper bound.
        fwhm : Sampler | [list of] float
            Sampling function for smoothing width.
            If float: upper bound.
        label_air : [list of] int
            Labels corresponding to air.
        label_bone : [list of] int
            Labels corresponding to bone and teeth.
        """
        super().__init__(shared=shared, **kwargs)
        self.mu_tissue = Uniform.make(make_range(0, mu_tissue))
        self.sigma_tissue = Uniform.make(make_range(0, sigma_tissue))
        self.mu_bone = Uniform.make(make_range(0, mu_bone))
        self.sigma_bone = Uniform.make(make_range(0, sigma_bone))
        self.fwhm = Uniform.make(make_range(0, fwhm))
        self.label_air = label_air
        self.label_bone = label_bone
        self.dtype = dtype

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)

        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        label_air = ensure_list(self.label_air)
        label_bone = ensure_list(self.label_bone or [])
        label_tissue = [i for i in range(n) if i not in label_bone + label_air]

        mu = [0] * n
        sigma = [0] * n
        fwhm = [0] * n

        if isinstance(self.mu_tissue, Sampler):
            for i in label_tissue:
                mu[i] = self.mu_tissue()
        elif isinstance(self.mu_tissue, (list, tuple)):
            for i in label_tissue:
                mu[i] = self.mu_tissue[i]
        else:
            for i in label_tissue:
                mu[i] = self.mu_tissue

        if isinstance(self.sigma_tissue, Sampler):
            for i in label_tissue:
                sigma[i] = self.sigma_tissue()
        elif isinstance(self.sigma_tissue, (list, tuple)):
            for i in label_tissue:
                sigma[i] = self.sigma_tissue[i]
        else:
            for i in label_tissue:
                sigma[i] = self.sigma_tissue

        if isinstance(self.mu_bone, Sampler):
            for i in label_bone:
                mu[i] = self.mu_bone()
        elif isinstance(self.mu_bone, (list, tuple)):
            for i in label_bone:
                mu[i] = self.mu_bone[i]
        else:
            for i in label_tissue:
                mu[i] = self.mu_bone

        if isinstance(self.sigma_bone, Sampler):
            for i in label_bone:
                sigma[i] = self.sigma_bone()
        elif isinstance(self.sigma_bone, (list, tuple)):
            for i in label_bone:
                sigma[i] = self.sigma_bone[i]
        else:
            for i in label_bone:
                sigma[i] = self.sigma_bone

        if isinstance(self.fwhm, Sampler):
            for i in range(n):
                fwhm[i] = self.fwhm()
        elif isinstance(self.fwhm, (list, tuple)):
            for i in range(n):
                fwhm[i] = self.fwhm[i]
        else:
            for i in range(n):
                fwhm[i] = self.fwhm

       # Negate
        mu = [-v for v in mu]

        return GaussianMixtureTransform(
            mu, sigma, fwhm, dtype=self.dtype, **self.get_prm()
        ).make_final(x, max_depth-1)


class SusceptibilityToFieldmapTransform(FinalTransform):
    """
    Convert a susceptibiity map (in ppm) into a field map (in Hz)
    """

    def __init__(
        self,
        axis: tx.Union[int, cct.VectorLike[float]] = -1,
        field_strength: float = 3,
        larmor: float = 42.576E6,
        s0: float = 0.4,
        s1: float = -9.5,
        voxel_size: float = 1,
        mask_air: tx.Union[bool, tx.Literal['fill']] = False,
        **kwargs
     ) -> None:
        """

        Parameters
        ----------
        axis : int | sequence[float]
            Direction of the main magnetic field.
            - If a vector or floats, it is unit vector that encodes the
              direction of the main magnetic field in the "scaled voxel"
              coordinate systems.
            - If an int, the main magnetic field is assumed to be aligned
              with one of the dimensions of the voxel grid, and `zaxis` is
              the index of this dimension.
            I.e., `0` is equivalent to `[1, 0, 0]`.
        field_strength : float
            Strength of the main magnetic field, in Tesla.
            If None, return a fieldmap in ppm.
        larmor : float
            Larmor frequency (default: Larmor frequency of water)
        s0 : float, default=0.4
            Susceptibility of air (ppm)
        s1 : float, default=-9.5
            Susceptibility of tissue minus susceptiblity of air (ppm)
            (only used if `ds` is a boolean mask)
        voxel_size : [list of] float
            Voxel size
        mask_air : bool
            Mask air (where delta susceptibility == 0) from
            resulting fieldmap.

        """
        super().__init__(**kwargs)
        self.axis = axis
        self.field_strength = field_strength
        self.larmor = larmor
        self.s0 = s0
        self.s1 = s1
        self.mask_air = mask_air
        self.voxel_size = voxel_size

    def xform(self, x: Tensor) -> Returned:
        ndim = x.ndim - 1

        axis = self.axis
        if isinstance(axis, int):
            axis = 1 + ((x.ndim - 1 + axis) if axis < 0 else axis)
        field = b0.chi_to_fieldmap(x, zaxis=axis, ndim=x.ndim-1,
                                   s0=self.s0, s1=self.s1, vx=self.voxel_size)

        if self.mask_air:
            field.masked_fill_(x == 0, 0)

            if self.mask_air == 'fill':
                mask0 = mask = (x == 0)
                while mask.any():
                    smask = mask.logical_not().to(field)
                    sfield = smoothnd(field, fwhm=[16] * ndim, bound="replicate")
                    smask = smoothnd(smask, fwhm=[16] * ndim, bound="replicate")
                    mask = smask > 0
                    sfield[mask] /= smask[mask]
                    mask = mask.logical_not_()
                    field[mask0] = sfield[mask0]

        if self.field_strength:
            field = b0.ppm_to_hz(field, self.field_strength, self.larmor)

        return prepare_output(dict(input=x, output=field, fieldmap=field),
                              self.returns)


class ShimTransform(FinalTransform):
    """Apply a shim field to the input field map."""

    def __init__(
        self,
        linear: tx.Optional[cct.VectorLike[float]] = None,
        quadratic: tx.Optional[cct.VectorLike[float]] = None,
        isocenter: tx.Optional[cct.VectorLike[float]] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        linear : (3|1,) tensor | [list of] float
            Linear components (3D: [XY, XZ, YX], 2D: [XY])
        quadratic : (2|1,) tensor | [list of] float
            Quadratic components (3D: [XXYY, XXZZ], 2D: [XXYY])
        isocenter : (3|2,) tensor | [list of] float
            Coordinates of the isocenter, in voxels

        Other Parameters
        ------------------
        returns : [(list | dict) of] {'input', 'output', 'shim'}
        """
        super().__init__(**kwargs)
        self.linear = linear
        self.quadratic = quadratic
        self.isocenter = isocenter

    def xform(self, x: Tensor) -> Returned:
        ndim = x.ndim - 1
        order = 1 if self.quadratic is None else 2
        nb_lin = 3 if ndim == 3 else 1
        nb_quad = 0 if order < 2 else 2 if ndim == 3 else 1
        if self.linear is None:
            linear = [0] * nb_lin
        else:
            linear = make_vector(self.linear, nb_lin).unbind()
        if self.quadratic is None:
            quadratic = [0] * nb_quad
        else:
            quadratic = make_vector(self.quadratic, nb_quad).unbind()
        basis = b0.yield_spherical_harmonics(x.shape[1:], order=order)

        shim = 0
        for b in basis:
            if linear:
                alpha, *linear = linear
            elif quadratic:
                alpha, *quadratic = quadratic
            else:
                break
            shim += alpha * b
        y = x + shim
        return prepare_output(
            dict(input=x, output=y, shim=shim), self.returns
        )


class OptimalShimTransform(NonFinalTransform):
    """
    Compute a linear combination of spherical harmonics that flattens the
    input field map
    """

    Next = Final = AddValueTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        max_order: int = 2,
        lam_abs: float = 1,
        lam_grad: float = 10,
        mask: tx.Union[bool, Tensor] = True,
        isocenter: tx.Optional[cct.VectorLike[float]] = None,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        max_order : int
            Maximum order of spherical basis functions
        lam_abs : float
            Regularization factor for absolute values
        lam_grad : float
            Regularization factor for first order gradients
        mask : bool | Tensor
            Mask zeros/NaNs from objective functions
        isocenter : (3|2,) tensor | [list of] float
            Coordinates of the isocenter, in voxels.
            Defaults to the center of the image.

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'shim'}
        """
        super().__init__(**kwargs)
        self.max_order = max_order
        self.lam_abs = lam_abs
        self.lam_grad = lam_grad
        self.mask = mask
        self.isocenter = isocenter

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)
        mask = self.mask
        if mask is True:
            mask = (x != 0) if self.mask else None
        elif mask is False:
            mask = None
        shim = b0.shim(
            x,
            max_order=self.max_order,
            ndim=x.ndim-1,
            mask=mask,
            lam_abs=self.lam_abs,
            lam_grad=self.lam_grad,
            isocenter=self.isocenter,
            returns='correction'
        ).neg_()
        return AddValueTransform(
            shim, value_name='shim', **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomShimTransform(NonFinalTransform):
    """
    Sample a random (imperfect) shim.

    This function randomly samples the coefficients of a field encoded
    by spherical harmonics.
    """

    Next = ShimTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = AddValueTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        coefficients: cct.SamplerOrBound[int] = 5,
        max_order: cct.SamplerOrBound[int] = 2,
        *,
        shared: cct.SharedT = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        coefficients : Sampler | int
            Sampler for spherical harmonics coefficients (or upper bound)
        max_order : Sampler | int
            Sampler for spherical harmonics order (or upper bound)

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'shim'}
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
        """
        super().__init__(shared=shared, **kwargs)
        self.coefficients = Uniform.make(make_range(0, coefficients))
        self.max_order = RandInt.make(make_range(1, max_order))

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        ndim = x.ndim - 1
        batch = len(x)
        if 'channels' in self.shared:
            batch = 1

        if isinstance(self.max_order, Sampler):
            order = self.max_order()
        else:
            order = self.max_order

        nb_bases = 2*order
        if ndim == 2:
            nb_bases -= 1
        elif ndim == 3:
            nb_bases += 1
        else:
            raise ValueError('Only dim 2 and 3 implemented')

        coefficients = self.coefficients
        if isinstance(coefficients, Sampler):
            coefficients = coefficients([nb_bases, batch])
        else:
            coefficients = ensure_list(coefficients, nb_bases)

        nb_lin = 3 if ndim == 3 else 1
        lin = coefficients[:nb_lin]
        quad = coefficients[nb_lin:]

        return ShimTransform(
            lin, None if order < 2 else quad, **self.get_prm()
        ).make_final(x, max_depth-1)


class HertzToPhaseTransform(FinalTransform):
    """Converts a ΔB0 field (in Hz) into a Phase shift field Δφ (in rad)"""

    def __init__(self, te: float = 0, **kwargs) -> None:
        """

        Parameters
        ----------
        te : float
            Echo time, in sec.

        """
        super().__init__(**kwargs)
        self.te = te

    def xform(self, x: Tensor) -> Tensor:
        return (2 * math.pi * self.te) * x


class HertzToVoxelShiftTransform(FinalTransform):
    """Converts a ΔB0 field (in Hz) into a voxel shift field Δv"""

    def __init__(self, bandwidth: float = 30, **kwargs) -> None:
        """

        Parameters
        ----------
        bandwidth : float
            Bandwidth per pixel along the slow direction, in Hz/pixel.

            This is the phase-encoding direction in an EPI sequence.
            It can also be the frequency-encoding direction in a
            non-EPI sequence, although in such cases the bandwidth per
            pixel is much larger, and displacements are only meaningful
            in extremely high-resolution scans.

            !!! changedin "![v0.4](https://img.shields.io/badge/v0.4-yellow) \
                        Default  changed from `140` to `30`"

        """
        super().__init__(**kwargs)
        self.bandwidth = bandwidth

    def xform(self, x: Tensor) -> Tensor:
        return x / self.bandwidth



class ApplyB0DistortionTransform(FinalTransform):
    """Apply a pre-computed B0 voxel displacement map."""

    def __init__(
        self,
        flow: tx.Union[Tensor, str, None] = None,
        vdm: tx.Union[Tensor, str, None] = None,
        controls: tx.Union[Tensor, str, None] = None,
        axis: tx.Union[int, tx.Sequence[float]] = -1,
        order: int = 3,
        bound: cct.TorchBound = 'border',
        nearest_if_label: bool = True,
        *,
        dtype: tx.Optional[torch.dtype] = None,
        device: tx.Optional[cct.TorchDevice] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        flow : (C, D, *spatial) tensor | [list of] (str | int)
            Flow field (in voxels),
            If an index or list of indices, they are used to retrieve
            the flow from the called arguments.
            (If not provided, `vdm` of `controls` must be provided)
        vdm : (C, *spatial) tensor
            Voxel displacement field
            If an index or list of indices, they are used to retrieve
            the flow from the called arguments.
            (If not provided, `controls` or `flow` must be provided)
        controls : (C, *shape) tensor
            Spline control points
            If an index or list of indices, they are used to retrieve
            the flow from the called arguments.
            (If not provided, `vdm` or `flow` must be provided)
        axis : int | sequence[float]
            If int, the distortion is applied along this dimension.
            If sequence of floats, it is a unit vector that encodes the
            direction of the distortion in the voxel coordinate system.
        order : 1..7
            Order of the splines that encode the smooth deformation.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode used for the deformed image.
        nearest_if_label : bool
            By default, if a tensor has an integer data type, it
            is deformed using label-specific resampling (each unique
            label is extracted and resampled using linear interpolation,
            and an argmax output label map is computed on the fly).
            If `nearest_if_label=True`, the entire label map will be
            resampled at once using nearest-neighbour interpolation.
        """
        super().__init__(**kwargs)
        self.flow = flow
        self.vdm = vdm
        self.controls = controls
        self.order = order
        self.bound = bound
        self.nearest_if_label = nearest_if_label
        self.axis = axis
        self.backend = dict(dtype=dtype, device=device)

    @classmethod
    def _make_vdm(
        cls, shape: tx.Sequence[int], controls: Tensor, order: int
    ) -> Tensor:
        return interpol.resize(
            controls, shape=shape, interpolation=order,
            prefilter=False
        )

    def make_vdm(
        self,
        shape: tx.Sequence[int],
        controls: tx.Optional[Tensor] = None
    ) -> Tensor:
        """Upsample the control points to the final full size

        Parameters
        ----------
        shape : list[int]
            Target shape
        controls : (C, *shape) tensor, default=`self.controls`
            Spline control points

        Returns
        -------
        vdm : (C, *fullshape) tensor
            Upampled voxel displacement field

        """
        if controls is None:
            controls = self.controls
        return self._make_vdm(shape, controls, self.order)

    @classmethod
    def _make_flow(cls, vdm: Tensor, axis: cct.VectorLike) -> Tensor:
        ndim = len(vdm.shape) - 1
        if isinstance(axis, int):
            axis_index = axis
            axis = [0] * ndim
            axis[axis_index] = 1
        axis = make_vector(axis, ndim, dtype=vdm.dtype, device=vdm.device)
        flow = vdm[..., None] * axis        # (C, *spatial, D)
        return torch.movedim(flow, -1, 1)   # (C, D, *spatial)

    def make_flow(self, vdm: Tensor) -> Tensor:
        """Make a flow field from the voxel displacement map"""
        return self._make_flow(vdm, self.axis)

    def xform(
        self, x: Tensor, /, *, args: Arguments = NoArguments()
    ) -> Returned:
        """Deform the input tensor

        Parameters
        ----------
        x : (C, *spatial) tensor
            Input tensor

        Returns
        -------
        out : [dict or list of] tensor
            The tensors returned by this function depend on the
            value of `self.returns`. See `ElasticTransform`.

        """
        x = x.to(**self.backend)

        # Get flow/vdm/controls tensors from `self` or `args`
        controls, vdm, flow = self.controls, self.vdm, self.flow
        if controls is not None and not torch.is_tensor(controls):
            controls = nested_get(args, controls)
        if vdm is not None and not torch.is_tensor(vdm):
            vdm = nested_get(args, vdm)
        if flow is not None and not torch.is_tensor(flow):
            flow = nested_get(args, flow)

        def _get_controls():
            if controls is not None:
                return cast_like(controls, x)
            raise ValueError('Controls requested but not stored.')

        def _get_vdm(controls=None):
            if vdm is not None:
                return cast_like(vdm, x)
            if controls is None:
                controls = _get_controls()
            return self.make_vdm(controls, x.shape[1:])

        def _get_flow(vdm=None, controls=None):
            if flow is not None:
                return cast_like(flow, x)
            if vdm is None:
                vdm = _get_vdm(controls)
            return self.make_flow(vdm)

        required = return_requires(self.returns)

        if required.intersection({'controls'}):
            controls = _get_controls()
        if required.intersection({'vdm'}):
            vdm = _get_vdm(controls)
        if required.intersection({'flow', 'output'}):
            flow = _get_flow(vdm, controls)

        y = None
        if 'output' in required:
            mode = 'bilinear'
            if not x.dtype.is_floating_point and self.nearest_if_label:
                mode ='nearest'
            y = warps.apply_flow(x[None], flow.movedim(1, -1), mode=mode)[0]

        return prepare_output(
            dict(input=x, output=y, flow=flow, vdm=vdm, controls=controls),
            self.returns
        )


class B0DistortionTransform(NonFinalTransform):
    """Elastic distortion along a single dimension.

    The number of control points is fixed but coefficients are
    randomly sampled from a uniform distribution.

    This is not a realistic B0 distortion model.
    For a more realistic model, see `RealisticB0DistortionTransform`.
    """

    Final = Next = ApplyB0DistortionTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        dmax: cct.NumberOrSequence[float] = 0.1,
        unit: tx.Literal['fov', 'vox'] = 'fov',
        shape: cct.NumberOrSequence[int] = 5,
        bound: cct.TorchBound = 'circular',
        order: int = 3,
        nearest_if_label: bool = True,
        axis: tx.Union[int, tx.Sequence[float]] = -1,
        *,
        dtype: tx.Optional[torch.dtype] = None,
        device: tx.Optional[cct.TorchDevice] = None,
        shared: cct.SharedT = True,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        dmax : float
            Max displacement
        unit : {'fov', 'vox'}
            Unit of `dmax`.
        shape : [list of] int
            Number of spline control points
        bound : {'zeros', 'border', 'reflection', 'circular'}
            Padding mode used for the deformed image.
        order : 1..7
            Order of the splines that encode the smooth deformation.
        nearest_if_label : bool
            By default, if a tensor has an integer data type, it
            is deformed using label-specific resampling (each unique
            label is extracted and resampled using linear interpolation,
            and an argmax output label map is computed on the fly).
            If `nearest_if_label=True`, the entire label map will be
            resampled at once using nearest-neighbour interpolation.
        axis : int | sequence[float]
            If int, the distortion is applied along this dimension.
            If sequence of floats, it is a unit vector that encodes the
            direction of the distortion in the voxel coordinate system.

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'vdm', 'controls'}

            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The flow field
            - 'vdm': The voxel displacement map (VDM)
            - 'controls': The control points of the VDM
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        if unit not in ('fov', 'vox'):
            raise ValueError(
                f'Unit must be one of {"fov", "vox"} but got "{unit}".'
            )
        if bound not in ('zeros', 'border', 'reflection'):
            raise ValueError(
                f'Bound must be one of {"zeros", "border", "reflection"} '
                f'but got "{bound}".'
            )
        self.dmax = dmax
        self.unit = unit
        self.bound = bound
        self.shape = shape
        self.order = order
        self.nearest_if_label = nearest_if_label
        self.axis = axis
        self.backend = dict()
        if dtype:
            self.backend["dtype"] = dtype
        if device:
            self.backend["device"] = device

    def make_final(
        self, x: Tensor, /, max_depth: int = inf,
        *, vdm: bool = True, flow: bool = True
    ) -> Transform:
        """
        Generate a deterministic transform with constant parameters

        Parameters
        ----------
        x : (C, *spatial) tensor
            Tensor to deform
        max_depth : int
            Maximum number of transforms to unroll
        vdm : bool
            Precompute the upsampled voxel displacement field
        flow : bool
            Precompute the flow field

        Returns
        -------
        xform : B0DistortionTransform.Final
            Final transform with parameters

            - `vdm : (C, spatial) tensor`, the upsampled VDM
            - `control : (C, *shape) tensor`, the spline control points

        """
        if max_depth == 0:
            return self
        batch, *fullshape = x.shape
        if 'channels' in self.shared:
            batch = 1
        ndim = len(fullshape)
        smallshape = ensure_list(self.shape, ndim)
        backend = dict(dtype=x.dtype, device=x.device)
        if self.backend.get("device", None):
            backend["device"] = self.backend["device"]
        if self.backend.get("dtype", None):
            backend["dtype"] = self.backend["dtype"]
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        dmax = make_vector(self.dmax, 1, **backend)[0]
        controls = torch.rand([batch, 1, *smallshape], **backend)
        controls = controls.sub_(0.5).mul_(2)
        if getattr(dmax, 'requires_grad', False):
            controls1 = controls.clone()
            controls.copy_(dmax*controls1)
        else:
            controls.mul_(dmax)
        if vdm or flow:
            vdm = self.Next._make_vdm(fullshape, controls, self.order)
        else:
            vdm = None
        if flow:
            flow = self.Next._make_flow(vdm, self.axis)
        else:
            flow = None
        return self.Next(
            flow, vdm, controls,
            self.axis, self.order, self.bound, self.nearest_if_label,
            **self.backend, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomB0DistortionTransform(FinalTransform):
    """Randomized elastic distortion along a single dimension."""

    Next = B0DistortionTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = ApplyB0DistortionTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        dmax: cct.SamplerOrBound[float] = 0.1,
        shape: cct.SamplerOrBound[int] = 5,
        unit: tx.Literal['fov', 'vox'] = 'fov',
        bound: cct.TorchBound = 'circular',
        order: int = 3,
        nearest_if_label: bool = True,
        axis: tx.Union[Sampler, int, tx.Sequence[float], None] = None,
        *,
        dtype: tx.Optional[torch.dtype] = None,
        device: tx.Optional[cct.TorchDevice] = None,
        shared: cct.SharedT = True,
        shared_vdm: tx.Optional[cct.SharedT] = None,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        dmax : Sampler | [list of] float
            Max displacement per dimension
        shape : Sampler | [list of] int
            Number of spline control points
        unit : {'fov', 'vox'}
            Unit of `dmax`.
        bound : {'zeros', 'border', 'reflection', 'circular'}
            Padding mode used for the deformed image.
        order : 1..7
            Order of the splines that encode the smooth deformation.
        nearest_if_label : bool
            By default, if a tensor has an integer data type, it
            is deformed using label-specific resampling (each unique
            label is extracted and resampled using linear interpolation,
            and an argmax output label map is computed on the fly).
            If `nearest_if_label=True`, the entire label map will be
            resampled at once using nearest-neighbour interpolation.
        axis : Sampler | int | sequence[float]
            If int, the distortion is applied along this dimension.
            If sequence of floats, it is a unit vector that encodes the
            direction of the distortion in the voxel coordinate system.

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'vdm', 'controls'}

            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The flow field
            - 'vdm': The voxel displacement map (VDM)
            - 'controls': The control points of the VDM
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Apply same transform to all images/channels
        shared_vdm : {'channels', 'tensors', 'channels+tensors', '', None} | bool
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """
        super().__init__(shared=shared, **kwargs)
        self.dmax = Uniform.make(make_range(0, dmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.unit = unit
        self.bound = bound
        self.order = order
        self.nearest_if_label = nearest_if_label
        self.axis = axis
        self.shared_vdm = shared_vdm
        self.backend = dict()
        if dtype:
            self.backend["dtype"] = dtype
        if device:
            self.backend["device"] = device


    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        ndim = x.ndim - 1

        dmax, shape, order, axis = self.dmax, self.shape, self.order, self.axis
        if isinstance(dmax, Sampler):
            dmax = dmax()
        if isinstance(shape, Sampler):
            shape = shape(ndim)
        if isinstance(order, Sampler):
            order = order()
        if isinstance(axis, Sampler):
            axis = axis()
        shared_vdm = self.shared_vdm
        if shared_vdm is None:
            shared_vdm = self.shared

        return B0DistortionTransform(
            dmax=dmax, shape=shape, order=order,
            unit=self.unit, bound=self.bound, shared=shared_vdm,
            nearest_if_label=self.nearest_if_label, axis=axis,
            **self.backend,
            **self.get_prm(),
        ).make_final(x, max_depth-1)


class GradientEchoTransform(FinalTransform):
    """Spoiled Gradient Echo forward model"""

    Parameters = namedtuple('Parameters', ['PD', 'T1', 'T2', 'MT', 'B1'])

    def __init__(
        self,
        tr: float = 25e-3,
        te: float = 7e-3,
        alpha: float = 20,
        pd: tx.Optional[float] = None,
        t1: tx.Optional[float] = None,
        t2: tx.Optional[float] = None,
        b1: tx.Optional[float] = 1,
        mt: tx.Optional[float] = 0,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        tr : float
            Repetition time, in sec
        te : float
            Echo time, in sec
        alpha : float
            Nominal flip angle, in degree
        pd : float | None
            Proton density (PD).
            If None, the first input channel is PD.
        t1 : float | None
            Longitudinal relaxation time (T1), in sec.
            If None, the second input channel is T1.
        t2 : float | None
            Apparent transverse relaxation time (T2*), in sec.
            If None, the third input channel is T2*.
        b1 : float | None
            Transmit efficiency (B1+). `1` means 100% efficiency.
            If None, the fourth input channel is B1+.
        mt : float | None
            Magnetization transfer saturation (MTsat).
            If None, the fifth input channel is MTsat.

        """
        super().__init__(**kwargs)
        self.te = te
        self.tr = tr
        self.alpha = alpha
        self.pd = pd
        self.t1 = t1
        self.t2 = t2
        self.b1 = b1
        self.mt = mt

    def get_parameters(
        self, x: Tensor
    ) -> Parameters:
        """Assign each input channel to the appropriate parameter."""
        if self.pd is None:
            pd, x = x[:1], x[1:]
        else:
            pd = self.pd
        if self.t1 is None:
            t1, x = x[:1], x[1:]
        else:
            t1 = self.t1
        if self.t2 is None:
            t2, x = x[:1], x[1:]
        else:
            t2 = self.t2
        if self.mt is None:
            mt, x = x[:1], x[1:]
        else:
            mt = self.mt
        if self.b1 is None:
            b1 = x
        else:
            b1 = self.b1

        return self.Parameters(pd, t1, t2, mt, b1)

    def xform(self, x: Tensor) -> Returned:
        """
        Parameters
        ----------
        x : (K, *spatial)
            Parameters that were not set during class instantiation
            (i.e., whose value was `None`), in the order:

                - pd: Proton density
                - t1: Longitudinal relaxation time (T1), in sec.
                - t2: Apparent transverse relaxation time (T2*), in sec.
                - b1: Transmit efficiency (B1+). `1` means 100% efficiency.
                - mt: Magnetization transfer saturation (MTsat).
        """
        pd, t1, t2, mt, b1 = self.get_parameters(x)
        alpha = (math.pi * self.alpha / 180) * b1
        if torch.is_tensor(alpha):
            sinalpha, cosalpha = alpha.sin(), alpha.cos()
        else:
            sinalpha, cosalpha = math.sin(alpha), math.cos(alpha)

        e1 = (-self.tr) / t1
        e2 = (-self.te) / t2
        e1 = exp_(e1)
        e2 = exp_(e2)
        s = pd * sinalpha * e2 * (1 - mt) * (1 - e1)
        s = div_(s, (1 - (1 - mt) * cosalpha * e1))

        return prepare_output(
            dict(input=x, pd=pd, t1=t1, t2=t2, b1=b1, mt=mt,
                 te=self.te, tr=self.tr, alpha=self.alpha, output=s),
            self.returns)


class ReturnGradientEchoParameters(ReturnValueTransform):
    """Store parameter maps"""

    def __init__(self, param, **kwargs):
        super().__init__(param, **kwargs, value_name='param')

    def xform(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        return self.param.to(x.device, dtype)


class GMMGradientEchoTransform(FinalTransform):
    """Apply GRE forward model to parameters, then mask"""

    def __init__(
        self,
        prm: ReturnGradientEchoParameters,
        fwd: GradientEchoTransform,
        mask: tx.Optional[Transform] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        prm : ReturnGradientEchoParameters
            Transform that returns the parameter maps required by `fwd`
        fwd : GradientEchoTransform
            Forward model that generates the GRE image from the parameters
        mask : Transform, optional
            Transform that applies a mask to the output of `fwd`.
        """
        super().__init__(**kwargs)
        self.prm = prm
        self.fwd = fwd
        self.mask = mask

    def xform(self, x: Tensor) -> Returned:
        y = self.prm(x)
        y = self.fwd(y)
        if self.mask is not None:
            out = returns_find('output', y, self.fwd.returns)
            y = returns_update(self.mask(out), 'output', y, self.fwd.returns)
        return y


class RandomGMMGradientEchoTransform(NonFinalTransform):
    """
    Generate a Spoiled Gradient Echo image from synthetic PD/T1/T2 maps.
    """

    Final = Next = GMMGradientEchoTransform
    """The transform type returned by `make_final`."""

    Parameters = namedtuple(
        'Parameters',
        ['tr', 'te', 'alpha', 'pd', 't1', 't2', 'mt', 'b1', 'sigma', 'fwhm']
    )

    def __init__(
        self,
        tr: cct.SamplerOrBound[float] = 50e-3,
        te: cct.SamplerOrBound[float] = 50e-3,
        alpha: cct.SamplerOrBound[float] = 90,
        pd: cct.SamplerOrBound[float] = 1,
        t1: cct.SamplerOrBound[float] = 10,
        t2: cct.SamplerOrBound[float] = 100,
        mt: cct.SamplerOrBound[float] = 0.1,
        b1: Transform = RandomMulFieldTransform(vmax=1.5),
        sigma: cct.SamplerOrBound[float] = 0.2,
        fwhm: cct.SamplerOrBound[float] = 2,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        tr : Sampler | float
            Random sampler for repetition time, or upper bound
        te : Sampler | float
            Random sampler for echo time, or upper bound
        alpha : Sampler | float
            Random sampler for nominal flip angle, or upper bound
        pd : Sampler | float
            Random sampler for proton density, or upper bound
        t1 : Sampler | float
            Random sampler for longitudinal relaxation, or upper bound
        t2 : Sampler | float
            Random sampler for apparent transverse relaxation, or upper bound
        mt : Sampler | float
            Random sampler for magnetization transfer saturation, or upper bound
        b1 : Transform
            A transformation that samples a smooth multiplicative field
        sigma : Sampler | float
            Random sampler for intra-class standard deviation (in percent), or upper bound
        fwhm : Sampler | float
            Random sampler for intra-class smoothing (in voxels), or upper bound
        """  # noqa: E501
        super().__init__(**kwargs)
        self.tr = Uniform.make(make_range(0, tr))
        self.te = Uniform.make(make_range(0, te))
        self.alpha = Uniform.make(make_range(0, alpha))
        self.pd = Uniform.make(make_range(0, pd))
        self.t1 = Uniform.make(make_range(0, t1))
        self.t2 = Uniform.make(make_range(0, t2))
        self.mt = Uniform.make(make_range(0, mt))
        self.sigma = 1 + Uniform.make(make_range(0, sigma))
        self.fwhm = Uniform.make(make_range(0, fwhm))
        self.b1 = b1

    def get_parameters(self, x: Tensor) -> Parameters:
        """Sample each parameter."""
        dtype = (x.dtype if x.dtype.is_floating_point
                 else torch.get_default_dtype())
        backend = dict(dtype=dtype, device=x.device)
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        tr = self.tr() if isinstance(self.tr, Sampler) else self.tr
        te = self.te() if isinstance(self.te, Sampler) else self.te
        alpha = self.alpha() if isinstance(self.alpha, Sampler) else self.alpha
        pd = make_vector(
            self.pd(n) if isinstance(self.pd, Sampler) else self.pd,
            n, **backend
        )
        t1 = make_vector(
            self.t1(n) if isinstance(self.t1, Sampler) else self.t1,
            n, **backend
        )
        t2 = make_vector(
            self.t2(n) if isinstance(self.t2, Sampler) else self.t2,
            n, **backend
        )
        mt = make_vector(
            self.mt(n) if isinstance(self.mt, Sampler) else self.mt,
            n, **backend
        )
        sigma = make_vector(
            self.sigma(n*4) if isinstance(self.sigma, Sampler) else self.sigma,
            n*4, **backend
        )
        fwhm = make_vector(
            self.fwhm(n*4) if isinstance(self.fwhm, Sampler) else self.fwhm,
            n*4, **backend
        )
        if self.b1 is not None:
            b1 = self.b1
            if isinstance(b1, Transform):
                b1 = b1.make_final(x)
                b1 = b1(x.new_ones([], dtype=dtype).expand(x.shape))
            elif isinstance(self.b1, Sampler):
                b1 = b1()
        else:
            b1 = 1

        return self.Parameters(tr, te, alpha, pd, t1, t2, mt, b1, sigma, fwhm)

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        tr, te, alpha, pd, t1, t2, mt, b1, sigma, fwhm = self.get_parameters(x)

        def logit(x):
            return (x).log() - (1-x).log()

        def sigmoid_(x):
            if x.requires_grad:
                return x.neg().exp().add(1).reciprocal()
            else:
                return x.neg_().exp_().add_(1).reciprocal_()

        logpd = pd.log()
        logt1 = t1.log()
        logt2 = t2.log()
        logmt = logit(mt)
        logsigma = sigma.log()

        nb1 = len(b1) if (torch.is_tensor(b1) and b1.ndim > 0) else 1

        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        y = x.new_zeros([4 + nb1, *x.shape[1:]], dtype=dtype)
        # PD
        y[0] = exp_(GaussianMixtureTransform(
            mu=logpd, sigma=logsigma[:n], fwhm=fwhm[:n],
            background=0, dtype=dtype
        )(x)).squeeze(0)
        # T1
        y[1] = exp_(GaussianMixtureTransform(
            mu=logt1, sigma=logsigma[n:2*n], fwhm=fwhm[n:2*n],
            background=0, dtype=dtype
        )(x)).squeeze(0)
        # T2*
        y[2] = exp_(GaussianMixtureTransform(
            mu=logt2, sigma=logsigma[2*n:3*n], fwhm=fwhm[2*n:3*n],
            background=0, dtype=dtype
        )(x)).squeeze(0)
        # MT
        y[3] = sigmoid_(GaussianMixtureTransform(
            mu=logmt, sigma=logsigma[3*n:4*n], fwhm=fwhm[3*n:4*n],
            background=0, dtype=dtype
        )(x)).squeeze(0)
        # B1
        y[4:].copy_(b1)  # NOTE: y[4:] = b1 breaks the graph in torch 1.11

        # GRE forward mode
        mask = (1 - x[0]) if x.dtype.is_floating_point else (x != 0)
        return self.Next(
            ReturnGradientEchoParameters(y),
            GradientEchoTransform(tr, te, alpha, b1=None, mt=None,
                                  **self.get_prm()),
            MulValueTransform(mask),
        ).make_final(x, max_depth-1)
