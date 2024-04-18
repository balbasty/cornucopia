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
    'GradientEchoTransform',
    'RandomGMMGradientEchoTransform',
]

import torch
import math
from .base import FinalTransform, NonFinalTransform
from .baseutils import prepare_output, returns_find
from .labels import GaussianMixtureTransform
from .intensity import (
    RandomMulFieldTransform, AddValueTransform, MulValueTransform
)
from .random import Sampler, Uniform, RandInt, make_range
from .utils.py import ensure_list, make_vector
from .utils import b0


class RandomSusceptibilityMixtureTransform(NonFinalTransform):
    """
    A RandomGaussianMixtureTransform tailored to susceptibility maps.

    This transform returns a delta susceptibility map (with respect to
    air), in ppm.
    """

    def __init__(self,
                 mu_tissue=Uniform(9, 10),
                 sigma_tissue=0.01,
                 mu_bone=Uniform(12, 13),
                 sigma_bone=0.1,
                 fwhm=2,
                 label_air=0,
                 label_bone=None,
                 dtype=None,
                 *, shared='channels', **kwargs):
        """

        Parameters
        ----------
        mu_tissue : float or Sampler
            Distribution of negative susceptibility offsets (in ppm)
            of soft tissues with respect to air.
            Will be negated (air susceptibility is larger than all tissues).
            If float: upper bound.
        sigma_tissue : float or Sampler
            Standard deviation of susceptibility offsets, within class.
            If float: uper bound.
        mu_bone : float or Sampler
            Distribution of negative susceptibility offsets (in ppm)
            of hard tissues with respect to air.
            Will be negated (air susceptibility is larger than all tissues).
            If float: upper bound.
        sigma_bone: float or Sampler
            Standard deviation of susceptibility offsets, within class.
            If float: upper bound.
        fwhm : Sampler or [list of] float
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

    def make_final(self, x, max_depth=float('inf')):
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
        elif isinstance(self.mu, (list, tuple)):
            for i in label_tissue:
                mu[i] = self.mu_tissue[i]
        else:
            for i in label_tissue:
                mu[i] = self.mu_tissue

        if isinstance(self.sigma_tissue, Sampler):
            for i in label_tissue:
                sigma[i] = self.sigma_tissue()
        elif isinstance(self.mu, (list, tuple)):
            for i in label_tissue:
                sigma[i] = self.sigma_tissue[i]
        else:
            for i in label_tissue:
                sigma[i] = self.sigma_tissue

        if isinstance(self.mu_bone, Sampler):
            for i in label_bone:
                mu[i] = self.mu_bone()
        elif isinstance(self.mu, (list, tuple)):
            for i in label_bone:
                mu[i] = self.mu_bone[i]
        else:
            for i in label_tissue:
                mu[i] = self.mu_tissue

        if isinstance(self.sigma_bone, Sampler):
            for i in label_bone:
                sigma[i] = self.sigma_bone()
        elif isinstance(self.mu, (list, tuple)):
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

        return GaussianMixtureTransform(
            mu, sigma, fwhm, dtype=self.dtype, **self.get_prm()
        ).make_final(x, max_depth-1)


class SusceptibilityToFieldmapTransform(FinalTransform):
    """
    Convert a susceptibiity map (in ppm) into a field map (in Hz)
    """

    def __init__(self, axis=-1, field_strength=3, larmor=42.576E6,
                 s0=0.4, s1=-9.5, voxel_size=1, mask_air=False, **kwargs):
        """

        Parameters
        ----------
        axis : int
            Readout dimension
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
            reuslting fieldmap.

        """
        super().__init__(**kwargs)
        self.axis = axis
        self.field_strength = field_strength
        self.larmor = larmor
        self.s0 = s0
        self.s1 = s1
        self.mask_air = mask_air
        self.voxel_size = voxel_size

    def apply(self, x):
        axis = 1 + ((x.ndim - 1 + self.axis) if self.axis < 0 else self.axis)
        field = b0.chi_to_fieldmap(x, zdim=axis, dim=x.ndim-1,
                                   s0=self.s0, s1=self.s1, vx=self.voxel_size)
        if self.mask_air:
            field.masked_fill_(x == 0, 0)
        if self.field_strength:
            field = b0.ppm_to_hz(field, self.field_strength, self.larmor)
        return prepare_output(dict(input=x, output=field, fieldmap=field),
                              self.returns)


class ShimTransform(FinalTransform):

    def __init__(self, linear=None, quadratic=None, isocenter=None, **kwargs):
        """
        Parameters
        ----------
        linear : (3 or 1) tensor or list of float
            Linear components (3D: [XY, XZ, YX], 2D: [XY])
        quadratic : (2 or 1) tensor or list of float
            Quadratic components (3D: [XXYY, XXZZ], 2D: [XXYY])
        isocenter : (3 or 2) tensor or list of float
            Coordinates of the isocenter, in voxels

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'shim'}
        """
        super().__init__(**kwargs)
        self.linear = linear
        self.quadratic = quadratic
        self.isocenter = isocenter

    def apply(self, x):
        ndim = x.ndim - 1
        order = 1 if self.quadratic is None else 1
        nb_lin = 3 if ndim == 3 else 1
        nb_quad = 2 if ndim == 3 else 1
        if self.linear is None:
            linear = [0] * nb_lin
        else:
            linear = make_vector(self.linear, nb_lin).tolist()
        if self.quadratic is None:
            quadratic = [0] * nb_quad
        else:
            quadratic = make_vector(self.quadratic, nb_quad).tolist()
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
            dict(input=x, outut=y, shim=shim), self.returns
        )


class OptimalShimTransform(NonFinalTransform):
    """
    Compute a linear combination of spherical harmonics that flattens the
    input field map
    """

    def __init__(self, max_order=2, lam_abs=1, lam_grad=10, mask=True,
                 **kwargs):
        """

        Parameters
        ----------
        max_order : int
            Maximum order of spherical basis functions
        lam_abs : float
            Regularization factor for absolute values
        lam_grad : float
            Regularization factor for first order gradients
        mask : bool
            Mask zeros/NaNs from objective functions

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'shim'}
        """
        super().__init__(**kwargs)
        self.max_order = max_order
        self.lam_abs = lam_abs
        self.lam_grad = lam_grad
        self.mask = mask

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)
        mask = (x != 0) if self.mask else None
        shim = b0.shim(x, max_order=self.max_order, dim=x.ndim-1, mask=mask,
                       lam_abs=self.lam_abs, lam_grad=self.lam_grad,
                       returns='correction').neg_()
        return AddValueTransform(
            shim, value_name='shim', **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomShimTransform(NonFinalTransform):
    """
    Sample a random (imperfect) shim.

    This function randomly samples the coefficients of a field encoded
    by spherical harmonics.
    """

    def __init__(self, coefficients=5, max_order=2, shared=False, **kwargs):
        """
        Parameters
        ----------
        coefficients : int or Sampler
            Sampler for spherical harmonics coefficients (or upper bound)
        max_order : int or Sampler
            Sampler for spherical harminocs order (or upper bound)

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'shim'}
        shared : {'channels', 'tensors', 'channels+tensors', ''}
        """
        super().__init__(shared=shared, **kwargs)
        self.coefficients = Uniform.make(make_range(0, coefficients))
        self.max_order = RandInt.make(make_range(0, max_order))

    def make_final(self, x, max_depth=float('inf')):
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
            lin, quad, **self.get_prm()
        ).make_final(x, max_depth-1)


class HertzToPhaseTransform(FinalTransform):
    """Converts a ΔB0 field (in Hz) into a Phase shift field Δφ (in rad)"""

    def __init__(self, te=0, **kwargs):
        """

        Parameters
        ----------
        te : float
            Echo time, in sec.

        """
        super().__init__(**kwargs)
        self.te = te

    def apply(self, x):
        return (2 * math.pi * self.te) * x


class GradientEchoTransform(FinalTransform):
    """Spoiled Gradient Echo forward model"""

    def __init__(self, tr=25e-3, te=7e-3, alpha=20,
                 pd=None, t1=None, t2=None, b1=1, mt=0,
                 **kwargs):
        """

        Parameters
        ----------
        tr : float
            Repetition time, in sec
        te : float
            Echo time, in sec
        alpha : float
            Nominal flip angle, in degree
        pd : float, optional
            Proton density (PD).
            If None, the first input channel is PD.
        t1 : float, optional
            Longitudinal relaxation time (T1), in sec.
            If None, the second input channel is T1.
        t2 : float, optional
            Apparent transverse relaxation time (T2*), in sec.
            If None, the third input channel is T2*.
        b1 : float, optional
            Transmit efficiency (B1+). `1` means 100% efficiency.
            If None, the fourth input channel is B1+.
        mt : float, optional
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

    def get_parameters(self, x):
        x = x.unbind(0)
        if self.pd is None:
            pd, *x = x
            pd = pd[None]
        else:
            pd = self.pd
        if self.t1 is None:
            t1, *x = x
            t1 = t1[None]
        else:
            t1 = self.t1
        if self.t2 is None:
            t2, *x = x
            t2 = t2[None]
        else:
            t2 = self.t2
        if self.b1 is None:
            b1, *x = x
            b1 = b1[None]
        else:
            b1 = self.b1
        if self.mt is None:
            mt, *x = x
            mt = mt[None]
        else:
            mt = self.mt

        return pd, t1, t2, b1, mt

    def apply(self, x):
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
        pd, t1, t2, b1, mt = self.get_parameters(x)
        alpha = (math.pi * self.alpha / 180) * b1
        if torch.is_tensor(alpha):
            sinalpha, cosalpha = alpha.sin(), alpha.cos()
        else:
            sinalpha, cosalpha = math.sin(alpha), math.cos(alpha)

        e1 = (-self.tr) / t1
        e2 = (-self.te) / t2
        e1 = e1.exp() if torch.is_tensor(e1) else math.exp(e1)
        e2 = e2.exp() if torch.is_tensor(e2) else math.exp(e2)
        s = pd * sinalpha * e2 * (1 - mt) * (1 - e1)
        s /= (1 - (1 - mt) * cosalpha * e1)

        return prepare_output(
            dict(input=x, pd=pd, t1=t1, t2=t2, b1=b1, mt=mt,
                 te=self.te, tr=self.tr, alpha=self.alpha, output=s),
            self.returns)


class RandomGMMGradientEchoTransform(NonFinalTransform):
    """
    Generate a Spoiled Gradient Echo image from synthetic PD/T1/T2 maps.
    """

    def __init__(self, tr=50e-3, te=50e-3, alpha=90,
                 pd=1, t1=10, t2=100, mt=0.1,
                 b1=RandomMulFieldTransform(vmax=1.5),
                 sigma=0.2, fwhm=2, **kwargs):
        """
        Parameters
        ----------
        tr : Sampler or float
            Random sampler for repetition time, or upper bound
        te : Sampler or float
            Random sampler for echo time, or upper bound
        alpha : Sampler or float
            Random sampler for nominal flip angle, or upper bound
        pd : Sampler or float
            Random sampler for proton density, or upper bound
        t1 : Sampler or float
            Random sampler for longitudinal relaxation, or upper bound
        t2 : Sampler or float
            Random sampler for apparent transverse relaxation, or upper bound
        mt : Sampler or float
            Random sampler for magnetization transfer saturation, or upper bound
        b1 : Transform
            A transformation that samples a smooth multiplicative field
        sigma : Sampler or float
            Random sampler for intra-class standard deviation (in percent), or upper bound
        fwhm : Sampler or float
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
        self.b1 = b1
        self.sigma = 1 + Uniform.make(make_range(0, sigma))
        self.fwhm = Uniform.make(make_range(0, fwhm))

    def get_parameters(self, x):
        dtype = (x.dtype if x.dtype.is_floating_point
                 else torch.get_default_dtype())
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        tr = self.tr() if isinstance(self.tr, Sampler) else self.tr
        te = self.te() if isinstance(self.te, Sampler) else self.te
        alpha = self.alpha() if isinstance(self.alpha, Sampler) else self.alpha
        pd = ensure_list(
            self.pd(n) if isinstance(self.pd, Sampler) else self.pd,
            n
        )
        t1 = ensure_list(
            self.t1(n) if isinstance(self.t1, Sampler) else self.t1,
            n
        )
        t2 = ensure_list(
            self.t2(n) if isinstance(self.t2, Sampler) else self.t2,
            n
        )
        mt = ensure_list(
            self.mt(n) if isinstance(self.mt, Sampler) else self.mt,
            n
        )
        sigma = ensure_list(
            self.sigma(n*4) if isinstance(self.sigma, Sampler) else self.sigma,
            n*4
        )
        fwhm = ensure_list(
            self.fwhm(n*4) if isinstance(self.fwhm, Sampler) else self.fwhm,
            n*4
        )
        if self.b1:
            b1 = self.b1.make_final(x)
            b1 = b1(x.new_ones([], dtype=dtype).expand(x.shape))
        else:
            b1 = 1
        return tr, te, alpha, pd, t1, t2, mt, b1, sigma, fwhm

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        tr, te, alpha, pd, t1, t2, mt, b1, sigma, fwhm = self.get_parameters(x)

        def logit(x):
            return math.log(x) - math.log(1-x)

        def sigmoid_(x):
            return x.neg_().exp_().add_(1).reciprocal_()

        logpd = list(map(math.log, pd))
        logt1 = list(map(math.log, t1))
        logt2 = list(map(math.log, t2))
        logmt = list(map(logit, mt))
        logsigma = list(map(math.log, sigma))

        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        y = x.new_zeros([5, *x.shape[1:]], dtype=dtype)
        # PD
        y[0] = GaussianMixtureTransform(
            mu=logpd, sigma=logsigma[:n], fwhm=fwhm[:n],
            background=0, dtype=dtype
        )(x).exp_().squeeze(0)
        # T1
        y[1] = GaussianMixtureTransform(
            mu=logt1, sigma=logsigma[n:2*n], fwhm=fwhm[n:2*n],
            background=0, dtype=dtype
        )(x).exp_().squeeze(0)
        # T2*
        y[2] = GaussianMixtureTransform(
            mu=logt2, sigma=logsigma[2*n:3*n], fwhm=fwhm[2*n:3*n],
            background=0, dtype=dtype
        )(x).exp_().squeeze(0)
        # B1
        y[4:] = b1
        # MT
        y[3] = sigmoid_(GaussianMixtureTransform(
            mu=logmt, sigma=logsigma[3*n:4*n], fwhm=fwhm[3*n:4*n],
            background=0, dtype=dtype
        )(x)).squeeze(0)

        # GRE forward mode
        mask = (1 - x[0]) if x.dtype.is_floating_point else (x != 0)
        return self.Final(
            self.GREParameters(y),
            GradientEchoTransform(tr, te, alpha, b1=None, mt=None,
                                  **self.get_prm()),
            MulValueTransform(mask),
        )

    class GREParameters(FinalTransform):
        """Store parameter maps"""
        def __init__(self, param, **kwargs):
            super().__init__(**kwargs)
            self.param = param

        def apply(self, x):
            dtype = x.dtype
            if not dtype.is_floating_point:
                dtype = torch.get_default_dtype()
            return self.param.to(x.device, dtype)

    class Final(FinalTransform):
        """Apply GRE forward model to parameters, then mask"""
        def __init__(self, prm, fwd, mask, **kwargs):
            super().__init__(**kwargs)
            self.prm = prm
            self.fwd = fwd
            self.mask = mask

        def apply(self, x):
            y = self.prm(x)
            y = self.fwd(y)

            out = returns_find('output', y, self.fwd.returns)
            if out is not None:
                out.copy_(self.mask(out))
            return y
