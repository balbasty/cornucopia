__all__ = [
    'RandomSusceptibilityMixtureTransform',
    'SusceptibilityToFieldmapTransform',
    'ShimTransform',
    'RandomShimTransform',
    'HertzToPhaseTransform',
    'GradientEchoTransform',
    'RandomGMMGradientEchoTransform',
]

import torch
import math
from .base import Transform, RandomizedTransform, prepare_output
from .labels import GaussianMixtureTransform
from .intensity import RandomMultFieldTransform
from .random import Sampler, Uniform, make_range
from .utils.py import ensure_list, make_vector
from .utils import b0


class RandomSusceptibilityMixtureTransform(RandomizedTransform):
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
        super().__init__(
            GaussianMixtureTransform,
            dict(mu_tissue=Uniform.make(make_range(0, mu_tissue)),
                 sigma_tissue=Uniform.make(make_range(0, sigma_tissue)),
                 mu_bone=Uniform.make(make_range(0, mu_bone)),
                 sigma_bone=Uniform.make(make_range(0, sigma_bone)),
                 fwhm=Uniform.make(make_range(0, fwhm)),
                 label_air=label_air,
                 label_bone=label_bone,
                 **kwargs),
            shared=shared)

    def get_parameters(self, x):
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        label_air = ensure_list(self.label_air)
        label_bone = ensure_list(self.label_bone or [])
        label_tissue = [i for i in range(n) if i not in label_bone + label_air]

        sample = dict(self.sample)
        self.sample['mu'] = [0] * n
        self.sample['sigma'] = [0] * n

        if isinstance(self.sample['mu_tissue'], Sampler):
            for i in label_tissue:
                self.sample['mu'][i] = self.sample['mu_tissue']()
        if isinstance(self.sample['sigma_tissue'], Sampler):
            for i in label_tissue:
                self.sample['sigma'][i] = self.sample['sigma_tissue']()
        if isinstance(self.sample['mu_bone'], Sampler):
            for i in label_bone:
                self.sample['mu'][i] = self.sample['mu_bone']()
        if isinstance(self.sample['sigma_bone'], Sampler):
            for i in label_bone:
                self.sample['sigma'][i] = self.sample['sigma_bone']()

        out = super().get_parameters(x)
        self.sample = sample
        return out


class SusceptibilityToFieldmapTransform(Transform):

    def __int__(self, dim=-1, field_strength=3, larmor=42.576E6,
                s0=0.4, s1=-9.5, voxel_size=1, mask_air=False, **kwargs):
        """

        Parameters
        ----------
        dim : int
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
        self.dim = dim
        self.field_strength = field_strength
        self.larmor = larmor
        self.s0 = s0
        self.s1 = s1
        self.mask_air = mask_air

    def apply_transform(self, x, parameters=None):
        dim = 1 + ((x.ndim - 1 + self.dim) if self.dim < 0 else self.dim)
        field = b0.chi_to_fieldmap(x, zdim=dim, dim=x.ndim-1,
                                   s0=self.s0, s1=self.s1, vx=self.voxel_size)
        if self.mask_air:
            field.masked_fill_(x == 0, 0)
        if self.field_strength:
            field = b0.ppm_to_hz(field, self.field_strength, self.larmor)
        return prepare_output(dict(input=x, output=field, fieldmap=field),
                              self.returns)


class ShimTransform(Transform):

    def __init__(self, max_order=2, lam_abs=1, lam_grad=10, mask=True, **kwargs):
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
        """
        super().__init__(**kwargs)
        self.max_order = max_order
        self.lam_abs = lam_abs
        self.lam_grad = lam_grad
        self.mask = mask

    def get_parameters(self, x):
        mask = (x != 0) if self.mask else None
        shim = b0.shim(x, max_order=self.max_order, dim=x.ndim-1, mask=mask,
                       lam_abs=self.lam_abs, lam_grad=self.lam_grad,
                       returns='correction')
        return shim

    def apply_transform(self, x, parameters):
        shim = parameters
        y = x - shim
        return prepare_output(
            dict(input=x, output=y, shim=shim), self.results)


class RandomShimTransform(Transform):
    """
    Sample a random (imperfect) shim.

    This function randomly samples the coefficients of a field encoded
    by spherical harmonics.
    """

    def __init__(self, coefficients=5, max_order=2, **kwargs):
        """
        Parameters
        ----------
        coefficients : int or Sampler
            Sampler for spherical harmonics coefficients (or upper bound)
        max_order : int or Sampler
            Sampler for spherical harminocs order (or upper bound)
        """
        super().__init__(**kwargs)
        self.coefficients = Uniform.make(make_range(0, coefficients))
        self.max_order = Uniform.make(make_range(0, max_order))

    def get_parameters(self, x):
        ndim = x.ndim - 1
        batch = len(x)

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

        shim = torch.zeros_like(x).movedim(0, -1)
        basis = b0.yield_spherical_harmonics(
            x.shape[1:], order, dtype=x.dtype, device=x.device)
        for c, B in zip(coefficients, basis):
            c = make_vector(c, batch, dtype=x.dtype, device=x.device)
            shim.addcmul_(B[..., None], c)
        shim = shim.movedim(-1, 0)
        return shim

    def apply_transform(self, x, parameters):
        shim = parameters
        y = x - shim
        return prepare_output(
            dict(input=x, output=y, shim=shim), self.results)


class HertzToPhaseTransform(Transform):
    """Converts a ΔB0 field (in Hz) into a Phase shift field Δφ (in rad)"""

    def __int__(self, te, **kwargs):
        """

        Parameters
        ----------
        te : float
            Echo time, in sec.

        """
        super().__init__(**kwargs)
        self.te = te

    def apply_transform(self, x, parameters=None):
        y = (2 * math.pi * self.te) * x
        return prepare_output(dict(input=x, output=y), self.returns)


class GradientEchoTransform(Transform):
    """Spoiled Gradient Echo forward model"""

    def __init__(self, tr=25e-3, te=7e-3, alpha=20, pd=None, t1=None, t2=None, b1=1, mt=0,
                 *, shared='channels', **kwargs):
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
            Transmit efficiency (B1+).
            If None, the fourth input channel is B1+.
        mt : float, optional
            Magnetization transfer saturation (MTsat).
            If None, the fifth input channel is MTsat.

        """
        super().__init__(shared=shared, **kwargs)
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

    def apply_transform(self, x, parameters):
        pd, t1, t2, b1, mt = parameters
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


class RandomGMMGradientEchoTransform(Transform):
    """
    Generate a SpoiledGradientEcho image from synthetic PD/T1/T2 maps.
    """

    def __init__(self, tr=50e-3, te=50e-3, alpha=90, pd=1, t1=10, t2=100, mt=0.1,
                 b1=RandomMultFieldTransform(vmax=1.5),
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
        """
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
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        tr = self.tr() if isinstance(self.tr, Sampler) else self.tr
        te = self.te() if isinstance(self.te, Sampler) else self.te
        alpha = self.alpha() if isinstance(self.alpha, Sampler) else self.alpha
        pd = ensure_list(self.pd(n) if isinstance(self.pd, Sampler) else self.pd, n)
        t1 = ensure_list(self.t1(n) if isinstance(self.t1, Sampler) else self.t1, n)
        t2 = ensure_list(self.t2(n) if isinstance(self.t2, Sampler) else self.t2, n)
        mt = ensure_list(self.mt(n) if isinstance(self.mt, Sampler) else self.mt, n)
        b1 = self.b1.get_parameters(x) if self.b1 else 1
        while isinstance(b1, Transform):
            b1 = b1.get_parameters(x)
        sigma = ensure_list(self.sigma(n*4) if isinstance(self.sigma, Sampler) else self.sigma, n*4)
        fwhm = ensure_list(self.fwhm(n*4) if isinstance(self.fwhm, Sampler) else self.fwhm, n*4)
        return tr, te, alpha, pd, t1, t2, mt, b1, sigma, fwhm

    def apply_transform(self, x, parameters):
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        tr, te, alpha, pd, t1, t2, mt, b1, sigma, fwhm = parameters

        logit = lambda x: math.log(x) - math.log(1-x)
        sigmoid_ = lambda x: x.neg_().exp_().add_(1).reciprocal_()

        logpd = list(map(math.log, pd))
        logt1 = list(map(math.log, t1))
        logt2 = list(map(math.log, t2))
        logmt = list(map(logit, mt))
        logsigma = list(map(math.log, sigma))

        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        y = x.new_zeros([5, *x.shape[1:]], dtype=dtype)
        # PD
        y[0] = GaussianMixtureTransform(
            mu=logpd, sigma=logsigma[:n], fwhm=fwhm[:n], background=0, dtype=dtype
        )(x).exp_().squeeze(0)
        # T1
        y[1] = GaussianMixtureTransform(
            mu=logt1, sigma=logsigma[n:2*n], fwhm=fwhm[n:2*n], background=0, dtype=dtype
        )(x).exp_().squeeze(0)
        # T2*
        y[2] = GaussianMixtureTransform(
            mu=logt2, sigma=logsigma[2*n:3*n], fwhm=fwhm[2*n:3*n], background=0, dtype=dtype
        )(x).exp_().squeeze(0)
        # B1
        y[4:] = b1
        # MT
        y[3] = sigmoid_(GaussianMixtureTransform(
            mu=logmt, sigma=logsigma[3*n:4*n], fwhm=fwhm[3*n:4*n], background=0, dtype=dtype
        )(x)).squeeze(0)

        # GRE forward model
        y = GradientEchoTransform(tr, te, alpha, b1=None, mt=None)(y)
        y *= (1 - x[0]) if x.dtype.is_floating_point else (x != 0)

        return prepare_output(
            dict(input=x, output=y, tr=tr, te=te, alpha=alpha,
                 pd=pd, t1=t1, t2=t2, mt=mt, b1=b1), self.returns)

