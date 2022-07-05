import torch
from .random import Uniform, Sampler
from .base import Transform, RandomizedTransform
from .utils.conv import smoothnd
from .utils.py import ensure_list


__all__ = ['OneHotTransform']


class OneHotTransform(Transform):
    """Transform a volume of integer labels into a one-hot representation"""

    def __init__(self, label_map=None, label_ref=None, keep_background=True,
                 dtype=None):
        """

        Parameters
        ----------
        label_map : list or [list of] int
            Map one-hot classes to [list of] labels or label names
            (!! Should not include the background class !!)
        label_ref : dict[int] -> str
            Map label values to label names
        keep_background : bool
            If True, the first one-hot class is the background class,
            and the one hot tensor sums to one.
        dtype : torch.dtype
            Use a different dtype for the one-hot
        """
        super().__init__()
        self.label_map = label_map
        self.label_ref = label_ref
        self.keep_background = keep_background
        self.dtype = dtype

    def get_parameters(self, x):
        def get_key(map, value):
            if isinstance(map, dict):
                for k, v in map:
                    if v == value:
                        return k
            else:
                for k, v in enumerate(map):
                    if v == value:
                        return k
            raise ValueError(f'Cannot find "{value}"')

        label_map = self.label_map
        if label_map is None:
            label_map = x.unique(sorted=True)
            if label_map[0] == 0:
                label_map = label_map[1:]
            return label_map.tolist()
        label_ref = self.label_ref
        if label_ref is not None:
            new_label_map = []
            for label in label_map:
                if isinstance(label, (list, tuple)):
                    label = [get_key(label_ref, l) for l in label]
                else:
                    label = get_key(label_ref, label)
                new_label_map.append(label)
            return new_label_map
        return label_map

    def apply_transform(self, x, parameters):
        if len(x) != 1:
            raise ValueError('Cannot one-hot multi-channel tensors')
        x = x[0]

        lmax = len(parameters) + self.keep_background
        y = x.new_zeros([lmax, *x.shape], dtype=self.dtype)

        for new_l, old_l in enumerate(parameters):
            new_l += self.keep_background
            if isinstance(old_l, (list, tuple)):
                for old_l1 in old_l:
                    y[new_l, x == old_l1] = 1
            else:
                y[new_l, x == old_l] = 1

        if self.keep_background:
            y[0] = 1 - y[1:].sum(0)

        return y


class ArgMaxTransform(Transform):

    def apply_transform(self, x, parameters):
        return x.argmax(0)


class GaussianMixtureTransform(Transform):
    """Sample from a Gaussian mixture with known cluster assignment"""

    def __init__(self, mu=None, sigma=None, fwhm=0, shared=False):
        """

        Parameters
        ----------
        mu : list[float]
            Mean of each cluster
        sigma : list[float]
            Standard deviation of each cluster
        fwhm : float or list[float], optional
            Width of a within-class smoothing kernel.
        """
        super().__init__(shared=shared)
        self.mu = mu
        self.sigma = sigma
        self.fwhm = fwhm

    def get_parameters(self, x):
        mu = self.mu
        sigma = self.sigma
        if mu is None:
            mu = torch.rand([len(x)]).mul_(255)
        if sigma is None:
            sigma = torch.full([len(x)], 255 / len(x))
        if x.dtype.is_floating_point:
            backend = dict(dtype=x.dtype, device=x.device)
        else:
            backend = dict(dtype=torch.get_default_dtype(), device=x.device)
        mu = torch.as_tensor(mu).to(**backend)
        sigma = torch.as_tensor(sigma).to(**backend)
        return mu, sigma

    def apply_transform(self, x, parameters):
        mu, sigma = parameters
        fwhm = ensure_list(self.fwhm, x.dim() - 1) if self.fwhm else None

        y = 0
        if x.dtype.is_floating_point:
            backend = dict(dtype=x.dtype, device=x.device)
            mu = mu.to(**backend)
            sigma = sigma.to(**backend)
            for k in range(len(x)):
                y1 = torch.randn(x.shape[1:], **backend)
                if fwhm:
                    y1 = smoothnd(y1, fwhm=fwhm)
                y += x[k] * y1.mul_(sigma[k]).add_(mu[k])
            y = y[None]
        else:
            backend = dict(dtype=torch.get_default_dtype(), device=x.device)
            mu = mu.to(**backend)
            sigma = sigma.to(**backend)
            for k in range(x.max().item()+1):
                y1 = torch.randn(x.shape[1:], **backend)
                if fwhm:
                    y1 = smoothnd(y1, fwhm=fwhm)
                y += (x == k) * y1.mul_(sigma[k]).add_(mu[k])
        return y


class RandomGaussianMixtureTransform(RandomizedTransform):
    """
    Sample from a randomized Gaussian mixture with known cluster assignment.
    """

    def __init__(self, mu=255, sigma=16,
                 fwhm=2, shared='channels'):
        """

        Parameters
        ----------
        mu : Sampler or [list of] float
            Sampling function for cluster means, or upper bound
        sigma : callable or [list of] float
            Sampling function for cluster standard deviations, or upper bound
        fwhm : callable or [list of] float
            Sampling function for smoothing width, or upper bound
        """
        def to_range(vmax):
            if not isinstance(vmax, Sampler):
                if isinstance(vmax, (list, tuple)):
                    vmax = ([0] * len(vmax), vmax)
                else:
                    vmax = (0, vmax)
            return vmax

        sample = dict(mu=Uniform.make(to_range(mu)),
                      sigma=Uniform.make(to_range(sigma)),
                      fwhm=Uniform.make(to_range(fwhm)))
        super().__init__(GaussianMixtureTransform, sample, shared=shared)

    def get_parameters(self, x):
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        return self.subtransform(**{k: f(n) if callable(f)
                                    else ensure_list(f, n)
                                    for k, f in self.sample.items()})
