__all__ = ['OneHotTransform', 'ArgMaxTransform', 'GaussianMixtureTransform',
           'RandomGaussianMixtureTransform', 'SmoothLabelMap',
           'ErodeLabelTransform', 'RandomErodeLabelTransform',
           'BernoulliTransform', 'SmoothBernoulliTransform',
           'BernoulliDiskTransform', 'SmoothBernoulliDiskTransform',
           'RelabelTransform']

import torch
from .random import Uniform, Sampler, RandInt
from .base import Transform, RandomizedTransform
from .intensity import BaseFieldTransform
from .utils.conv import smoothnd
from .utils.py import ensure_list
from .utils.morpho import erode
import interpol
import distmap
import random as pyrand
import math as pymath


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


class RelabelTransform(Transform):
    """Relabel a label map"""

    def __init__(self, labels):
        """

        Parameters
        ----------
        labels : list of [list of] int, optional
            Relabeling scheme.
            The labels in this list are mapped to the range {1..len(labels)}.
            If an element of this list is a sublist of indices, they are merged.
            All labels absent from the list are mapped to 0.
        """
        super().__init__(shared=False)
        self.labels = labels

    def apply_transform(self, x, parameters=None):
        y = torch.zeros_like(x)
        for out, inp in enumerate(self.labels):
            out = out + 1
            if not isinstance(inp, (list, tuple)):
                inp = [inp]
            for inp1 in inp:
                y.masked_fill_(x == inp1, out)
        return y


class GaussianMixtureTransform(Transform):
    """Sample from a Gaussian mixture with known cluster assignment"""

    def __init__(self, mu=None, sigma=None, fwhm=0, background=None, shared=False):
        """

        Parameters
        ----------
        mu : list[float]
            Mean of each cluster
        sigma : list[float]
            Standard deviation of each cluster
        fwhm : float or list[float], optional
            Width of a within-class smoothing kernel.
        background : int, optional
            Index of background channel
        """
        super().__init__(shared=shared)
        self.mu = mu
        self.sigma = sigma
        self.fwhm = fwhm
        self.background = background

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

        if x.dtype.is_floating_point:
            backend = dict(dtype=x.dtype, device=x.device)
            y = torch.zeros_like(x[0], **backend)
            mu = mu.to(**backend)
            sigma = sigma.to(**backend)
            for k in range(len(x)):
                if self.background is not None and k == self.background:
                    continue
                y1 = torch.randn(x.shape[1:], **backend)
                if fwhm:
                    y1 = smoothnd(y1, fwhm=fwhm)
                y += x[k] * y1.mul_(sigma[k]).add_(mu[k])
            y = y[None]
        else:
            backend = dict(dtype=torch.get_default_dtype(), device=x.device)
            y = torch.zeros_like(x, **backend)
            mu = mu.to(**backend)
            sigma = sigma.to(**backend)
            for k in range(x.max().item()+1):
                if self.background is not None and k == self.background:
                    continue
                if fwhm:
                    y1 = torch.randn(x.shape[1:], **backend)
                    if fwhm:
                        y1 = smoothnd(y1, fwhm=fwhm)
                    y += (x == k) * y1.mul_(sigma[k]).add_(mu[k])
                else:
                    mask = x == k
                    y[mask] = torch.randn(mask.sum(), **backend)
        return y


class RandomGaussianMixtureTransform(RandomizedTransform):
    """
    Sample from a randomized Gaussian mixture with known cluster assignment.
    """

    def __init__(self, mu=255, sigma=16, fwhm=2, background=None, shared='channels'):
        """

        Parameters
        ----------
        mu : Sampler or [list of] float
            Sampling function for cluster means, or upper bound
        sigma : callable or [list of] float
            Sampling function for cluster standard deviations, or upper bound
        fwhm : callable or [list of] float
            Sampling function for smoothing width, or upper bound
        background : int, optional
            Index of background channel
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
                      fwhm=Uniform.make(to_range(fwhm)),
                      background=background)
        super().__init__(GaussianMixtureTransform, sample, shared=shared)

    def get_parameters(self, x):
        n = len(x) if x.dtype.is_floating_point else x.max() + 1
        return self.subtransform(**{k: f(n) if callable(f)
                                    else ensure_list(f, n) if k != 'background'
                                    else f for k, f in self.sample.items()})


class SmoothLabelMap(Transform):
    """Generate a random label map"""

    def __init__(self, nb_classes=2, shape=5, soft=False, shared=False):
        """

        Parameters
        ----------
        nb_classes : int
            Number of classes
        shape : [list of] int
            Number of spline control points
        soft : bool
            Return a soft (one-hot) label map
        shared : bool
            Apply the same field to all channels
        """
        super().__init__(shared=shared)
        self.nb_classes = nb_classes
        self.shape = shape
        self.soft = soft

    def get_parameters(self, x):
        batch, *fullshape = x.shape
        smallshape = ensure_list(self.shape, len(fullshape))
        backend = dict(dtype=x.dtype, device=x.device)
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        if self.soft:
            if batch > 1:
                raise ValueError('Cannot generate a batched soft label map')
            b = torch.rand([self.nb_classes, *smallshape], **backend)
            b = interpol.resize(b, shape=fullshape, interpolation=3,
                                prefilter=False)
            b = b.softmax(0)
        else:
            maxprob = torch.full_like(x, float('-inf'), **backend)
            b = torch.zeros_like(x, dtype=torch.long)
            for k in range(self.nb_classes):
                b1 = torch.rand([batch, *smallshape], **backend)
                b1 = interpol.resize(b1, shape=fullshape, interpolation=3,
                                     prefilter=False)
                mask = maxprob < b1
                b.masked_fill_(mask, k)
                maxprob[mask] = b1[mask]
        return b

    def apply_transform(self, x, parameters):
        return parameters


class ErodeLabelTransform(Transform):

    def __init__(self, labels, radius=3, output_labels=0):
        """

        Parameters
        ----------
        labels : [sequence of] int
            Labels to erode
        radius : [sequence of] int
            Erosion radius (per label)
        output_labels : [sequence of] int
            Output label (per input label)
        """
        super().__init__(shared=True)
        self.labels = ensure_list(labels)
        self.radius = ensure_list(radius, len(self.labels))
        self.output_labels = ensure_list(output_labels, len(self.labels))

    def get_parameters(self, x):
        return None

    def apply_transform(self, x, parameters):
        y = x.clone()
        for label, radius, outlabel \
                in zip(self.labels, self.radius, self.output_labels):
            x1 = x == label
            x1 = erode(x1, nb_iter=radius, dim=x.dim()-1).logical_xor_(x1)
            y[x1] = outlabel
        return y


class RandomErodeLabelTransform(RandomizedTransform):

    def __init__(self, labels=0.5, radius=3, output_labels=0, shared=False):
        """

        Parameters
        ----------
        labels : Sampler or float or [sequence of] int
            Labels to erode.
            If a float in 0..1, probability of eroding a label
        radius : Sampler or int
            Erosion radius (per label).
            Either an int sampler, or an upper bound.
        output_labels : int or 'unique'
            Output label
            If 'unique', assign a novel unique label (per input label).
        """
        def to_range(value):
            if not isinstance(value, Sampler):
                if not isinstance(value, (list, tuple)):
                    value = (0, value)
                value = tuple(value)
            return value

        super().__init__(ErodeLabelTransform,
                         dict(labels=labels,
                              radius=RandInt.make(to_range(radius)),
                              output_labels=output_labels),
                         shared=shared)

    def get_parameters(self, x):
        sample = dict(self.sample)
        n = None
        if isinstance(sample['labels'], float):
            prob = sample['labels']
            labels = x.unique().tolist()
            n = int(pymath.ceil(len(labels) * prob))
            def label_sampler():
                pyrand.shuffle(labels)
                return labels[:n]
            sample['labels'] = label_sampler
        if sample['output_labels'] == 'unique':
            max_label = x.unique().max().item() + 1
            if n is None:
                n = len(ensure_list(sample['labels']()
                                    if callable(sample['labels']) else
                                    sample['labels']))
            sample['output_labels'] = list(range(max_label, max_label + n))
        if callable(sample['radius']):
            if n is None:
                n = len(ensure_list(sample['labels']()
                                    if callable(sample['labels']) else
                                    sample['labels']))
            sampler = sample['radius']
            def sample_radius():
                return [sampler() for _ in range(n)]
            sample['radius'] = sample_radius

        return self.subtransform(**{k: f() if callable(f) else f
                                    for k, f in sample.items()})


class BernoulliTransform(Transform):
    """Randomly mask voxels"""

    def __init__(self, prob=0.1, shared=False):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
        shared : bool
            Same mask shared across channels
        """
        super().__init__(shared=shared)
        self.prob = prob

    def get_parameters(self, x):
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        return torch.rand_like(x, dtype=dtype) > self.prob

    def apply_transform(self, x, parameters):
        return x * (~parameters)


class SmoothBernoulliTransform(BaseFieldTransform):
    """Randomly mask voxels"""

    def __init__(self, prob=0.1, shape=5, shared=False):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
        shape : int or sequence[int]
            Number of control points in the smooth field
        shared : bool
            Same mask shared across channels
        """
        super().__init__(shape=shape, shared=shared)
        self.prob = prob

    def get_parameters(self, x):
        ndim = x.dim() - 1
        prob = super().get_parameters(x)
        prob /= prob.sum(list(range(-ndim, 0)), keepdim=True)
        prob *= self.prob * x.shape[1:].numel()
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        return torch.rand_like(x, dtype=dtype) > (1 - prob)

    def apply_transform(self, x, parameters):
        return x * (~parameters)


class BernoulliDiskTransform(Transform):
    """Randomly mask voxels in balls at random locations"""

    def __init__(self, prob=0.1, radius=2, shared=False):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
        radius : float or Sampler
            Disk radius
        shared : bool
            Same mask shared across channels
        """
        super().__init__(shared=shared)
        self.prob = prob
        self.radius = radius

    def get_parameters(self, x):
        ndim = x.dim() - 1
        nvoxball = pymath.pow(pymath.pi, ndim/2) / pymath.gamma(ndim/2 + 1)

        # sample radii
        if isinstance(self.radius, Sampler):
            radius = self.radius(x.shape, device=x.device)
            nvoxball = (nvoxball * radius).mean().item()
        else:
            radius = self.radius
            nvoxball *= radius
        nvoxball = max(nvoxball, 1)

        # sample locations
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        loc = torch.rand_like(x, dtype=dtype) > (1 - self.prob / nvoxball)

        dist = distmap.euclidan_distance_transform(~loc, ndim=ndim)
        return dist < radius

    def apply_transform(self, x, parameters):
        return x * (~parameters)


class SmoothBernoulliDiskTransform(BaseFieldTransform):
    """Randomly mask voxels in balls at random locations"""

    def __init__(self, prob=0.1, radius=2, shape=5, shared=False):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
            A probability field is sampled from a smooth random field.
        radius : float or (float, float) or Sampler
            Disk radius.
            If a float (max) or two floats (min, max), radius is sampled
            from a smooth random field.
            If a Sampler, radius is sampled from the sampler.
        shape : int or sequence[int]
            Number of control points in the random field
        shared : bool
            Same mask shared across channels
        """
        super().__init__(shape=shape, shared=shared)
        self.prob = prob
        self.radius = radius

    def get_parameters(self, x):
        ndim = x.dim() - 1
        nvoxball = pymath.pow(pymath.pi, ndim / 2) / pymath.gamma(ndim / 2 + 1)

        # sample radii
        if isinstance(self.radius, Sampler):
            radius = self.radius(x.shape, device=x.device)
        else:
            radius = ensure_list(self.radius)
            if len(radius) == 1:
                mn, mx = 0, radius[0]
            else:
                mn, mx = radius
            radius = super().get_parameters(x)
            radius.mul_(mx - mn).add_(mn)
        nvoxball = (nvoxball * radius).mean().item()
        nvoxball = max(nvoxball, 1)

        # sample locations
        prob = super().get_parameters(x)
        prob /= prob.sum(list(range(-ndim, 0)), keepdim=True)
        prob *= self.prob * x.shape[1:].numel() / nvoxball

        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        loc = torch.rand_like(x, dtype=dtype) > (1 - prob)

        dist = distmap.euclidan_distance_transform(~loc, ndim=ndim)
        return dist < radius

    def apply_transform(self, x, parameters):
        return x * (~parameters)