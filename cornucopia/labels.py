__all__ = ['OneHotTransform', 'ArgMaxTransform', 'GaussianMixtureTransform',
           'RandomGaussianMixtureTransform', 'SmoothLabelMap',
           'ErodeLabelTransform', 'RandomErodeLabelTransform',
           'DilateLabelTransform', 'RandomDilateLabelTransform',
           'SmoothMorphoLabelTransform', 'RandomSmoothMorphoLabelTransform',
           'SmoothShallowLabelTransform', 'RandomSmoothShallowLabelTransform',
           'BernoulliTransform', 'SmoothBernoulliTransform',
           'BernoulliDiskTransform', 'SmoothBernoulliDiskTransform',
           'RelabelTransform']

import torch
from .random import Uniform, Sampler, RandInt, Fixed, upper_range, lower_range
from .base import Transform, RandomizedTransform
from .intensity import BaseFieldTransform
from .utils.conv import smoothnd
from .utils.py import ensure_list
from .utils.morpho import bounded_distance
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
    """
    Morphological erosion.
    """

    def __init__(self, labels=tuple(), radius=3, method='conv'):
        """
        Parameters
        ----------
        labels : [sequence of] int
            Labels to erode
        radius : [sequence of] int
            Erosion radius (per label)
        method : {'conv', 'l1', 'l2'}
            Method to use to compute the distance map.
            If 'conv', use the L1 distance computed using a series of
            convolutions. The radius will be rounded to the nearest integer.
            Otherwise, use the appropriate distance transform.
        """
        super().__init__(shared=True)
        self.labels = ensure_list(labels)
        self.radius = ensure_list(radius, min(len(self.labels), 1))
        self.method = method

    def get_parameters(self, x):
        return None

    def apply_transform(self, x, parameters):
        max_radius = max(self.radius)
        if self.method == 'conv':
            dist = lambda x, r: bounded_distance(x, int(round(r)), dim=x.dim()-1)
            dtype = torch.int
            dmax = max_radius+1
        elif self.method == 'l1':
            dist = lambda x, r: distmap.l1_signed_transform(x, ndim=x.dim()-1).neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        elif self.method == 'l2':
            dist = lambda x, r: distmap.euclidean_signed_transform(x, ndim=x.dim()-1).neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        else:
            raise ValueError('Unknown method', self.method)

        all_labels = x.unique()
        foreground_labels = self.labels
        if not foreground_labels:
            foreground_labels = all_labels[all_labels != 0]
        foreground_radius = ensure_list(self.radius, len(foreground_labels))

        y = torch.zeros_like(x)
        d = torch.full_like(x, dmax, dtype=dtype)
        for label in all_labels:
            is_foreground = label in foreground_labels
            if is_foreground:
                radius = foreground_radius[foreground_labels.index(label)]
            else:
                radius = max_radius
            x0 = x == label
            d1 = dist(x0, radius)
            if is_foreground:
                mask = d1 < -radius
            else:
                mask = d1 < d
            d[mask] = d1[mask]
            y.masked_fill_(mask, label)
        return y


class DilateLabelTransform(Transform):
    """
    Morphological dilation.
    """

    def __init__(self, labels=tuple(), radius=3, method='conv'):
        """
        Parameters
        ----------
        labels : [sequence of] int
            Labels to dilate. By default, all but zero.
        radius : [sequence of] int
            Dilation radius (per label)
        method : {'conv', 'l1', 'l2'}
            Method to use to compute the distance map.
            If 'conv', use the L1 distance computed using a series of
            convolutions. The radius will be rounded to the nearest integer.
            Otherwise, use the appropriate distance transform.
        """
        super().__init__(shared=True)
        self.labels = ensure_list(labels)
        self.radius = ensure_list(radius, min(len(self.labels), 1))
        self.method = method

    def get_parameters(self, x):
        return None

    def apply_transform(self, x, parameters):
        max_radius = max(self.radius)
        if self.method == 'conv':
            dist = lambda x, r: bounded_distance(x, int(round(r)), dim=x.dim()-1)
            dtype = torch.int
            dmax = max_radius+1
        elif self.method == 'l1':
            dist = lambda x, r: distmap.l1_signed_transform(x, ndim=x.dim()-1).neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        elif self.method == 'l2':
            dist = lambda x, r: distmap.euclidean_signed_transform(x, ndim=x.dim()-1).neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        else:
            raise ValueError('Unknown method', self.method)

        all_labels = x.unique()
        foreground_labels = self.labels
        if not foreground_labels:
            foreground_labels = all_labels[all_labels != 0]
        foreground_radius = ensure_list(self.radius, len(foreground_labels))

        y = x.clone()
        d = torch.full_like(x, dmax, dtype=dtype)
        for label in all_labels:
            is_foreground = label in foreground_labels
            if not is_foreground:
                continue
            radius = foreground_radius[foreground_labels.index(label)]
            x0 = x == label
            d1 = dist(x0, radius)
            if is_foreground:
                mask = d1 < radius
                mask = mask & (d1 < d)
            d[mask] = d1[mask]
            y.masked_fill_(mask, label)
        return y


class RandomErodeLabelTransform(RandomizedTransform):
    """
    Morphological erosion with random radius/labels.
    """

    def __init__(self, labels=0.5, radius=3, method='conv', shared=False):
        """

        Parameters
        ----------
        labels : Sampler or float or [sequence of] int
            Labels to erode.
            If a float in 0..1, probability of eroding a label
        radius : Sampler or int
            Erosion radius (per label).
            Either an int sampler, or an upper bound.
        method : {'conv', 'l1', 'l2'}
            Method to use to compute the distance map.
            If 'conv', use the L1 distance computed using a series of
            convolutions. The radius will be rounded to the nearest integer.
            Otherwise, use the appropriate distance transform.
        """
        def to_range(value):
            if not isinstance(value, Sampler):
                if not isinstance(value, (list, tuple)):
                    value = (0, value)
                value = tuple(value)
            return value

        super().__init__(ErodeLabelTransform,
                         dict(labels=labels,
                              radius=Uniform.make(to_range(radius)),
                              method=Fixed(method)),
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


class RandomDilateLabelTransform(RandomizedTransform):
    """
    Morphological dilation with random radius/labels.
    """

    def __init__(self, labels=0.5, radius=3, method='conv', shared=False):
        """

        Parameters
        ----------
        labels : Sampler or float or [sequence of] int
            Labels to dilate.
            If a float in 0..1, probability of eroding a label
        radius : Sampler or int
            Dilation radius (per label).
            Either an int sampler, or an upper bound.
        method : {'conv', 'l1', 'l2'}
            Method to use to compute the distance map.
            If 'conv', use the L1 distance computed using a series of
            convolutions. The radius will be rounded to the nearest integer.
            Otherwise, use the appropriate distance transform.
        """
        def to_range(value):
            if not isinstance(value, Sampler):
                if not isinstance(value, (list, tuple)):
                    value = (0, value)
                value = tuple(value)
            return value

        super().__init__(DilateLabelTransform,
                         dict(labels=labels,
                              radius=Uniform.make(to_range(radius)),
                              method=Fixed(method)),
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


class SmoothMorphoLabelTransform(Transform):
    """
    Morphological erosion with spatially varying radius.

    We're actually computing the level set of each label and pushing it
    up and down using a smooth "radius" map. In theory, this can
    create "holes" or "islands", which would not happen with a normal
    erosion. With radii that are small and radius map that are smooth
    compared to the label size, it should be fine.
    """

    def __init__(self, labels=tuple(), min_radius=-3, max_radius=3, shape=5, method='conv'):
        """
        Parameters
        ----------
        labels : [sequence of] int
            Labels to erode
        min_radius : [sequence of] int
            Minimum erosion (if < 0) or dilation (if > 0) radius (per label)
        max_radius : [sequence of] int
            Maximum erosion (if < 0) or dilation (if > 0)  radius (per label)
        shape : [sequence of] int
            Number of nodes in the smooth radius map
        method : {'conv', 'l1', 'l2'}
            Method to use to compute the distance map.
            If 'conv', use the L1 distance computed using a series of
            convolutions. The radius will be rounded to the nearest integer.
            Otherwise, use the appropriate distance transform.
        """
        super().__init__(shared=True)
        self.labels = ensure_list(labels)
        self.min_radius = ensure_list(min_radius, min(len(self.labels), 1))
        self.max_radius = ensure_list(max_radius, min(len(self.labels), 1))
        self.shape = shape
        self.method = method

    def get_parameters(self, x):
        return None

    def apply_transform(self, x, parameters):
        max_abs_radius = max(max(map(abs, self.min_radius)),
                             max(map(abs, self.max_radius))) + 1
        if self.method == 'conv':
            dist = lambda x: bounded_distance(x, nb_iter=max_abs_radius, dim=x.dim()-1)
        elif self.method == 'l1':
            dist = lambda x: distmap.l1_signed_transform(x, ndim=x.dim()-1).neg_()
        elif self.method == 'l2':
            dist = lambda x: distmap.euclidean_signed_transform(x, ndim=x.dim()-1).neg_()
        else:
            raise ValueError('Unknown method', self.method)

        all_labels = x.unique()
        foreground_labels = self.labels
        if not foreground_labels:
            foreground_labels = all_labels[all_labels != 0]
        foreground_min_radius = ensure_list(self.min_radius, len(foreground_labels))
        foreground_max_radius = ensure_list(self.max_radius, len(foreground_labels))

        dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
        y = torch.zeros_like(x)
        d = torch.full_like(x, float('inf'), dtype=dtype)
        all_labels = x.unique()
        for label in all_labels:
            x0 = x == label
            is_foreground = label in foreground_labels
            if is_foreground:
                label_index = foreground_labels.index(label)
                min_radius = foreground_min_radius[label_index]
                max_radius = foreground_max_radius[label_index]
                radius = BaseFieldTransform(self.shape, min_radius, max_radius)
                radius = radius.get_parameters(x)
            else:
                radius = 0
            d1 = dist(x0).to(d).sub_(radius)
            mask = d1 < d
            d[mask] = d1[mask]
            y.masked_fill_(mask, label)
        return y


class RandomSmoothMorphoLabelTransform(RandomizedTransform):
    """
    Morphological erosion/dilation with smooth random radius/labels.
    """

    def __init__(self, labels=0.5, min_radius=-3, max_radius=3,
                 shape=5, method='conv', shared=False):
        """

        Parameters
        ----------
        labels : Sampler or float or [sequence of] int
            Labels to dilate.
            If a float in 0..1, probability of eroding a label
        min_radius : [sequence of] int
            Minimum erosion (if < 0) or dilation (if > 0) radius (per label)
            Either an int sampler, or an upper bound.
        max_radius : [sequence of] int
            Maximum erosion (if < 0) or dilation (if > 0)  radius (per label)
            Either an int sampler, or an upper bound.
        shape : [sequence of] int
            Number of nodes in the smooth radius map
            Either an int sampler, or an upper bound.
        method : {'conv', 'l1', 'l2'}
            Method to use to compute the distance map.
            If 'conv', use the L1 distance computed using a series of
            convolutions. The radius will be rounded to the nearest integer.
            Otherwise, use the appropriate distance transform.
        """
        super().__init__(SmoothMorphoLabelTransform,
                         dict(labels=labels,
                              min_radius=Uniform.make(lower_range(min_radius)),
                              max_radius=Uniform.make(upper_range(max_radius)),
                              shape=RandInt.make(upper_range(shape)),
                              method=Fixed(method)),
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
        if callable(sample['min_radius']):
            if n is None:
                n = len(ensure_list(sample['labels']()
                                    if callable(sample['labels']) else
                                    sample['labels']))
            sampler = sample['min_radius']
            def sample_min_radius():
                return [sampler() for _ in range(n)]
            sample['min_radius'] = sample_min_radius
        if callable(sample['max_radius']):
            if n is None:
                n = len(ensure_list(sample['labels']()
                                    if callable(sample['labels']) else
                                    sample['labels']))
            sampler = sample['max_radius']
            def sample_max_radius():
                return [sampler() for _ in range(n)]
            sample['max_radius'] = sample_max_radius

        return self.subtransform(**{k: f() if callable(f) else f
                                    for k, f in sample.items()})


class SmoothShallowLabelTransform(Transform):
    """Make labels "empty", with a border of a given size."""

    def __init__(self, labels=tuple(), max_width=5, min_width=1, shape=5,
                 background_labels=tuple(), method='l2', shared=False):
        """
        Parameters
        ----------
        labels : [sequence of] int
            Labels to make shallow
        max_width : [sequence of] int
            Maximum border width (per label)
        min_width : [sequence of] int
            Minimum border width (per label)
        shape : [sequence of] int
            Number of nodes in the smooth width map
        background_labels : [sequence of] int
            Labels that are allowed to grow into the shallow space
            (default: all that are not in labels)
        method : {'l1', 'l2'}
            Method to use to compute the distance map.
        """
        super().__init__(shared=shared)
        self.labels = ensure_list(labels)
        self.background_labels = ensure_list(background_labels)
        self.shape = shape
        self.min_width = ensure_list(min_width, min(len(self.labels), 1))
        self.max_width = ensure_list(max_width, min(len(self.labels), 1))
        self.method = method

    def get_parameters(self, x):
        return None

    def apply_transform(self, x, parameters):
        if self.method == 'l1':
            dist = lambda x: distmap.l1_signed_transform(x, ndim=x.dim()-1).neg_()
        elif self.method == 'l2':
            dist = lambda x: distmap.euclidean_signed_transform(x, ndim=x.dim()-1).neg_()
        else:
            raise ValueError('Unknown method', self.method)

        all_labels = x.unique()
        foreground_labels = self.labels
        if not foreground_labels:
            foreground_labels = all_labels[all_labels != 0]
        background_labels = self.background_labels
        if not background_labels:
            background_labels = [l for l in all_labels if l not in foreground_labels]
        foreground_min_width = ensure_list(self.min_width, len(foreground_labels))
        foreground_max_width = ensure_list(self.max_width, len(foreground_labels))

        dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
        y = torch.zeros_like(x)
        d = torch.full_like(x, float('inf'), dtype=dtype)
        m = torch.zeros_like(x, dtype=torch.bool)
        all_labels = x.unique()

        # fill object borders and keep track of object interiors
        for label, min_width, max_width in zip(foreground_labels,
                                               foreground_min_width,
                                               foreground_max_width):
            x0 = x == label
            radius = BaseFieldTransform(self.shape, -min_width, -max_width)
            radius = radius.get_parameters(x)
            d1 = dist(x0).to(d)
            mask = (d1 < 0) & (d1 > radius)
            m.masked_fill_(d1 < radius, True)
            d[mask] = d1[mask]
            y.masked_fill_(mask, label)

        # elsewhere, use maximum probability labels
        for label in all_labels:
            if label in foreground_labels:
                continue
            x0 = x == label
            if label not in background_labels:
                y.masked_fill_(x0, label)
                d.masked_fill_(x0, -1)
            elif len(background_labels) == 1:
                y.masked_fill_(x0, label)
                d.masked_fill_(x0, -1)
                y.masked_fill_(m, label)
                d.masked_fill_(m, -1)
            else:
                d1 = dist(x0).to(d)
                mask = d1 < d
                d[mask] = d1[mask]
                y.masked_fill_(mask, label)

        return y


class RandomSmoothShallowLabelTransform(RandomizedTransform):
    """
    Make labels "empty", with a border of a given (random) size.
    """

    def __init__(self, labels=0.5, max_width=5, min_width=1, shape=5,
                 background_labels=tuple(), method='l2', shared=False):
        """
        Parameters
        ----------
        labels : Sampler or float or [sequence of] int
            Labels to make shallow
            If a float in 0..1, probability of eroding a label
        max_width : Sampler or [sequence of] float
            Maximum border width (per label)
            Either an int sampler, or an upper bound.
        min_width : Sampler or [sequence of] float
            Minimum border width (per label)
            Either an int sampler, or an upper bound.
        shape : Sampler or [sequence of] int
            Number of nodes in the smooth width map
            Either an int sampler, or an upper bound.
        background_labels : [sequence of] int
            Labels that are allowed to grow into the shallow space
            (default: all that are not in labels)
        method : {'l1', 'l2'}
            Method to use to compute the distance map.
            If 'conv', use the L1 distance computed using a series of
            convolutions. The radius will be rounded to the nearest integer.
            Otherwise, use the appropriate distance transform.
        """
        super().__init__(SmoothShallowLabelTransform,
                         dict(labels=labels,
                              min_width=Uniform.make(lower_range(min_width)),
                              max_width=Uniform.make(upper_range(max_width)),
                              shape=RandInt.make(upper_range(shape)),
                              background_labels=Fixed(background_labels),
                              method=Fixed(method)),
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
        if callable(sample['min_width']):
            if n is None:
                n = len(ensure_list(sample['labels']()
                                    if callable(sample['labels']) else
                                    sample['labels']))
            sampler = sample['min_width']
            def sample_min_width():
                return [sampler() for _ in range(n)]
            sample['min_width'] = sample_min_width
        if callable(sample['max_width']):
            if n is None:
                n = len(ensure_list(sample['labels']()
                                    if callable(sample['labels']) else
                                    sample['labels']))
            sampler = sample['max_width']
            def sample_max_width():
                return [sampler() for _ in range(n)]
            sample['max_width'] = sample_max_width

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