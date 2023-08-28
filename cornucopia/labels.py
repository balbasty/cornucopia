__all__ = [
    'OneHotTransform',
    'ArgMaxTransform',
    'RelabelTransform',
    'GaussianMixtureTransform',
    'RandomGaussianMixtureTransform',
    'SmoothLabelMap',
    'RandomSmoothLabelMap',
    'ErodeLabelTransform',
    'RandomErodeLabelTransform',
    'DilateLabelTransform',
    'RandomDilateLabelTransform',
    'SmoothMorphoLabelTransform',
    'RandomSmoothMorphoLabelTransform',
    'SmoothShallowLabelTransform',
    'RandomSmoothShallowLabelTransform',
    'BernoulliTransform',
    'SmoothBernoulliTransform',
    'BernoulliDiskTransform',
    'SmoothBernoulliDiskTransform',
]

import torch
from .random import Uniform, Sampler, RandInt, Fixed, RandKFrom, make_range
from .base import FinalTransform, NonFinalTransform
from .baseutils import prepare_output
from .intensity import AddFieldTransform, MulValueTransform, FillValueTransform
from .utils.conv import smoothnd
from .utils.py import ensure_list
from .utils.morpho import bounded_distance
from . import ctx
import interpol
import distmap
import math as pymath


class OneHotTransform(NonFinalTransform):
    """Transform a volume of integer labels into a one-hot representation"""

    def __init__(self, label_map=None, label_ref=None, keep_background=True,
                 dtype=None, **kwargs):
        """

        Parameters
        ----------
        label_map : list or [list of] int
            Map one-hot classes to [list of] labels or label names
            !!! warning "Should not include the background class"
        label_ref : dict[int] -> str
            Map label values to label names
        keep_background : bool
            If True, the first one-hot class is the background class,
            and the one hot tensor sums to one.
        dtype : torch.dtype
            Use a different dtype for the one-hot
        """
        super().__init__(**kwargs)
        self.label_map = label_map
        self.label_ref = label_ref
        self.keep_background = keep_background
        self.dtype = dtype

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self

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
            label_map = label_map.tolist()
        else:
            label_ref = self.label_ref
            if label_ref is not None:
                new_label_map = []
                for label in label_map:
                    if isinstance(label, (list, tuple)):
                        label = [get_key(label_ref, lab) for lab in label]
                    else:
                        label = get_key(label_ref, label)
                    new_label_map.append(label)
                label_map = new_label_map

        return self.Final(
            label_map, self.keep_background, self.dtype, **self.get_prm()
        ).make_final(x, max_depth-1)

    class Final(FinalTransform):

        def __init__(self, labels, keep_background=True, dtype=None,
                     **kwargs):
            super().__init__(**kwargs)
            self.labels = labels
            self.keep_background = keep_background
            self.dtype = dtype

        def apply(self, x):
            if len(x) != 1:
                raise ValueError('Cannot one-hot multi-channel tensors')
            x = x[0]

            lmax = len(self.labels) + self.keep_background
            y = x.new_zeros([lmax, *x.shape], dtype=self.dtype)

            for new_l, old_l in enumerate(self.labels):
                new_l += self.keep_background
                if isinstance(old_l, (list, tuple)):
                    for old_l1 in old_l:
                        y[new_l, x == old_l1] = 1
                else:
                    y[new_l, x == old_l] = 1

            if self.keep_background:
                y[0] = 1 - y[1:].sum(0)

            return y


class ArgMaxTransform(FinalTransform):

    def apply(self, x):
        return x.argmax(0)[None]


class RelabelTransform(NonFinalTransform):
    """Relabel a label map"""

    def __init__(self, labels=None, **kwargs):
        """

        !!! note

            - The `labels` are mapped to the range `{1..len(labels)}`.

            - If an element of this list is a sublist of indices,
              indices in the sublist are merged.

            - All labels absent from the list are mapped to `0`.

        Parameters
        ----------
        labels : list of [list of] int, optional
            Relabeling scheme.

        """
        super().__init__(**kwargs)
        self.labels = labels

    def make_final(self, x, max_depth=float('inf')):
        if self.is_final:
            return self
        labels = self.labels
        if labels is None:
            labels = x.unique().tolist()[1:]
        return self.Final(
            labels, **self.get_prm()
        ).make_final(x, max_depth-1)

    class Final(FinalTransform):

        def __init__(self, labels, **kwargs):
            super().__init__(**kwargs)
            self.labels = labels

        def apply(self, x):
            if self.labels is None:
                return self.make_final(x)(x)
            assert self.labels is not None
            y = torch.zeros_like(x)
            for out, inp in enumerate(self.labels):
                out = out + 1
                if not isinstance(inp, (list, tuple)):
                    inp = [inp]
                for inp1 in inp:
                    y.masked_fill_(x == inp1, out)
            return y


class GaussianMixtureTransform(NonFinalTransform):
    """Sample from a Gaussian mixture with known cluster assignment"""

    def __init__(self, mu=None, sigma=None, fwhm=0, background=None,
                 dtype=None, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        mu : list[float]
            Mean of each cluster. Default: random in (0, 1).
        sigma : list[float]
            Standard deviation of each cluster. Default: `1/nb_classes`.
        fwhm : float or list[float]
            Width of a within-class smoothing kernel.
        background : int
            Index of background channel, which does not get filled.
            Default: fill all classes.
        dtype : torch.dtype
            Output data type. Only used if input is an integer label map.
        """
        super().__init__(shared=shared, **kwargs)
        self.mu = mu
        self.sigma = sigma
        self.fwhm = fwhm
        self.background = background
        self.dtype = dtype

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        nk = len(x) if x.is_floating_point() else x.unique().numel()
        mu = self.mu
        sigma = self.sigma
        if mu is None:
            mu = torch.rand([nk]).tolist()
        if sigma is None:
            sigma = [1 / nk] * nk
        return self.Final(
            mu, sigma, self.fwhm, self.background, self.dtype,
            **self.get_prm()
        ).make_final(x, max_depth-1)

    class Final(FinalTransform):

        def __init__(self, mu, sigma, fwhm, background=None, dtype=None,
                     **kwargs):
            super().__init__(**kwargs)
            self.mu = mu
            self.sigma = sigma
            self.fwhm = fwhm
            self.background = background
            self.dtype = dtype

        def apply(self, x):
            mu, sigma = self.mu, self.sigma
            ndim = x.ndim - 1

            def getitem(x):
                return x.item() if torch.is_tensor(x) else x

            if x.is_floating_point():
                backend = dict(dtype=x.dtype, device=x.device)
                y = torch.zeros_like(x[0])
                fwhm = ensure_list(self.fwhm or [0], len(x))
                for k in range(len(x)):
                    muk, sigmak = getitem(mu[k]), getitem(sigma[k])
                    if self.background is not None and k == self.background:
                        continue
                    y1 = torch.randn(x.shape[1:], **backend)
                    if fwhm[k]:
                        y1 = smoothnd(y1, fwhm=[fwhm[k]]*ndim)
                    y += y1.mul_(sigmak).add_(muk).mul_(x[k])
                y = y[None]
            else:
                backend = dict(dtype=self.dtype or torch.get_default_dtype(),
                               device=x.device)
                y = torch.zeros_like(x, **backend)
                nk = x.max().item()+1
                fwhm = ensure_list(self.fwhm or [0], nk)
                for k in range(nk):
                    muk, sigmak = getitem(mu[k]), getitem(sigma[k])
                    if self.background is not None and k == self.background:
                        continue
                    if fwhm[k]:
                        y1 = torch.randn(x.shape, **backend)
                        y1 = smoothnd(y1, fwhm=[fwhm[k]]*ndim)
                        y += y1.mul_(sigmak).add_(muk).masked_fill_(x != k, 0)
                    else:
                        mask = x == k
                        numel = mask.sum()
                        if numel:
                            y1 = torch.randn(numel, **backend)
                            y1 = y1.mul_(sigmak).add_(muk)
                            y[mask] = y1
            return y


class RandomGaussianMixtureTransform(NonFinalTransform):
    """
    Sample from a randomized Gaussian mixture with known cluster assignment.
    """

    def __init__(self, mu=1, sigma=0.05, fwhm=2, background=None, dtype=None,
                 *, shared=False, **kwargs):
        """

        Parameters
        ----------
        mu : Sampler or [list of] float
            Sampling function for cluster means, or upper bound
        sigma : Sampler or [list of] float
            Sampling function for cluster standard deviations, or upper bound
        fwhm : Sampler or [list of] float
            Sampling function for smoothing width, or upper bound
        background : int, optional
            Index of background channel
        """
        super().__init__(shared=shared, **kwargs)
        self.mu = Uniform.make(make_range(0, mu))
        self.sigma = Uniform.make(make_range(0, sigma))
        self.fwhm = Uniform.make(make_range(0, fwhm))
        self.background = background
        self.dtype = dtype

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if ('channels' in self.shared
                and not x.is_floating_point()
                and len(x) > 1):
            return self.make_per_channel(x, max_depth)

        n = len(x) if x.dtype.is_floating_point else x.max() + 1

        mu, sigma, fwhm = self.mu, self.sigma, self.fwhm
        if isinstance(mu, Sampler):
            mu = mu(n)
        if isinstance(sigma, Sampler):
            sigma = sigma(n)
        if isinstance(fwhm, Sampler):
            fwhm = fwhm(n)

        return GaussianMixtureTransform(
            mu, sigma, fwhm, self.background, self.dtype,
            shared=self.shared, **self.get_prm()
        ).make_final(x, max_depth-1)


class SmoothLabelMap(NonFinalTransform):
    """Generate a random label map"""

    def __init__(self, nb_classes=2, shape=5, soft=False,
                 *, shared=False, **kwargs):
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
        super().__init__(shared=shared, **kwargs)
        self.nb_classes = nb_classes
        self.shape = shape
        self.soft = soft

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self

        ndim = x.ndim-1

        def normfield_(field):
            mn = field.reshape([len(field), -1]).min(-1).values
            mx = field.reshape([len(field), -1]).max(-1).values
            for _ in range(ndim):
                mn = mn.unsqueeze(-1)
                mx = mx.unsqueeze(-1)
            field = field.sub_(mn).div_(mx - mn)
            return field

        batch, *fullshape = x.shape
        if 'channels' in self.shared:
            batch = 1
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
            b = normfield_(b)
            b = b.softmax(0)
        else:
            maxprob = torch.full_like(x, float('-inf'), **backend)
            b = torch.zeros_like(x, dtype=torch.long)
            for k in range(self.nb_classes):
                b1 = torch.rand([batch, *smallshape], **backend)
                b1 = interpol.resize(b1, shape=fullshape, interpolation=3,
                                     prefilter=False)
                b1 = normfield_(b1)
                mask = maxprob < b1
                b.masked_fill_(mask, k)
                maxprob = torch.where(mask, b1, maxprob)
        return self.Final(b, **self.get_prm()).make_final(x, max_depth-1)

    class Final(FinalTransform):

        def __init__(self, labelmap, **kwargs):
            super().__init__(**kwargs)
            self.labelmap = labelmap

        def apply(self, x):
            return self.labelmap.to(x.device)


class RandomSmoothLabelMap(NonFinalTransform):
    """Generate a random label map with random hyper-parameters"""

    def __init__(self, nb_classes=8, shape=5, soft=False,
                 *, shared=False, shared_field=None, **kwargs):
        """

        Parameters
        ----------
        nb_classes : Sampler or int
            Maximum number of classes
        shape : Sampler or [list of] int
            Maximum number of spline control points
        soft : bool
            Return a soft (one-hot) label map

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Shared hyperparameters across tensors/channels
        shared_field : {'channels', 'tensors', 'channels+tensors', ''}
            Shared random fields across tensors/channels
        """
        super().__init__(shared=shared, **kwargs)
        self.nb_classes = RandInt.make(make_range(2, nb_classes))
        self.shape = RandInt.make(make_range(2, shape))
        self.soft = Fixed.make(soft)
        self.shared_field = self._prepare_shared(shared_field)

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' in self.shared and len(x) == 1:
            return self.make_per_channel(x, max_depth)
        nb_classes, shape, soft = self.nb_classes, self.shape, self.soft
        if isinstance(nb_classes, Sampler):
            nb_classes = nb_classes()
        if isinstance(shape, Sampler):
            shape = shape(x.ndim-1)
        if isinstance(soft, Sampler):
            soft = soft()
        shared_field = self.shared_field
        if shared_field is None:
            shared_field = self.shared
        return SmoothLabelMap(
            nb_classes, shape, soft, shared=shared_field, **self.get_prm()
        ).make_final(x, max_depth-1)


class ErodeLabelTransform(FinalTransform):
    """
    Morphological erosion.
    """

    def __init__(self, labels=tuple(), radius=3, method='conv',
                 new_labels=False, **kwargs):
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
        new_labels : bool or [sequence of] int
            If not False, the eroded portion of each label gives rise to
            a new label. If True, create new unique labels.
        """
        super().__init__(**kwargs)
        self.labels = ensure_list(labels)
        self.radius = ensure_list(radius, min(len(self.labels), 1))
        self.method = method
        self.new_labels = new_labels

    def apply(self, x):
        if self.new_labels is not False:
            return self._apply_newlabels(x)

        max_radius = max(self.radius)
        if self.method == 'conv':
            def dist(x, r):
                return bounded_distance(
                    x, nb_iter=int(round(r)), dim=x.dim()-1
                )
            dtype = torch.int
            dmax = max_radius+1
        elif self.method == 'l1':
            def dist(x, r):
                return distmap.l1_signed_transform(
                    x, ndim=x.dim()-1
                ).neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        elif self.method == 'l2':
            def dist(x, r):
                return distmap.euclidean_signed_transform(
                    x, ndim=x.dim()-1
                ).neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        else:
            raise ValueError('Unknown method', self.method)

        all_labels = x.unique().tolist()
        foreground_labels = self.labels
        if not foreground_labels:
            foreground_labels = [lab for lab in all_labels if lab != 0]
        foreground_radius = ensure_list(self.radius, len(foreground_labels))

        y = torch.zeros_like(x)
        d = torch.full_like(x, dmax, dtype=dtype)
        for label in all_labels:
            is_foreground = label in foreground_labels
            if is_foreground:
                index = foreground_labels.index(label)
                radius = foreground_radius[index]
            else:
                radius = max_radius
            x0 = x == label
            d1 = dist(x0, radius)
            if is_foreground:
                mask = d1 < -radius
            else:
                mask = d1 < d
            d = torch.where(mask, d1, d)
            y.masked_fill_(mask, label)
        return y

    def _apply_newlabels(self, x):
        if self.method == 'conv':
            def dist(x, r):
                return bounded_distance(
                    x, nb_iter=int(round(r)), dim=x.dim()-1
                )
        elif self.method == 'l1':
            def dist(x, r):
                return distmap.l1_distance_transform(
                    x, ndim=x.dim()-1
                ).neg_()
        elif self.method == 'l2':
            def dist(x, r):
                return distmap.euclidean_distance_transform(
                    x, ndim=x.dim()-1
                ).neg_()
        else:
            raise ValueError('Unknown method', self.method)

        all_labels = x.unique().tolist()
        foreground_labels = self.labels
        if not foreground_labels:
            foreground_labels = [lab for lab in all_labels if lab != 0]
        foreground_radius = ensure_list(self.radius, len(foreground_labels))
        if self.new_labels is True:
            max_label = all_labels[-1]
            new_labels = list(range(max_label+1,
                                    max_label+len(foreground_labels)+1))
        else:
            new_labels = ensure_list(self.new_labels, len(foreground_labels))

        y = x.clone()
        for label, olabel, radius in zip(
                foreground_labels, new_labels, foreground_radius):
            x0 = x == label
            d1 = dist(x0, radius)
            y.masked_fill_(d1 < -radius, label)
            y.masked_fill_((d1 < 0) & (d1 >= -radius), olabel)
        return y


class DilateLabelTransform(FinalTransform):
    """
    Morphological dilation.
    """

    def __init__(self, labels=tuple(), radius=3, method='conv', **kwargs):
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
        super().__init__(**kwargs)
        self.labels = ensure_list(labels)
        self.radius = ensure_list(radius, min(len(self.labels), 1))
        self.method = method

    def apply(self, x):
        max_radius = max(self.radius)
        if self.method == 'conv':
            def dist(x, r):
                return bounded_distance(
                    x, nb_iter=int(round(r)), dim=x.dim()-1
                )
            dtype = torch.int
            dmax = max_radius+1
        elif self.method == 'l1':
            def dist(x, r):
                d = distmap.l1_signed_transform(x, ndim=x.dim()-1)
                return d.neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        elif self.method == 'l2':
            def dist(x, r):
                d = distmap.euclidean_signed_transform(x, ndim=x.dim()-1)
                return d.neg_()
            dtype = torch.get_default_dtype()
            dmax = float('inf')
        else:
            raise ValueError('Unknown method', self.method)

        all_labels = x.unique().tolist()
        foreground_labels = self.labels
        if not foreground_labels:
            foreground_labels = [lab for lab in all_labels if lab != 0]
        foreground_radius = ensure_list(self.radius, len(foreground_labels))

        y = x.clone()
        d = torch.full_like(x, dmax, dtype=dtype)
        for label in foreground_labels:
            radius = foreground_radius[foreground_labels.index(label)]
            x0 = x == label
            d1 = dist(x0, radius)
            mask = d1 < radius
            mask = mask & (d1 < d)
            d = torch.where(mask, d1, d)
            y.masked_fill_(mask, label)
        return y


class RandomErodeLabelTransform(NonFinalTransform):
    """
    Morphological erosion with random radius/labels.
    """

    def __init__(self, labels=0.5, radius=3, method='conv',
                 new_labels=False, *, shared=False, **kwargs):
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
        new_labels : bool or [sequence of] int
            If not False, the eroded portion of each label gives rise to
            a new label. If True, create new unique labels.
        """
        super().__init__(shared=shared, **kwargs)
        self.labels = labels
        self.radius = Uniform.make(make_range(0, radius))
        self.method = Fixed(method)
        self.new_labels = Fixed(new_labels)

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)
        n = None
        labels, radius = self.labels, self.radius
        if isinstance(labels, float):
            prob = labels
            labels = x.unique().tolist()
            n = int(pymath.ceil(len(labels) * prob))
            labels = RandKFrom(labels, n)
        if isinstance(radius, Sampler) and n is None:
            if isinstance(labels, Sampler):
                labels = labels()
            n = len(labels)

        method, new_labels = self.method, self.new_labels
        if isinstance(radius, Sampler):
            radius = radius(n)
        if isinstance(labels, Sampler):
            labels = labels()
        if isinstance(method, Sampler):
            method = method()
        if isinstance(new_labels, Sampler):
            new_labels = new_labels()

        return ErodeLabelTransform(
            labels, radius, method, new_labels, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomDilateLabelTransform(NonFinalTransform):
    """
    Morphological dilation with random radius/labels.
    """

    def __init__(self, labels=0.5, radius=3, method='conv',
                 *, shared=False, **kwargs):
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
        super().__init__(shared=shared, **kwargs)
        self.labels = labels
        self.radius = Uniform.make(make_range(0, radius))
        self.method = Fixed(method)

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)
        n = None
        labels, radius = self.labels, self.radius
        if isinstance(labels, float):
            prob = labels
            labels = x.unique().tolist()
            n = int(pymath.ceil(len(labels) * prob))
            labels = RandKFrom(labels, n)
        if isinstance(radius, Sampler) and n is None:
            if isinstance(labels, Sampler):
                labels = labels()
            n = len(labels)

        method = self.method
        if isinstance(radius, Sampler):
            radius = radius(n)
        if isinstance(labels, Sampler):
            labels = labels()
        if isinstance(method, Sampler):
            method = method()

        return DilateLabelTransform(
            labels, radius, method, **self.get_prm()
        ).make_final(x, max_depth-1)


class SmoothMorphoLabelTransform(NonFinalTransform):
    """
    Morphological erosion with spatially varying radius.

    !!! note
        We're actually computing the level set of each label and pushing it
        up and down using a smooth "radius" map. In theory, this can
        create "holes" or "islands", which would not happen with a normal
        erosion. With radii that are small and radius map that are smooth
        compared to the label size, it should be fine.

    !!! warning
        The radius maps are sampled for each tensor. It is impossible
        to share them across tensors. The only `shared` options are
        therefore `channels` or `False`.
    """

    def __init__(self, labels=tuple(), min_radius=-3, max_radius=3, shape=5,
                 method='conv', shared=False, **kwargs):
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
        super().__init__(shared=shared, **kwargs)
        self.labels = ensure_list(labels)
        self.min_radius = ensure_list(min_radius, min(len(self.labels), 1))
        self.max_radius = ensure_list(max_radius, min(len(self.labels), 1))
        self.shape = shape
        self.method = method

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        return self.Final(
            AddFieldTransform(self.shape, shared=self.shared), self.labels,
            self.min_radius, self.max_radius, self.method, **self.get_prm()
        ).make_final(x, max_depth-1)

    class Final(FinalTransform):

        def __init__(self, fields, labels, min_radius=-30, max_radius=3,
                     method='conv', **kwargs):
            super().__init__(**kwargs)
            self.fields = fields
            self.labels = ensure_list(labels)
            self.min_radius = ensure_list(min_radius, min(len(self.labels), 1))
            self.max_radius = ensure_list(max_radius, min(len(self.labels), 1))
            self.method = method

        def apply(self, x):
            max_abs_radius = 1 + int(pymath.ceil(max(
                max(map(abs, self.min_radius)),
                max(map(abs, self.max_radius))
            )))
            if self.method == 'conv':
                def dist(x):
                    return bounded_distance(
                        x, nb_iter=max_abs_radius, dim=x.dim()-1
                    )
            elif self.method == 'l1':
                def dist(x):
                    return distmap.l1_signed_transform(
                        x, ndim=x.dim()-1
                    ).neg_()
            elif self.method == 'l2':
                def dist(x):
                    return distmap.euclidean_signed_transform(
                        x, ndim=x.dim()-1
                    ).neg_()
            else:
                raise ValueError('Unknown method', self.method)

            all_labels = x.unique()
            foreground_labels = self.labels
            if not foreground_labels:
                foreground_labels = all_labels[all_labels != 0].tolist()
            foreground_min_radius = ensure_list(
                self.min_radius, len(foreground_labels)
            )
            foreground_max_radius = ensure_list(
                self.max_radius, len(foreground_labels)
            )

            dtype = x.dtype
            if not dtype.is_floating_point:
                dtype = torch.get_default_dtype()

            y = torch.zeros_like(x)
            z = x.new_zeros([], dtype=dtype).expand(x.shape)
            d = torch.full_like(x, float('inf'), dtype=dtype)
            all_labels = x.unique()
            for label in all_labels:
                x0 = x == label
                is_foreground = label in foreground_labels
                if is_foreground:
                    label_index = foreground_labels.index(label)
                    min_radius = foreground_min_radius[label_index]
                    max_radius = foreground_max_radius[label_index]
                    radius = self.fields(z)
                    radius.mul_(max_radius - min_radius).add_(min_radius)
                else:
                    radius = 0
                d1 = dist(x0).to(d).sub_(radius)
                mask = d1 < d
                if mask.any():
                    d = torch.where(mask, d1, d)
                    y.masked_fill_(mask, label)
            return prepare_output(dict(input=x, output=y), self.returns)


class RandomSmoothMorphoLabelTransform(NonFinalTransform):
    """
    Morphological erosion/dilation with smooth random radius/labels.
    """

    def __init__(self, labels=0.5, min_radius=-3, max_radius=3,
                 shape=5, method='conv',
                 *, shared=False, shared_fields=None, **kwargs):
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
        if isinstance(min_radius, (float, int)):
            min_radius = (make_range(min_radius, 0) if min_radius < 0
                          else make_range(0, min_radius))
        if isinstance(max_radius, (float, int)):
            max_radius = (make_range(max_radius, 0) if max_radius < 0
                          else make_range(0, max_radius))
        super().__init__(shared=shared, **kwargs)
        self.labels = labels
        self.min_radius = Uniform.make(min_radius)
        self.max_radius = Uniform.make(max_radius)
        self.shape = RandInt.make(make_range(2, shape))
        self.method = Fixed(method)
        self.shared_fields = shared_fields

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)

        n = None
        labels = self.labels
        min_radius, max_radius = self.min_radius, self.max_radius
        if isinstance(labels, float):
            prob = labels
            labels = x.unique().tolist()
            n = int(pymath.ceil(len(labels) * prob))
            labels = RandKFrom(labels, n)
        if isinstance(min_radius, Sampler) or \
                isinstance(max_radius, Sampler) and n is None:
            if isinstance(labels, Sampler):
                labels = labels()
            n = len(labels)

        shape = self.shape
        method = self.method
        if isinstance(min_radius, Sampler):
            min_radius = min_radius(n)
        if isinstance(max_radius, Sampler):
            max_radius = max_radius(n)
        if isinstance(labels, Sampler):
            labels = labels()
        if isinstance(shape, Sampler):
            shape = shape(x.ndim-1)
        if isinstance(method, Sampler):
            method = method()
        shared_fields = self.shared_fields
        if shared_fields is None:
            shared_fields = self.shared

        return SmoothMorphoLabelTransform(
            labels, min_radius, max_radius, shape, method,
            shared=shared_fields, **self.get_prm()
        ).make_final(x, max_depth-1)


class SmoothShallowLabelTransform(NonFinalTransform):
    """Make labels "empty", with a border of a given size."""

    def __init__(self, labels=tuple(), max_width=5, min_width=1, shape=5,
                 background_labels=tuple(), method='l2',
                 *, shared=False, **kwargs):
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
        super().__init__(shared=shared, **kwargs)
        self.labels = ensure_list(labels)
        self.background_labels = ensure_list(background_labels)
        self.shape = shape
        self.min_width = ensure_list(min_width, min(len(self.labels), 1))
        self.max_width = ensure_list(max_width, min(len(self.labels), 1))
        self.method = method

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        xform = AddFieldTransform(self.shape, shared=self.shared)
        return self.Final(
            xform, self.labels, self.max_width, self.min_width,
            self.background_labels, self.method, **self.get_prm()
        ).make_final(x, max_depth-1)

    class Final(FinalTransform):

        def __init__(self, fields, labels=tuple(), max_width=5, min_width=1,
                     background_labels=tuple(), method='l2', **kwargs):
            super().__init__(**kwargs)
            self.fields = fields
            self.labels = labels
            self.max_width = max_width
            self.min_width = min_width
            self.background_labels = background_labels
            self.method = method

        def apply(self, x):
            if self.method == 'l1':
                def dist(x):
                    return distmap.l1_signed_transform(
                        x, ndim=x.dim()-1
                    ).neg_()
            elif self.method == 'l2':
                def dist(x):
                    return distmap.euclidean_signed_transform(
                        x, ndim=x.dim()-1
                    ).neg_()
            else:
                raise ValueError('Unknown method', self.method)

            all_labels = x.unique()
            foreground_labels = self.labels
            if not foreground_labels:
                foreground_labels = all_labels[all_labels != 0]
            background_labels = self.background_labels
            if not background_labels:
                background_labels = [
                    lab for lab in all_labels if lab not in foreground_labels
                ]
            foreground_min_width = ensure_list(
                self.min_width, len(foreground_labels)
            )
            foreground_max_width = ensure_list(
                self.max_width, len(foreground_labels)
            )

            dtype = x.dtype
            if not x.dtype.is_floating_point:
                dtype = torch.get_default_dtype()
            y = torch.zeros_like(x)
            z = x.new_zeros([], dtype=dtype).expand(x.shape)
            d = torch.full_like(x, float('inf'), dtype=dtype)
            m = torch.zeros_like(x, dtype=torch.bool)
            all_labels = x.unique()

            # fill object borders and keep track of object interiors
            for label, min_width, max_width in zip(
                    foreground_labels,
                    foreground_min_width,
                    foreground_max_width):
                x0 = x == label
                radius = self.fields(z)
                radius.mul_(min_width-max_width).sub_(min_width)
                d1 = dist(x0).to(d)
                mask = (d1 < 0) & (d1 > radius)
                m.masked_fill_(d1 < radius, True)
                d = torch.where(mask, d1, d)
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
                    d = torch.where(mask, d1, d)
                    y.masked_fill_(mask, label)

            return prepare_output(dict(input=x, output=y), self.returns)


class RandomSmoothShallowLabelTransform(NonFinalTransform):
    """
    Make labels "empty", with a border of a given (random) size.
    """

    def __init__(self, labels=0.5, max_width=5, min_width=1, shape=5,
                 background_labels=tuple(), method='l2',
                 *, shared=False, shared_fields=None, **kwargs):
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
        super().__init__(shared=shared, **kwargs)
        self.labels = labels
        self.min_width = Uniform.make(make_range(0, min_width))
        self.max_width = Uniform.make(make_range(0, max_width))
        self.shape = RandInt.make(make_range(2, shape))
        self.background_labels = background_labels
        self.method = method
        self.shared_fields = shared_fields

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)

        n = None
        labels = self.labels
        min_width, max_width = self.min_width, self.max_width
        if isinstance(labels, float):
            prob = labels
            labels = x.unique().tolist()
            n = int(pymath.ceil(len(labels) * prob))
            labels = RandKFrom(labels, n)
        if isinstance(min_width, Sampler) or \
                isinstance(max_width, Sampler) and n is None:
            if isinstance(labels, Sampler):
                labels = labels()
            n = len(labels)

        shape = self.shape
        background_labels = self.background_labels
        method = self.method
        if isinstance(min_width, Sampler):
            min_width = min_width(n)
        if isinstance(max_width, Sampler):
            max_width = max_width(n)
        if isinstance(shape, Sampler):
            shape = shape(x.ndim-1)
        if isinstance(labels, Sampler):
            labels = labels()
        if isinstance(background_labels, Sampler):
            background_labels = background_labels()
        if isinstance(method, Sampler):
            method = method()
        shared_fields = self.shared_fields
        if shared_fields is None:
            shared_fields = self.shared

        return SmoothShallowLabelTransform(
            labels, max_width, min_width, shape, background_labels, method,
            shared=shared_fields, **self.get_prm()
        ).make_final(x, max_depth-1)


class BernoulliTransform(NonFinalTransform):
    """Randomly mask voxels"""

    def __init__(self, prob=0.1, *, shared=False, **kwargs):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensor to return
        shared : bool
            Same mask shared across channels
        """
        super().__init__(shared=shared, **kwargs)
        self.prob = prob

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        batch, *shape = x.shape
        if 'channels' in self.shared:
            batch = 1
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        pmap = torch.rand([batch, *shape], device=x.device, dtype=dtype)
        return MulValueTransform(
            pmap <= self.prob,
            value_name='noise', **self.get_prm()
        ).make_final(x, max_depth-1)


class SmoothBernoulliTransform(NonFinalTransform):
    """Randomly mask voxels"""

    def __init__(self, prob=0.1, shape=5,
                 *, shared=False, shared_noise=False, **kwargs):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
        shape : int or sequence[int]
            Number of control points in the smooth field
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensor to return
        shared : bool
            Same probability field shared across channels/tensor
        shared_noise : bool
            Same mask shared across channels/tensor
        """
        super().__init__(shared=shared, **kwargs)
        self.shape = shape
        self.prob = prob
        self.shared_noise = self._prepare_shared(shared_noise)

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self

        ndim = x.dim() - 1
        batch, *shape = x.shape
        if 'channels' in self.shared:
            batch = 1
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()

        z = x.new_zeros([], dtype=dtype, device=x.device)
        z = z.expand([batch, *shape])

        prob = AddFieldTransform(self.shape, shared=self.shared)(z)
        prob /= prob.sum(list(range(-ndim, 0)), keepdim=True)
        prob *= self.prob * x.shape[1:].numel()

        shared_noise = self.shared_noise
        if shared_noise is None:
            shared_noise = self.shared
        batch, *shape = x.shape
        if 'channels' in shared_noise:
            batch = 1
        pmap = torch.rand([batch, *shape], device=x.device, dtype=dtype)

        return MulValueTransform(
            pmap >= self.prob,
            value_name='noise', **self.get_prm()
        ).make_final(x, max_depth-1)


class BernoulliDiskTransform(NonFinalTransform):
    """Randomly mask voxels in balls at random locations"""

    def __init__(self, prob=0.1, radius=2, value=0, method='l2',
                 *, shared=False, **kwargs):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
        radius : float or Sampler
            Disk radius
        value : float or int or {'min', 'max', 'rand'}
            Value to set in the disks
        method : {'conv', 'l1', 'l2'}
            Method used to compute the distance map
        returns : [list or dict of] {'input', 'output', 'disks'}
            Which tensor to return
        shared : bool
            Same mask shared across channels
        """
        super().__init__(shared=shared, **kwargs)
        self.prob = prob
        self.radius = radius
        self.value = value
        self.method = method

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        ndim = x.dim() - 1
        nvoxball = pymath.pow(pymath.pi, ndim/2) / pymath.gamma(ndim/2 + 1)

        nvoxball *= self.radius
        nvoxball = max(nvoxball, 1)

        # sample locations
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()

        # dilate disks
        mask = torch.rand_like(x, dtype=dtype) > (1 - self.prob / nvoxball)
        mask = DilateLabelTransform(
            method=self.method, radius=self.radius
        )(mask)

        # set output value
        value = self.value
        if isinstance(value, str):
            if value == 'max':
                value = x.max()
            elif value == 'min':
                value = x.min()
            elif value.startswith('rand'):
                mn, mx = x.min().item(), x.max().item()
                if x.dtype.is_floating_point:
                    value = Uniform(mn, mx)()
                else:
                    value = RandInt(mn, mx)()
            else:
                raise ValueError('Unknown value mode: {}', value)

        return FillValueTransform(
            mask, value, mask_name='disks', **self.get_prm(),
        ).make_final(x, max_depth-1)


class SmoothBernoulliDiskTransform(NonFinalTransform):
    """Randomly mask voxels in balls at random locations"""

    def __init__(self, prob=0.1, radius=2, shape=5, value=0, method='l2',
                 *, shared=False, shared_field=None, **kwargs):
        """
        Parameters
        ----------
        prob : float
            Probability of masking out a voxel
            A probability field is sampled from a smooth random field.
        radius : float or (float, float)
            Max or Min/Max disk radius, sampled from a smooth random field.
        shape : int or sequence[int]
            Number of control points in the random field
        method : {'conv', 'l1', 'l2'}
            Method used to compute the distance map
        returns : [list or dict of] {'input', 'output', 'disks'}
            Which tensor to return
        shared : bool
            Same mask shared across channels
        """
        super().__init__(shared=shared, **kwargs)
        self.prob = prob
        self.shared_field = self._prepare_shared(shared_field)
        self.field = AddFieldTransform(shape=shape)
        min_radius, max_radius = 0, radius
        if isinstance(max_radius, (list, tuple)):
            min_radius, max_radius = max_radius
        self.dilate = SmoothMorphoLabelTransform(
            min_radius=min_radius, max_radius=max_radius,
            shape=shape, method=method)
        self.radius = (
          (0, radius) if not isinstance(radius, (list, tuple)) else
          tuple(radius) if len(radius) == 2 else
          (0, radius[0]) if len(radius) == 1 else
          (0, 2)
        )
        self.value = value

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        ndim = x.dim() - 1
        nvoxball = pymath.pow(pymath.pi, ndim / 2)
        nvoxball /= pymath.gamma(ndim / 2 + 1)
        nvoxball *= sum(self.radius) / 2
        nvoxball = max(nvoxball, 1)

        shared_field = self.shared_field
        if shared_field is None:
            shared_field = self.shared
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        fake_x = x.new_zeros([], dtype=dtype).expand(x.shape)

        # sample seeds
        with ctx.shared(self.field, shared_field):
            prob = self.field(fake_x)
        prob /= prob.sum(list(range(-ndim, 0)), keepdim=True)
        prob *= self.prob * x.shape[1:].numel() / nvoxball

        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        mask = torch.rand_like(x, dtype=dtype) > (1 - prob)

        # dilate balls
        with ctx.shared(self.dilate, shared_field):
            mask = self.dilate(mask)

        # set output value
        value = self.value
        if isinstance(value, str):
            if value == 'max':
                value = x.max()
            elif value == 'min':
                value = x.min()
            elif value.startswith('rand'):
                mn, mx = x.min().item(), x.max().item()
                if x.dtype.is_floating_point:
                    value = Uniform(mn, mx)()
                else:
                    value = RandInt(mn, mx)()
            else:
                raise ValueError('Unknown value mode: {}', value)

        return FillValueTransform(
            mask, value, mask_name='disks', **self.get_prm(),
        ).make_final(x, max_depth-1)
