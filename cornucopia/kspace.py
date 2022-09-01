__all__ = ['ArrayCoilTransform', 'SumOfSquaresTransform',
           'IntraScanMotionTransform', 'SmallIntraScanMotionTransform']

import torch
import math
import random
from .base import Transform
from .intensity import MultFieldTransform
from .geometric import RandomAffineTransform
from .random import Fixed
from .utils.py import cartesian_grid


class ArrayCoilTransform(Transform):
    """Generate and apply random coil sensitivities (real or complex)"""

    def __init__(self, ncoils=8, fwhm=0.5, diameter=0.8, jitter=0.01,
                 unit='fov', shape=4, sos=True, shared=True):
        """

        Parameters
        ----------
        ncoils : int
            Number of complex receiver channels
        fwhm : float
            Width of each receiver profile
        diameter : float
            Diameter of the ellipsoid on wich receivers are centered
        jitter : float
            Amount of jitter off the ellipsoid
        unit : {'fov', 'vox'}
            Unit of `fwhm`, `diameter`, `jitter`
        shape : int
            Number of control points for the underlying smooth component.
        sos : bool
            Return the sum-of-square image.
            Otherwise, return individual complex coil images.
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(shared=shared)
        self.ncoils = ncoils
        self.fwhm = fwhm
        self.diameter = diameter
        self.jitter = jitter
        self.unit = unit
        self.shape = shape
        self.sos = sos

    def get_parameters(self, x):

        ndim = x.dim() - 1
        backend = dict(dtype=x.dtype, device=x.device)
        fake_x = torch.zeros([], **backend).expand([2*self.ncoils, *x.shape[1:]])

        smooth_bias = MultFieldTransform(shape=self.shape, vmin=-1, vmax=1)
        smooth_bias = smooth_bias.get_parameters(fake_x)
        phase = smooth_bias[::2].atan2(smooth_bias[1::2])
        magnitude = smooth_bias[::2].square() + smooth_bias[1::2].square()

        fov = torch.as_tensor(x.shape[1:], **backend)
        grid = torch.stack(cartesian_grid(x.shape[1:], **backend), -1)
        fwhm = self.fwhm
        if self.unit == 'fov':
            fwhm = fwhm * fov
        lam = (2.355 / fwhm) ** 2
        for k in range(self.ncoils):

            loc = torch.randn(ndim, **backend)
            loc /= loc.square().sum().sqrt()
            loc *= self.diameter
            if self.jitter:
                jitter = torch.rand(ndim, **backend) * self.jitter
                loc += jitter
            loc = (1 + loc) / 2
            if self.unit == 'fov':
                loc *= fov
            exp_bias = (grid-loc).square_().mul_(lam).sum(-1).mul_(-0.5).exp_()
            magnitude[k] *= exp_bias

        if self.sos:
            return magnitude.sum(0).sqrt_()
        else:
            return magnitude, phase

    def apply_transform(self, x, parameters):
        if self.sos:
            return x.abs().mul_(parameters)
        else:
            assert len(x) == 1
            magnitude, phase = parameters
            return x * magnitude * (1j * phase).exp_()


class SumOfSquaresTransform(Transform):
    """Compute the sum-of-squares across coils/channels"""

    def apply_transform(self, x, parameters):
        return x.abs().square_().sum(0, keepdim=True).sqrt_()


class IntraScanMotionTransform(Transform):
    """Model intra-scan motion"""

    def __init__(self,
                 shots=4,
                 axis=-1,
                 freq=True,
                 pattern='sequential',
                 translations=0.1,
                 rotations=15,
                 sos=True,
                 coils=None,
                 shared='channels'):
        """

        Parameters
        ----------
        shots : int
            Number of acquisition shots.
            The object is in a different position in each shot.
        axis : int
            Axis along which shots are acquired (slice or phase-encode)
        freq : bool
            Motion happens across a phase-encode direction, which means
            that the k-space is build from pieces with different object
            position. This typically happens in "3D" acquisitions.
            If False, motion happens along the slice direction ("2D"
            acquisition).
        pattern : {'sequential', 'random'} or list[tensor[bool or int]]
            k-space (or slice) sampling pattern. This argument encodes
            the frequencies (or slices) that are acquired in each shot.
            The 'sequential' options assumes that frequencies are
            acquired in order. The 'random' option assumes that frequencies
            are randomly ditributed across shots.
        translations : Sampler or float
            Sampler (or upper-bound) for random translations (in % of FOV)
        rotations : Sampler or float
            Sampler (or upper-bound) for random rotations (in deg)
        sos : bool
            Return the sum-of-square image
            Otherwise, return each complex coil image
        coils : Transform
            A transform that generates a set of complex sensitivity profiles
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(shared=shared)
        self.shots = shots
        self.axis = axis
        self.pattern = pattern
        self.sos = sos
        self.coils = coils
        self.freq = freq
        self.motion = RandomAffineTransform(
            translations=translations, rotations=rotations,
            zooms=Fixed(0), shears=Fixed(0), bound='reflection')

    def get_pattern(self, n, device=None):
        shots = min(self.shots, n)
        pattern = []
        if self.pattern == 'sequential':
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            length = int(math.ceil(n/self.shots))
            for shot in range(shots):
                mask1 = mask.clone()
                mask1[shot*length:(shot+1)*length] = 1
                pattern.append(mask1)
        elif self.pattern == 'random':
            indices = list(range(n))
            random.shuffle(indices)
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            length = int(math.ceil(n/self.shots))
            for shot in range(shots):
                mask1 = mask.clone()
                index1 = indices[shot*length:(shot+1)*length]
                mask1[index1] = 1
                pattern.append(mask1)
        else:
            pattern = self.pattern
        return pattern

    def get_parameters(self, x):

        shots = min(self.shots, x.shape[self.axis])

        parameters = {}
        parameters['motion'] = []
        for shot in range(shots):
            parameters['motion'].append(self.motion.get_parameters(x))

        parameters['pattern'] = self.get_pattern(x.shape[self.axis], x.device)

        if self.coils:
            sos0 = self.coils.sos
            self.coils.sos = False
            parameters['coils'] = self.coils.get_parameters(x)
            self.coils.sos = sos0

        return parameters

    def apply_transform(self, x, parameters):

        motions = parameters['motion']
        patterns = parameters['pattern']

        if 'coils' in parameters:
            assert len(x) == 1
            magnitude, phase = parameters['coils']
            sens = (1j * phase).exp_().mul_(magnitude)
            y = x.new_empty([len(sens), *x.shape[1:]], dtype=torch.complex64)
        else:
            y = torch.empty_like(x, dtype=torch.complex64)
            sens = None

        x = x.movedim(self.axis, 1)
        y = y.movedim(self.axis, 1)

        for motion, pattern in zip(motions, patterns):
            moved = self.motion.apply_transform(x, motion)
            if sens is not None:
                moved = moved * sens
            if self.freq:
                moved = torch.fft.ifftshift(moved)
                moved = torch.fft.fft(moved, dim=1)
                moved = torch.fft.fftshift(moved)
            y[:, pattern] = moved[:, pattern]

        if self.freq:
            y = torch.fft.ifftshift(y)
            y = torch.fft.fft(y, dim=1)
            y = torch.fft.fftshift(y)
        y = y.movedim(1, self.axis)
        if self.sos:
            y = y.abs().square_().sum(0, keepdim=True).sqrt_()

        return y


class SmallIntraScanMotionTransform(IntraScanMotionTransform):
    """Model intra-scan motion that happens once across k-space"""

    def __init__(self, translations=0.05, rotations=5, axis=-1,
                 shared='channels'):
        """

        Parameters
        ----------
        translations : Sampler or float
            Sampler (or upper-bound) for random translations (in % of FOV)
        rotations : Sampler or float
            Sampler (or upper-bound) for random rotations (in deg)
        axis : int
            Axis along which shots are acquired (slice or phase-encode)
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(translations=translations, rotations=rotations,
                         shared=shared, shots=2, axis=axis)

    def get_pattern(self, n, device=None):
        k = random.randint(0, n-1)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[:k] = 1
        return [mask, ~mask]
