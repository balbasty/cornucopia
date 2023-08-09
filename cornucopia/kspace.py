__all__ = [
    'ArrayCoilTransform',
    'SumOfSquaresTransform',
    'IntraScanMotionTransform',
    'SmallIntraScanMotionTransform',
]

import torch
import math
import random
from .base import Transform, prepare_output
from .intensity import MulFieldTransform
from .geometric import RandomAffineTransform
from .random import Sampler, Fixed
from .utils.py import cartesian_grid


class ArrayCoilTransform(Transform):
    """Generate and apply random coil sensitivities (real or complex)"""

    def __init__(self, ncoils=8, fwhm=0.5, diameter=0.8, jitter=0.01,
                 unit='fov', shape=4, sos=True, *, shared=True, **kwargs):
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
        returns : [list or dict of] {'input', 'sos', 'uncombined', 'sens', 'netsens'}
            Default is 'uncombined'.

            - 'sos': Sum of square combined (magnitude) image
            - 'uncombined': Uncombined (complex) coil images
            - 'sens': Uncombined (complex) coil sensitivities
            - 'netsens': Net (magnitude) coil sensitivity
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(shared=shared, **kwargs)
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

        smooth_bias = MulFieldTransform(shape=self.shape, vmin=-1, vmax=1)
        smooth_bias = smooth_bias.get_parameters(fake_x)
        phase = smooth_bias[::2].atan2(smooth_bias[1::2])
        magnitude = (smooth_bias[::2].square() + smooth_bias[1::2].square()).sqrt_()

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
        return (1j * phase).exp_().mul_(magnitude)

    def apply_transform(self, x, parameters):
        uncombined = x * parameters
        netsens = parameters.abs().square().sum(0).sqrt_()[None]
        sos = uncombined.abs().square().sum(0).sqrt_()[None]
        return prepare_output(
            dict(input=x, sos=sos, output=uncombined, uncombined=uncombined,
                 netsens=netsens, sens=parameters),
            self.returns)


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
                 *,
                 shared='channels',
                 **kwargs):
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
            are randomly distributed across shots.
        translations : Sampler or float
            Sampler (or upper-bound) for random translations (in % of FOV)
        rotations : Sampler or float
            Sampler (or upper-bound) for random rotations (in deg)
        coils : Transform
            A transform that generates a set of complex sensitivity profiles
        returns : [list or dict of] {'input', 'sos', 'uncombined', 'sens', 'netsens', 'flow', 'matrix', 'pattern'}
            Default is 'sos'

            - 'sos': Sum of square combined (magnitude) image `(C, X, Y, Z)`
            - 'uncombined': Uncombined (complex) coil images `(K, X, Y, Z)`
            - 'sens': Uncombined (complex) coil sensitivities `(K, X, Y, Z)`
            - 'netsens': Net (magnitude) coil sensitivity `(1, X, Y, Z)`
            - 'flow': Displacement field in each shot `(N, 3, X, Y, Z)`
            - 'matrix': Rigid matrix in each shot `(N, 4, 4)`
            - 'pattern': Frequencies acquired in each shot `(N, X)`
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(shared=shared, **kwargs)
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
            pattern = torch.stack(pattern)
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
            pattern = torch.stack(pattern)
        elif isinstance(self.pattern, (list, tuple)):
            if max(map(max, self.pattern)) > 1:
                # indices
                mask = torch.zeros(n, dtype=torch.bool, device=device)
                for indices in self.pattern:
                    mask1 = mask.clone()
                    mask1[indices] = 1
                    pattern.append(mask1)
                pattern = torch.stack(pattern)
            else:
                pattern = torch.stack(pattern)
        else:
            assert torch.is_tensor(self.pattern)
            pattern = self.pattern
        return pattern.to(device, torch.bool)

    def get_parameters(self, x):

        shots = min(self.shots, x.shape[self.axis])

        parameters = {}
        parameters['motion'] = []
        for shot in range(shots):
            motion_trf = self.motion
            motion_prm = motion_trf.get_parameters(x)
            while isinstance(motion_prm, Transform):
                motion_trf = motion_prm
                motion_prm = motion_trf.get_parameters(x)
            parameters['motion'].append([motion_trf, motion_prm])

        parameters['pattern'] = self.get_pattern(x.shape[self.axis], x.device)

        if self.coils:
            coils = self.coils.get_parameters(x)
            while isinstance(coils, Transform):
                coils = coils.get_parameters(x)
            parameters['coils'] = coils

        return parameters

    def apply_transform(self, x, parameters):

        motions = parameters['motion']
        patterns = parameters['pattern']

        if 'coils' in parameters:
            assert len(x) == 1
            sens = parameters['coils']
            netsens = sens.abs().square_().sum(0).sqrt_()[None]
            y = x.new_empty([len(sens), *x.shape[1:]], dtype=torch.complex64)
        else:
            y = torch.empty_like(x, dtype=torch.complex64)
            sens = netsens = None

        x = x.movedim(self.axis, 1)
        y = y.movedim(self.axis, 1)

        matrix = torch.stack(list(map(lambda x: x[1][1], motions)))
        flow = torch.stack(list(map(lambda x: x[1][0], motions)))
        for (motion_trf, motion_prm), pattern in zip(motions, patterns):
            moved = motion_trf.apply_transform(x, motion_prm)
            if sens is not None:
                moved = moved * sens
            if self.freq:
                moved = torch.fft.ifftshift(moved)
                moved = torch.fft.fft(moved, dim=1)
                moved = torch.fft.fftshift(moved)
            y[:, pattern] = moved[:, pattern]

        if self.freq:
            y = torch.fft.ifftshift(y)
            y = torch.fft.ifft(y, dim=1)
            y = torch.fft.fftshift(y)
        y = y.movedim(1, self.axis)
        x = x.movedim(1, self.axis)

        sos = y.abs().square_().sum(0, keepdim=True).sqrt_()

        return prepare_output(
            dict(input=x, sos=sos, output=sos, uncombined=y, sens=sens,
                 netsens=netsens, pattern=patterns, flow=flow, matrix=matrix),
            self.returns)


class SmallIntraScanMotionTransform(IntraScanMotionTransform):
    """Model intra-scan motion that happens once across k-space"""

    def __init__(self, translations=0.05, rotations=5, axis=-1,
                 *, shared='channels', **kwargs):
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
                         shared=shared, shots=2, axis=axis, **kwargs)

    def get_pattern(self, n, device=None):
        k = random.randint(0, n-1)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[:k] = 1
        return [mask, ~mask]

