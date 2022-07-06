import torch
from .base import Transform
from .intensity import MultFieldTransform
from .utils.py import cartesian_grid


class ArrayCoilTransform(Transform):

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
            magnitude, phase = parameters
            return x * magnitude * (1j * phase).exp_()







