"""This module contains transforms that operate on image contrasts."""
__all__ = [
    'ContrastMixtureTransform',
    'ContrastMixtureFinalTransform',
    'ContrastLookupTransform',
    'ContrastLookupFinalTransform'
]
# stdlib
from math import inf

# dependencies
import torch
from torch import Tensor

# internals
from .base import NonFinalTransform, FinalTransform, Transform
from .special import PerChannelTransform
from .utils.gmm import fit_gmm


class ContrastMixtureFinalTransform(FinalTransform):
    """Classwise shift and rescaling."""

    def __init__(
        self, z: Tensor,
        mu0: Tensor, sigma0: Tensor, mu: Tensor, sigma: Tensor,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        z : (K, *spatial) tensor
            Probability that each voxel belongs to a given class.
        mu0 : (K, C) tensor
            Original means for each class.
        sigma0 : (K, C, C) tensor
            Original covariances for each class.
        mu : (K, C) tensor
            New means for each class.
        sigma : (K, C, C) tensor
            New covariances for each class.

        Other Parameters
        ----------------
        returns, append, prefix, include, exclude, consume
            See [`Transform`][cornucopia.base.Transform] for details.
        """
        super().__init__(**kwargs)
        self.z = z
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu = mu
        self.sigma = sigma

    def xform(self, x: Tensor) -> Tensor:
        z = self.z.to(x)
        mu0 = self.mu0.to(x)
        sigma0 = self.sigma0.to(x)
        mu = self.mu.to(x)
        sigma = self.sigma.to(x)

        # Whiten using fitted parameters
        chol = torch.linalg.cholesky(sigma0)
        chol = chol.inverse()
        x = x.movedim(0, -1)
        x = x[..., None, :] - mu0                       # [..., nk, nc]
        x = torch.matmul(chol, x[..., :, None])         # [..., nk, nc, 1]

        # Color using new parameters
        chol = torch.linalg.cholesky(sigma)
        x = torch.matmul(chol, x)                        # [..., nk, nc, 1]
        x = x[..., 0] + mu                               # [..., nk, nc]
        x = x.movedim(-1, 0)                             # [..., nc, nk]

        # Weight using posterior
        z = z.movedim(0, -1)
        x = torch.matmul(x[..., None, :], z[..., None])  # [..., nc, 1, 1]
        x = x[..., 0, 0]                                 # [..., nc]

        return x


class ContrastMixtureTransform(NonFinalTransform):
    """
    Find intensity modes using a GMM and change their means and covariances.

    ??? reference
        Meyer, M.I., de la Rosa, E., Pedrosa de Barros, N., Paolella, R.,
        Van Leemput, K. and Sima, D.M., 2021.
        [**A contrast augmentation approach to improve multi-scanner
        generalization in MRI.**](https://www.frontiersin.org/articles/10.3389/fnins.2021.708196)
        Frontiers in Neuroscience, 15, p.708196.

            @article{meyer2021,
                title     = {A contrast augmentation approach to improve multi-scanner generalization in MRI},
                author    = {Meyer, Maria Ines and de la Rosa, Ezequiel and Pedrosa de Barros, Nuno and Paolella, Roberto and Van Leemput, Koen and Sima, Diana M},
                journal   = {Frontiers in Neuroscience},
                volume    = {15},
                pages     = {708196},
                year      = {2021},
                publisher = {Frontiers Media SA},
                url       = {https://www.frontiersin.org/articles/10.3389/fnins.2021.708196}
            }

    """

    Final = Next = ContrastMixtureFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        nk: int = 16,
        keep_background: bool = True,
        *,
        shared: bool = False,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        nk : int
            Number of classes
        keep_background : bool
            Do not change background mean/cov.
            The background class is the class with minimum mean value.

        Other Parameters
        ----------------
        shared
            See [`NonFinalTransform`][cornucopia.base.NonFinalTransform]
            for details.
        returns, append, prefix, include, exclude, consume
            See [`Transform`][cornucopia.base.Transform] for details.
        """
        super().__init__(shared=shared, **kwargs)
        self.keep_background = keep_background
        self.nk = nk

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        z, mu0, sigma0, _ = fit_gmm(x, self.nk)
        mu, sigma = self._make_parameters(mu0, sigma0)
        return self.Next(
            z, mu0, sigma0, mu, sigma, **self.get_prm()
        ).make_final(x, max_depth-1)

    def _make_parameters(self, old_mu, old_sigma):
        backend = dict(dtype=old_mu.dtype, device=old_mu.device)
        nk, nc = old_mu.shape
        old_mu_min = old_mu.min(0).values
        old_mu_max = old_mu.max(0).values
        old_sigma_diag = old_sigma.diagonal(0, -1, -2)
        old_sigma_min = old_sigma_diag.min(0).values.sqrt()
        old_sigma_max = old_sigma_diag.max(0).values.sqrt()

        mu = torch.rand_like(
            old_mu).mul_(old_mu_max - old_mu_min).add_(old_mu_min)
        sigma = torch.rand_like(
            old_sigma_diag
        ).mul_(old_sigma_max - old_sigma_min).add_(old_sigma_min)
        corr = torch.rand([len(old_mu), nc*(nc-1)//2], **backend).mul_(0.5)

        fullsigma = torch.eye(nc, **backend).expand([nk, nc, nc]).clone()
        cnt = 0
        for i in range(nc):
            for j in range(i+1, nc):
                fullsigma[:, i, j] = fullsigma[:, j, i] = corr[:, cnt]
                cnt += 1
        fullsigma = fullsigma * sigma[:, :, None] * sigma[:, None, :]

        if self.keep_background:
            idx = old_mu.square().sum(-1).sqrt().min(0).indices
            mu[idx] = old_mu[idx]
            fullsigma[idx] = old_sigma[idx]

        return mu, fullsigma


class ContrastLookupFinalTransform(FinalTransform):
    """Binwise intensity shift."""

    def __init__(self, edges: Tensor, mu: Tensor, **kwargs) -> None:
        """
        Parameters
        ----------
        edges : (K+1,) tensor
            The limits of each input bin.
        mu : (K,) tensor
            The new mean value for each bin.

        Other Parameters
        ----------------
        returns, append, prefix, include, exclude, consume
            See [`Transform`][cornucopia.base.Transform] for details.
        """
        super().__init__(**kwargs)
        self.edges = edges
        self.mu = mu

    def xform(self, x: Tensor) -> Tensor:
        edges, mu = self.edges.to(x), self.mu.to(x)
        mu0 = (edges[:-1] + edges[1:]) / 2
        nk = len(mu)

        new_x = x.clone()
        for k in range(nk):
            mask = (edges[k] <= x) & (x < edges[k+1])
            new_x[mask] += mu[k] - mu0[k]
        return new_x


class ContrastLookupTransform(NonFinalTransform):
    """
    Segment intensities into equidistant bins and change their mean value.
    """

    Final = Next = ContrastLookupFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(self, nk=16, keep_background=True,
                 *, shared=False, **kwargs):
        """

        Parameters
        ----------
        nk : int
            Number of classes
        keep_background : bool
            Do not change background mean/cov.
            The background class is the class with minimum mean value.

        Other Parameters
        ----------------
        shared
            See [`NonFinalTransform`][cornucopia.base.NonFinalTransform]
            for details.
        returns, append, prefix, include, exclude, consume
            See [`Transform`][cornucopia.base.Transform] for details.
        """
        super().__init__(shared=shared, **kwargs)
        self.keep_background = keep_background
        self.nk = nk

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return PerChannelTransform(
                [self.make_final(x[i:i+1], max_depth) for i in range(len(x))],
                **self.get_prm()
            ).make_final(x, max_depth-1)

        vmin, vmax = x.min(), x.max()
        edges = torch.linspace(vmin, vmax, self.nk+1)
        new_mu = torch.rand(self.nk).to(x) * (vmax - vmin) + vmin
        return self.Next(
            edges, new_mu, **self.get_prm()
        ).make_final(x, max_depth-1)
