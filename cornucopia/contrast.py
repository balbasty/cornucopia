__all__ = ['ContrastMixtureTransform', 'ContrastLookupTransform']

import torch
from .base import Transform
from .utils.gmm import fit_gmm


class ContrastMixtureTransform(Transform):
    """
    Find intensity modes using a GMM and change their means and covariances.

    References
    ----------
    1. Meyer, M.I., de la Rosa, E., Pedrosa de Barros, N., Paolella, R.,
       Van Leemput, K. and Sima, D.M., 2021.
       [**A contrast augmentation approach to improve multi-scanner generalization in MRI.**](https://www.frontiersin.org/articles/10.3389/fnins.2021.708196)
       Frontiers in Neuroscience, 15, p.708196.

            @article{meyer2021,
              title={A contrast augmentation approach to improve multi-scanner generalization in MRI},
              author={Meyer, Maria Ines and de la Rosa, Ezequiel and Pedrosa de Barros, Nuno and Paolella, Roberto and Van Leemput, Koen and Sima, Diana M},
              journal={Frontiers in Neuroscience},
              volume={15},
              pages={708196},
              year={2021},
              publisher={Frontiers Media SA},
              url={https://www.frontiersin.org/articles/10.3389/fnins.2021.708196}
            }

    """

    def __init__(self, nk=16, keep_background=True, shared='channels'):
        """

        Parameters
        ----------
        nk : int
            Number of classes
        keep_background : bool
            Do not change background mean/cov.
            The background class is the class with minimum mean value.
        """
        super().__init__(shared=shared)
        self.keep_background = keep_background
        self.nk = nk

    def apply_transform(self, x, parameters):
        z, mu0, sigma0, mu, sigma = parameters

        # Whiten using fitted parameters
        chol = torch.linalg.cholesky(sigma0)
        chol = chol.inverse()
        x = x.movedim(0, -1)
        x = x[..., None, :] - mu0                      # [..., nk, nc]
        x = torch.matmul(chol, x[..., :, None])        # [..., nk, nc, 1]

        # Color using new parameters
        chol = torch.linalg.cholesky(sigma)
        x = torch.matmul(chol, x)                       # [..., nk, nc, 1]
        x = x[..., 0] + mu                              # [..., nk, nc]
        x = x.movedim(-1, 0)                            # [..., nc, nk]

        # Weight using posterior
        z = z.movedim(0, -1)
        x = torch.matmul(x[..., None, :], z[..., None]) # [..., nc, 1, 1]
        x = x[..., 0, 0]                                # [..., nc]

        return x

    def get_parameters(self, x):
        z, mu, sigma, _ = fit_gmm(x, self.nk)
        new_mu, new_sigma = self.make_parameters(mu, sigma)
        return z, mu, sigma, new_mu, new_sigma

    def make_parameters(self, old_mu, old_sigma):
        backend = dict(dtype=old_mu.dtype, device=old_mu.device)
        nk, nc = old_mu.shape
        old_mu_min = old_mu.min(0).values
        old_mu_max = old_mu.max(0).values
        old_sigma_min = old_sigma.diagonal(0, -1, -2).min(0).values.sqrt()
        old_sigma_max = old_sigma.diagonal(0, -1, -2).max(0).values.sqrt()

        mu = []
        for c in range(nc):
            mu1 = old_mu_min + (old_mu_max - old_mu_min) * torch.rand([nk], **backend)
            mu.append(mu1)
        mu = torch.stack(mu, -1)

        sigma = []
        for c in range(nc):
            s1 = old_sigma_min + (old_sigma_max - old_sigma_min) * torch.rand([nk], **backend)
            sigma.append(s1)
        sigma = torch.stack(sigma, -1)

        corr = torch.rand([len(old_mu), nc*(nc-1)//2], **backend)

        fullsigma = torch.eye(nc, **backend)
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


class ContrastLookupTransform(Transform):
    """
    Segment intensities into equidistant bins and change their mean value.
    """

    def __init__(self, nk=16, keep_background=True, shared=False):
        """

        Parameters
        ----------
        nk : int
            Number of classes
        keep_background : bool
            Do not change background mean/cov.
            The background class is the class with minimum mean value.
        """
        super().__init__(shared=shared)
        self.keep_background = keep_background
        self.nk = nk

    def get_parameters(self, x):
        vmin, vmax = x.min(), x.max()
        edges = torch.linspace(vmin, vmax, self.nk+1)
        new_mu = torch.rand(self.nk) * (vmax - vmin) + vmin
        return edges, new_mu

    def apply_transform(self, x, parameters):
        edges, new_mu = parameters
        old_mu = (edges[:-1] + edges[1:]) / 2

        new_x = x.clone()
        for k in range(self.nk):
            mask = (edges[k] <= x) & (x < edges[k+1])
            new_x[mask] += new_mu[k] - old_mu[k]
        return new_x




