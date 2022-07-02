import torch
from .base import Transform
from .utils.gmm import fit_gmm


__all__ = ['ContrastTransform']


class ContrastTransform(Transform):
    """
    Find intensity modes using a GMM and change their means and covariances.

    Notes
    -----
    As always, someone thought about it before us:
    .. https://www.frontiersin.org/articles/10.3389/fnins.2021.708196/full
    """

    def __init__(self, nk=16, keep_background=True, dtype=None, shared=False):
        """

        Parameters
        ----------
        nk : int, Number of classes
        keep_background : bool, Do not change background mean/cov
        """
        super().__init__(shared=shared)
        self.keep_background = keep_background
        self.dtype = dtype
        self.nk = nk

    def get_parameters(self, x):
        z, mu, sigma, _ = fit_gmm(x, self.nk)
        return z, mu, sigma

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

    def transform_with_parameters(self, x, parameters):
        z, mu, sigma = parameters

        # Whiten using fitted parameters
        chol = torch.linalg.cholesky(sigma)
        chol = chol.inverse()
        x = x.movedim(0, -1)
        x = x[..., None, :] - mu                       # [..., nk, nc]
        x = torch.matmul(chol, x[..., :, None])        # [..., nk, nc, 1]

        # Color using new parameters
        mu, sigma = self.make_parameters(mu, sigma)
        chol = torch.linalg.cholesky(sigma)
        x = torch.matmul(chol, x)                       # [..., nk, nc, 1]
        x = x[..., 0] + mu                              # [..., nk, nc]
        x = x.movedim(-1, 0)                            # [..., nc, nk]

        # Weight using posterior
        z = z.movedim(0, -1)
        x = torch.matmul(x[..., None, :], z[..., None]) # [..., nc, 1, 1]
        x = x[..., 0, 0]                                # [..., nc]

        return x
