import torch


# TODO:
#   [ ] Run on histograms rather than random samples
#       [ ] Function to build histograms (just use torch.histc?)
#       [ ] Adapt the GMM code so that it works with weighted values
#       [ ] Stabilize fit by adding the "binning uncertainty" variance bit


def fit_gmm(x, nk=5, max_iter=10, tol=1e-4, max_n='auto'):
    """Fit a multivariate Gaussian mixture model

    Parameters
    ----------
    x : (nc, *shape) tensor
        Multispectral image
    nk : int
        Number of clusters
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for eraly stopping

    Returns
    -------
    z : (nk, *shape) tensor
        Posterior cluster probabilities
    mu : (nk, nc) tensor
        Cluster means
    sigma : (nk, nc, nc) tensor
        Cluster covariance matrices
    pi : (nk,) tensor
        Cluster proportions

    """

    nc, *shape = x.shape
    x = xfull = x.reshape([nc, -1])

    # --- use random subset ---
    x = x[:, (x != 0).all(0)]
    nv0 = x.shape[-1]
    if max_n == 'auto':
        max_n = nk * 1e4
    if max_n:
        p = min(1, max_n / nv0)
        if p < 1:
            m = torch.rand_like(x[0]) < p
            x = x[:, m]
    nv = x.shape[-1]

    # --- wishart prior ---
    x0, x1, x2 = suffstat(x)
    scale = x2 / x0 - (x1 / x0).square()
    df = nc * 0.1
    wishart = (scale.diag().diag(), df)

    # --- initialize clusters ---
    mn, mx = x.min(-1).values, x.max(-1).values
    mu, sigma = [], []
    for c in range(nc):
        edges = torch.linspace(mn[c], mx[c], nk+1)
        centers = (edges[1:] + edges[:-1]) / 2
        fwhm = (mx[c] - mn[c]) / (nk+1)
        mu.append(centers)
        sigma.append(fwhm/2.355)
    mu = torch.stack(mu, -1)
    sigma = torch.stack(sigma).expand([nk, nc])
    sigma = torch.diag_embed(sigma)
    pi = mu.new_ones([nk]).div_(nk)

    # --- initialize responsibilities ---
    z = (x - mu[:, :, None]).div_(sigma.diagonal(0, -1, -2)[:, :, None])
    z = z.square_().sum(1)
    z += 2*sigma.diagonal(0, -1, -2).sum(-1)[:, None]
    z = z.mul_(-0.5)
    l = torch.logsumexp(z, dim=0).mean()
    z = torch.softmax(z, dim=0)

    for n_iter in range(max_iter):

        # --- compute suff stat ---
        x0, x1, x2 = suffstat(x, z)

        # --- update parameters ---
        mu, sigma, pi = params(x0, x1, x2, wishart)

        # --- compute log likelihood ---
        z = log_likelihood(x, mu, sigma, pi)

        # --- loss and softmax ---
        l0 = l
        l = torch.logsumexp(z, dim=0).mean()
        z = torch.softmax(z, dim=0)

        if (l - l0) < tol:
            break

    # E-step full resolution
    x = xfull
    m = (x == 0).all(0)
    z = log_likelihood(x, mu, sigma, pi)
    z = torch.softmax(z, dim=0)
    z[:, m] = 0

    z = z.reshape([nk, *shape])
    return z, mu, sigma, pi


def suffstat(x, z=None):
    if z is None:
        x0 = x.shape[-1]
        x1 = x.mean(-1)
        x2 = torch.matmul(x, x.T) / x0
    else:
        x0 = z.sum(-1)  # [nk]
        x1 = torch.matmul(z, x.T)  # [nk, nc]
        x2 = torch.matmul(x.T[:, :, None], x.T[:, None, :]).movedim(0, -1)
        x2 = torch.matmul(x2, z.T).movedim(-1, 0)  # [nk, nc, nc]
    return x0, x1, x2


def params(x0, x1, x2, wishart=None):

    x0 = x0.clamp_min(1e-6)

    # --- update means ---
    mu = x1 / x0[:, None]  # [nk, nc]

    # --- update covariances ----
    if wishart is None:
        sigma = x2 / x0[:, None, None]
        sigma -= torch.matmul(mu[:, None, :], mu[:, :, None])  # [nk, nc, nc]
    else:
        scale, df0 = wishart
        sigma = df0 * scale + x2
        sigma -= torch.matmul(x1[:, :, None,], x1[:, None, :]) / x0[:, None, None]
        sigma /= (x0[:, None, None] + df0)

    # --- update proportion ----
    pi = x0 / x0.sum()

    return mu, sigma, pi


def log_likelihood(x, mu, sigma, pi):

    chol = torch.linalg.cholesky(sigma)  # [nk, nc, nc]
    log_det = chol.diagonal(0, -1, -2).log().sum(-1).mul(2)  # [nk]
    chol = chol.inverse()

    z = (x - mu[:, :, None])  # [nk, nc, nv]
    z = z.movedim(-1, 0).unsqueeze(-1)  # [nv, nk, nc, 1]
    z = torch.matmul(chol, z)  # [nv, nk, nc, 1]
    z = z.squeeze(-1).square_().sum(-1)  # [nv, nk]
    z += log_det
    z *= -0.5
    z += pi.log()
    z = z.T  # [nk, nv]

    return z


