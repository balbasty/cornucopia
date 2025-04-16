
from ..utils import smart_math as math

LOG2 = math.log(2)
FWHM_FACTOR = (8 * LOG2) ** 0.5  # gaussian: fwhm = FWHM_FACTOR * sigma


_PARAMETERIZATIONS = {}


def _register_parameterization(names):
    if isinstance(names, str):
        names = [names]

    def wrapper(func):
        for name in names:
            _PARAMETERIZATIONS[name] = func
        return func

    return wrapper


def _get_prm(*names, **kwargs):
    value = None
    for name in names:
        value = kwargs.get(name, None)
        if value is not None:
            break
    return value


def distribution_parameters(name: str, **kwargs) -> dict:
    """
    Compute the natural parameters of a distribution from any parameterization.

    Defaults depend in the distribution,.

    Parameters
    ----------
    name : {"uniform", "gaussian", "lognormal", "gamma", "generalized"}
        Distribution name.

    Parameters common to most distribution
    --------------------------------------
    mean : float | tensor
        Mean.
    std : float | tensor
        Standard deviation.
    peak : float | tensor
        Location of the mode.
    fwhm : float | tensor
        Full width at half-maximum.

    Parameters of the Uniform distribution
    --------------------------------------
    vmin, a : float | tensor
        Lower bound.
    vmax, a : float | tensor
        Upper bound.

    Parameters of the Gaussian distribution
    ---------------------------------------
    mu : float | tensor
        Alias for `mean`.
    sigma : float | tensor
        Alias for `std`.

    Parameters of the Gamma distribution
    ------------------------------------
    alpha : float | tensor
        Shape parameter.
    beta : float | tensor
        Rate parameter.

    Parameters of the Log-Normal distribution
    -----------------------------------------
    mu : float | tensor
        Mean of the log of the data.
    sigma : float | tensor
        Standard-deviation of the log of the data.

    Parameters of the Generalized Normal distribution
    -------------------------------------------------
    mu : float | tensor
        Alias for `mean`.
    alpha : float | tensor
        Scale parameter.
    beta : float | tensor
        Shape parameter.

    Returns
    -------
    dict

    """
    func = _PARAMETERIZATIONS[name.lower()]
    return func(**kwargs)


@_register_parameterization("uniform")
def uniform_parameters(**kwargs) -> dict:
    """
    Compute the parameters of a uniform distribution from any
    parameterization.

    Two parameterizations can be used:

    * `(vmin, vmax)` is the distribution's natural parameterization,
      where `vmin` is the lower bound and `vmax` is the upper bound.
    * `(mean, std)` is a moment-based parameterization, where
      `mean` is the mean (or center) of the distribution and `std` its
      standard deviation. Alternatively, the width of the distribution
      `fwhm = sqrt(12) * std` can be used in place of `std`.

    By default, the `(vmin, vmax)` parameterization is used. To use,
    the other one, `mean` and `std` (or `fwhm`) must be explicity set as
    keyword arguments, and neither `vmin` nor `vmax` must be used.

    Note that we also accept `peak` as an alias for `mean`. The peak
    is ill defined for the uniform distibution, but can be defined as
    matching the mean by interpreting the uniform distribution as the
    limit of the generalized Gaussian distribution when the shape
    parameter goes to infinity.

    Parameters
    ----------
    vmin, a : float | tensor, default=0
        Lower bound.
    vmax, b : float | tensor, default=1
        Upper bound.
    mean, mu : float | tensor
        Mean: `mean = (a + b) / 2`
    std, sigma : float | tensor
        Standard deviation: `std = (b - a) / sqrt(12)`
    fwhm : float | tensor
        Width: `fwhm = (b - a)`

    Returns
    -------
    dict
        with keys {"a", "b", "vmin", "vmax", "mean", "std", "fwhm"}

    """
    vmin = _get_prm("a", "vmin", **kwargs)
    vmax = _get_prm("b", "vmax", **kwargs)
    mean = _get_prm("mean", "mu", **kwargs)
    std = _get_prm("std", "sigma", **kwargs)
    fwhm = _get_prm("fwhm", **kwargs)

    if (mean is not None) or (std is not None) or (fwhm is not None):
        if ((mean is None) or (std is None and fwhm is None)):
            raise ValueError(
                "(mean, std) must either both be used, or neither be used"
            )
        if (vmin is not None) or (vmax is not None):
            raise ValueError(
                "Cannot mix (mean, std) and (vmin, vmax) parameters"
            )
        if fwhm is None:
            fwhm = (12**0.5) * std
        else:
            std = fwhm / (12**0.5)
        (vmin, vmax) = (mean - fwhm / 2, mean + fwhm / 2)
    else:
        vmin = 0 if vmin is None else vmin
        vmax = 1 if vmax is None else vmax
        mean = (vmin + vmax) / 2
        fwhm = (vmax - vmin)
        std = fwhm / (12**0.5)

    return dict(
        vmin=vmin,
        vmax=vmax,
        a=vmin,
        b=vmax,
        mean=mean,
        mu=mean,
        peak=mean,
        std=std,
        sigma=std,
        fwhm=fwhm,
    )


@_register_parameterization(["gaussian", "normal"])
def gaussian_parameters(**kwargs) -> dict:
    """
    Compute the parameters of a Gaussian distribution from any
    parameterization.

    Parameters
    ----------
    mean, mu, peak : float | tensor, default=0
        Mean.
    std, sigma : float | tensor, default=1
        Standard deviation.
    fwhm : float | tensor
        Full-width at half maximum: `fwhm = sqrt(8 * log(2)) * std`

    Returns
    -------
    dict
        with keys {"mu", "sigma", "mean", "std", "peak", "fwhm"}

    """
    mean = _get_prm("mean", "mu", "peak", **kwargs)
    std = _get_prm("std", "sigma", **kwargs)
    fwhm = _get_prm("fwhm", **kwargs)

    mean = 0 if mean is None else mean
    std = 1 if std is None else std

    if fwhm is None:
        fwhm = FWHM_FACTOR * std
    else:
        std = fwhm / FWHM_FACTOR

    return dict(
        mean=mean,
        mu=mean,
        peak=mean,
        std=std,
        sigma=std,
        fwhm=fwhm,
    )


@_register_parameterization(["lognormal", "log-normal"])
def lognormal_parameters(**kwargs) -> dict:
    """
    Compute the parameters of a log-normal distribution from any
    parameterization.

    Three parameterizations can be used:

    * `(mu, sigma)` is the distribution's natural parameterization,
      where `mu` is the mean of the log of the data and `sigma` is
      the standard deviation of the log of the data.
    * `(mean, std)` is a moment-based parameterization, where
      `mean` is the mean of the distribution and `std` its
      standard deviation.
    * `(peak, fwhm)` is a shape-based parameterization, where
      `peak` is the location of the mode of the distribution,
      and `fwhm` is the full-width at half-maximum of the distribution.

    By default, the `(mean, std)` parameterization is used. To use,
    the other one, `mu` and `sigma` (or `peak` and `fwhm`) must be
    explicity set as keyword arguments, and neither `mean` nor `std`
    must be used.

    Parameters
    ----------
    mean : float  tensor, default=1
        Mean.
    std : float | tensor, default=1
        Standard deviation.
    peak : float | tensor
        Mode.
    fwhm : float | tensor
        Full-width at half maximum.
    mu : float | tensor
        Mean of the log.
    sigma : float | tensor
        Standard deviation of the log.

    Returns
    -------
    dict
        with keys {"mu", "sigma", "mean", "std", "peak", "fwhm"}

    """
    # NOTE
    # FWHM of lognormal taken here:
    # http://openafox.com/science/peak-function-derivations.html#lognormal

    mean = _get_prm("mean", **kwargs)
    std = _get_prm("std", **kwargs)
    fwhm = _get_prm("fwhm", **kwargs)
    peak = _get_prm("peak", **kwargs)
    mu = _get_prm("mu", **kwargs)
    sigma = _get_prm("sigma", **kwargs)

    if (mu is not None) or (sigma is not None):
        if ((mu is None) or (sigma is None)):
            raise ValueError(
                "(mu, sigma) must either both be used, or neither be used"
            )
        if (mean is not None) or (std is not None):
            raise ValueError(
                "(mean, std) cannot be set if (mu, sigma) is set."
            )
        if (peak is None) or (fwhm is not None):
            raise ValueError(
                "(peak, fwhm) cannot be set if (mu, sigma) is set."
            )
        sigma2 = sigma * sigma
        mean = math.exp(mu + 0.5 * sigma2)
        peak = math.exp(mu - sigma2)
        std = mean * math.sqrt(math.exp(sigma2) - 1)
        fwhm = math.exp(sigma * FWHM_FACTOR / 2)
        fwhm = peak * (fwhm - 1/fwhm)
    elif (peak is not None) or (fwhm is not None):
        if ((peak is None) or (fwhm is None)):
            raise ValueError(
                "(peak, fwhm) must either both be used, or neither be used"
            )
        if (mean is not None) or (std is not None):
            raise ValueError(
                "(mean, std) cannot be set if (mu, sigma) is set."
            )
        # fwhm/peak = tmp - 1/tmp
        # tmp = exp(sigma * FWHM_FACTOR / 2)
        # => fp    = fwhm / peak
        # => tmp   = 0.5 * (fp + (fp**2 + 4) ** 0.5)
        # => sigma = 2 * log(tmp) / FWHM_FACTOR
        sigma = fwhm / peak
        sigma = 0.5 * (sigma + (sigma * sigma + 4) ** 0.5)
        sigma = 2 * math.log(sigma) / FWHM_FACTOR
        mu = math.log(peak) + sigma * sigma
        mean = math.exp(mu + 0.5 * sigma2)
        std = mean * math.sqrt(math.exp(sigma2) - 1)
    else:
        mean = 1 if mean is None else mean
        std = 1 if std is None else std
        # mean = math.exp(mu + 0.5 * sigma2)
        # std = mean * math.sqrt(math.exp(sigma2) - 1)
        # => std/mean = math.sqrt(math.exp(sigma2) - 1)
        # => sigma2 = log((std/mean)**2 + 1)
        # => mu = log(mean) - 0.5 * sigma2
        sigma2 = math.log(math.square(std/mean) + 1)
        sigma = math.sqrt(sigma2)
        mu = math.log(mean) - 0.5 * sigma2
        peak = math.exp(mu - sigma2)
        fwhm = math.exp(sigma * FWHM_FACTOR / 2)
        fwhm = peak * (fwhm - 1/fwhm)

    return dict(
        mean=mean,
        std=std,
        peak=peak,
        fwhm=fwhm,
        mu=mu,
        sigma=sigma,
    )


@_register_parameterization(["gamma"])
def gamma_parameters(**kwargs) -> dict:
    """
    Compute the parameters of a Gamma distribution from any
    parameterization.

    Three parameterizations can be used:

    * `(alpha, beta)` is the distribution's natural parameterization,
      where `alpha` is the shape parameter and `beta` is the rate
      parameter.
    * `(mean, std)` is a moment-based parameterization, where
      `mean` is the mean of the distribution and `std` its
      standard deviation.
    * `(peak, fwhm)` is a shape-based parameterization, where
      `peak` is the location of the mode of the distribution,
      and `fwhm` is the full-width at half-maximum of the distribution.
      Note that the Gamma distribution does not have a maximum when
      `alpha < 1`, and therefore no FWHM as well. When `alpha > 1`,
      the FWHM does not have a nicely tracktable form, so we use the
      Laplace approximation instead (i.e., the FWHM of the best
      approximating Gaussian at its peak).

    By default, the `(mean, std)` parameterization is used. To use,
    the other one, `alpha` and `beta` must be explicity set as
    keyword arguments, and neither `mean` nor `std` must be used.

    Parameters
    ----------
    mean, mu : float  tensor, default=1
        Mean.
    std, sigma : float | tensor, default=1
        Standard deviation.
    peak : float | tensor
        Mode.
    fwhm : float | tensor
        Full-width at half maximum.
    alpha : float | tensor
        Shape parameter.
    beta : float | tensor
        Rate parameter.

    Returns
    -------
    dict
        with keys {"alpha", "beta", "mean", "std", "peak", "fwhm"}

    """
    mean = _get_prm("mu", "mean", **kwargs)
    std = _get_prm("sigma", "std", **kwargs)
    alpha = _get_prm("alpha", **kwargs)
    beta = _get_prm("beta", **kwargs)
    peak = _get_prm("peak", **kwargs)
    fwhm = _get_prm("fwhm", **kwargs)

    if (alpha is not None) or (beta is not None):
        if ((alpha is None) or (beta is None)):
            raise ValueError(
                "(alpha, beta) must either both be used, or neither be used"
            )
        if (mean is not None) or (std is not None):
            raise ValueError(
                "(mean, std) cannot be set if (alpha, beta) is set."
            )
        if (peak is not None) or (fwhm is not None):
            raise ValueError(
                "(peak, fwhm) cannot be set if (alpha, beta) is set."
            )
        mean = alpha / beta
        std = alpha**0.5 / beta
        peak = (math.max(alpha, 1) - 1) / beta
        laplace_sigma2 = (math.max(alpha, 1) - 1) / beta**2
        laplace_sigma = laplace_sigma2 ** 0.5
        fwhm = laplace_sigma * FWHM_FACTOR
    elif (peak is not None) or (fwhm is not None):
        if (mean is not None) or (std is not None):
            raise ValueError(
                "(peak, fwhm) cannot be set if (alpha, beta) is set."
            )
        # peak   = (alpha - 1) / beta
        # sigma2 = (alpha - 1) / beta**2
        laplace_sigma = fwhm / FWHM_FACTOR
        laplace_sigma2 = laplace_sigma * laplace_sigma
        beta = peak / laplace_sigma2
        alpha = 1 + peak * beta
        mean = alpha / beta
        std = alpha**0.5 / beta
    else:
        mean = 1 if mean is None else mean
        std = 1 if std is None else std
        var = std * std
        beta = mean / var
        alpha = mean * beta
        laplace_sigma2 = (math.max(alpha, 1) - 1) / beta**2
        laplace_sigma = laplace_sigma2 ** 0.5
        fwhm = laplace_sigma * FWHM_FACTOR

    return dict(
        mean=mean,
        std=std,
        mu=mean,
        sigma=std,
        peak=peak,
        fwhm=fwhm,
        alpha=alpha,
        beta=beta,
    )


@_register_parameterization(["generalized", "generalised"])
def generalized_normal_parameters(**kwargs) -> dict:
    """
    Compute the parameters of a generalized Gaussian distribution from
    any parameterization.

    Three parameterizations can be used:

    * `(mu, alpha)` is the distribution's natural parameterization,
      where `mu` is the mean and `alpha` is the scale parameter.
    * `(mean, std)` is a moment-based parameterization, where
      `mean` is the mean of the distribution and `std` its
      standard deviation.
    * `(peak, fwhm)` is a shape-based parameterization, where
      `peak` is the location of the mode of the distribution,
      and `fwhm` is the full-width at half-maximum of the distribution.
      Note that the Gamma distribution does not have a maximum when
      `alpha < 1`, and therefore no FWHM as well. When `alpha > 1`,
      the FWHM does not have a nicely tracktable form, so we use the
      Laplace approximation instead (i.e., the FWHM of the best
      approximating Gaussian at its peak).

    In Generalized Normal distributions, the mean and peak all equal `mu`.
    Furthermore, the distribution is parameterized by a shape parameter
    `beta`, with the following special cases:

    * `beta = 0`:   Dirac[mu]
    * `beta = 1`:   Laplace[mu, b=alpha]
    * `beta = 2`:   Normal[mu, sigma=alpha/sqrt(2)]
    * `beta = inf`: Uniform[a=mu-alpha, b=mu+alpha]

    Parameters
    ----------
    beta : float | tensor, default=2
        Shape parameter (1 -> Laplace, 2 -> Gaussian).
    alpha : float | tensor, default=1
        Scale parameter (Laplace -> b, Gaussian -> sigma/sqrt(2))
    mean, mu, peak : float | tensor, default=0
        Mean.
    std : float | tensor
        Standard deviation.
    fwhm : float | tensor
        Full-width at half maximum.

    Returns
    -------
    dict
        with keys {"mu", "sigma", "mean", "std", "peak", "fwhm"}

    """
    mean = _get_prm("mu", "mean", "peak", **kwargs)
    beta = _get_prm("beta", **kwargs)
    alpha = _get_prm("alpha", **kwargs)
    std = _get_prm("std", **kwargs)
    fwhm = _get_prm("fwhm", **kwargs)

    mean = 0 if mean is None else mean
    beta = 2 if beta is None else beta

    if sum([(alpha is not None), (std is not None), (fwhm is not None)]) > 1:
        raise ValueError("Only one of `{alpha, std, fwhm}` should be used.")

    if sum([(alpha is not None), (std is not None), (fwhm is not None)]) == 0:
        alpha = 1

    stdfac = math.exp(0.5*(math.gammaln(3/beta) - math.gammaln(1/beta)))

    if fwhm is not None:
        alpha = fwhm / (2 * (LOG2 ** (1/beta)))
        std = alpha * stdfac
    elif std is not None:
        alpha = std / stdfac
        fwhm = 2 * alpha * (LOG2 ** (1/beta))
    else:
        alpha = 1 if alpha is None else alpha
        std = alpha * stdfac
        fwhm = 2 * alpha * (LOG2 ** (1/beta))

    return dict(
        beta=beta,
        mean=mean,
        peak=mean,
        mu=mean,
        alpha=alpha,
        std=std,
        fwhm=fwhm,
    )
