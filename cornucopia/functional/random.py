__all__ = [
    "random_field_uniform",
    "random_field_gaussian",
    "random_field_lognormal",
    "random_field_gamma",
    "random_field_uniform_like",
    "random_field_gaussian_like",
    "random_field_lognormal_like",
    "random_field_gamma_like",
]
# stdlib
from typing import Sequence, Optional, Callable, Union, Mapping

# external
import torch

# internal
from ..baseutils import prepare_output, returns_update
from ..utils import smart_math as math
from ..utils.distributions import (
    uniform_parameters,
    gaussian_parameters,
    lognormal_parameters,
    gamma_parameters,
    generalized_normal_parameters,
)
from ._utils import _unsqz_spatial, _backend_float


Tensor = torch.Tensor
Value = Union[float, Tensor]
Output = Union[Tensor, Mapping[Tensor], Sequence[Tensor]]

LOG2 = math.log(2)
FWHM_FACTOR = (8 * LOG2) ** 0.5  # gaussian: fwhm = FWHM_FACTOR * sigma


def random_field(name: str, shape: Sequence[int], **kwargs) -> Output:
    """
    Sample a random field from a probability distribution

    Parameters
    ----------
    name : {"uniform", "gaussian", "gamma", "lognormal", "generalized"}
        Distribution name.
    shape : list[int]
        Output shape, including the channel dimension (!!): (C, *spatial).

    Other Parameters
    ----------------
    mean : float | ([C],) tensor
        Mean.
    std : float | ([C],) tensor
        Standard deviation.
    peak : float | ([C],) tensor
        Peak.
    fwhm : float | ([C],) tensor
        Width.
    vmin, vmax, alpha, beta, mu, sigma : float | ([C],) tensor
        Distribution-specific parameters
    returns : [list or dict of] {"output", "mean", "std", "peak", "fwhm", ...}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """  # noqa: E501
    name = name.lower()
    if name == "uniform":
        return random_field_uniform(shape, **kwargs)
    if name in ("normal", "gaussian"):
        return random_field_gaussian(shape, **kwargs)
    if name == "gamma":
        return random_field_gamma(shape, **kwargs)
    if name in ("lognormal", "log-normal"):
        return random_field_lognormal(shape, **kwargs)
    if name in ("generalized", "generalised"):
        return random_field_generalized(shape, **kwargs)


def random_field_uniform(
    shape: Sequence[int],
    vmin: Optional[Value] = None,
    vmax: Optional[Value] = None,
    **kwargs
) -> Output:
    """
    Sample a random field from a uniform distribution

    !!! note "Parameters"
        Two parameterizations can be used:

        * `(vmin, vmax)` is the distribution"s natural parameterization,
            where `vmin` is the lower bound and `vmax` is the upper bound.
        * `(mean, std)` is a moment-based parameterization, where
            `mean` is the mean of the distribution and `std` its
            standard deviation. Alternatively, the width of the distribution
            `fwhm` can be used in place of `std`.

        By default, the `(vmin, vmax)` parameterization is used. To use,
        the other one, `mean` and `std` (or `fwhm`) must be explicity set as
        keyword arguments, and neither `vmin` nor `vmax` must be used.

    Parameters
    ----------
    shape : list[int]
        Output shape, including the channel dimension (!!): (C, *spatial).
    vmin : float | ([C],) tensor, default=0
        Minimum value.
    vmax : float | ([C],) tensor, default=1
        Maximum value.

    Other Parameters
    ----------------
    mean : float | ([C],) tensor
        Mean.
    std : float | ([C],) tensor
        Standard deviation.
    fwhm : float | ([C],) tensor
        Width.
    returns : [list or dict of] {"output", "vmin", "vmax"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """
    prm = uniform_parameters(vmin=vmin, vmax=vmax, **kwargs)
    vmin, vmax = prm["vmin"], prm["vmax"]

    ndim = len(shape) - 1
    vmin_ = _unsqz_spatial(vmin, ndim)
    vmax_ = _unsqz_spatial(vmax, ndim)

    backend = _backend_float(vmin, vmax, **kwargs)
    output = torch.rand(shape, **backend)
    output = math.add_(math.mul_(output, (vmax_ - vmin_)), vmin_)

    kwargs.setdefault("returns", "output")
    return prepare_output({"output": output, **prm}, kwargs["returns"])


def random_field_gaussian(
    shape: Sequence[int],
    mean: Optional[Value] = None,
    std: Optional[Value] = None,
    **kwargs
) -> Output:
    """
    Sample a random field from a Gaussian distribution

    Parameters
    ----------
    shape : list[int]
        Output shape, including the channel dimension (!!): (C, *spatial).
    mean : float | ([C],) tensor, default=0
        Mean.
    std : float | ([C],) tensor, default=1
        Standard deviation.

    Other Parameters
    ----------------
    fwhm : float | ([C],) tensor
        The Full-width at half maximum can be specifed in place of the std.
    returns : [list or dict of] {"output", "mu", "sigma"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """
    prm = gaussian_parameters(mean=mean, std=std, **kwargs)
    mean, std = prm["mean"], prm["std"]

    ndim = len(shape) - 1
    mean_ = _unsqz_spatial(mean, ndim)
    std_ = _unsqz_spatial(std, ndim)

    backend = _backend_float(mean, std, **kwargs)
    output = torch.randn(shape, **backend)
    output = math.add_(math.mul_(output, std_), mean_)

    kwargs.setdefault("returns", "output")
    return prepare_output({"output": output, **prm}, kwargs["returns"])


def random_field_lognormal(
    shape: Sequence[int],
    mean: Optional[Value] = None,
    std: Optional[Value] = None,
    **kwargs
) -> Output:
    """
    Sample a random field from a log-normal distribution

    !!! note "Parameters"
        Three parameterizations can be used:

        * `(mu, sigma)` is the distribution"s natural parameterization,
            where `mu` is the mean of the log of the data and `sigma` is
            the standard deviation of the log of the data.
        * `(mean, std)` is a moment-based parameterization, where
            `mean` is the mean of the distribution and `std` its
            standard deviation. Alternatively, the width of the distribution
            `fwhm` can be used in place of `std`.
        * `(peak, fwhm)` is a shape-based parameterization, where
            `peak` is the location of the mode of the distribution,
            and `fwhm` is the full-width at half-maximum of the distribution.

        By default, the `(mean, std)` parameterization is used. To use,
        the other one, `mu` and `sigma` must be explicity set as
        keyword arguments, and neither `mean` nor `std` must be used.

    Parameters
    ----------
    shape : list[int]
        Output shape, including the channel dimension (!!): (C, *spatial).
    mean : float | ([C],) tensor, default=1
        Mean of the distribution.
        (!! mean(x) != {mu == mean(log(x))}).
    std : float | ([C],) tensor, default=1
        Standard deviation of the distribution.
        (!! std(x) != {sigma == std(log(x))}).

    Other Parameters
    ----------------
    peak : float | ([C],) tensor
        Location of the peak of the distribution.
    fwhm : float | ([C],) tensor
        Standard deviation.
    mu : float | ([C],) tensor
        Mean of the log.
    sigma : float | ([C],) tensor
        Standard deviation of the log.
    returns : [list or dict of] {"output", "mean", "std", "fwhm", "peak", "mu", "sigma"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """  # noqa: E501
    prm = lognormal_parameters(mean=mean, sts=std, **kwargs)
    mu, sigma = prm["mu"], prm["sigma"]

    ndim = len(shape) - 1
    mu_ = _unsqz_spatial(mu, ndim)
    sigma_ = _unsqz_spatial(sigma, ndim)

    backend = _backend_float(mu_, sigma_, **kwargs)
    output = torch.randn(shape, **backend)
    output = math.exp_(math.add_(math.mul_(output, sigma_), mu_))

    kwargs.setdefault("returns", "output")
    return prepare_output({"output": output, **prm}, kwargs["returns"])


def random_field_gamma(
    shape: Sequence[int],
    mean: Optional[Value] = None,
    std: Optional[Value] = None,
    **kwargs
) -> Output:
    """
    Sample a random field from a Gamma distribution

    !!! note "Parameters"
        Two parameterizations can be used:

        * `(alpha, beta)` is the distribution"s natural parameterization,
            where `alpha` is the shape parameter and `beta` is the rate
            parameter.
        * `(mean, std)` is a moment-based parameterization, where
            `mean` is the mean of the distribution and `std` its
            standard deviation.
        * `(peak, fwhm)` is a shape-based parameterization, where
            `peak` is the location of the mode of the distribution,
            and `fwhm` is the full-width at half-maximum of the distribution.
            Since the `fwhm` of the Gamma distribution does not have a
            nicely tracktable form, we use a Laplace approximation, which
            only exists for alpha > 1.

        By default, the `(mean, std)` parameterization is used. To use,
        the other one, `alpha` and `beta` must be explicity set as
        keyword arguments, and neither `mean` nor `std` must be used.

    Parameters
    ----------
    shape : list[int]
        Output shape, including the channel dimension (!!): (C, *spatial).
    mean : float | ([C],) tensor, default=1
        Mean.
    std : float | ([C],) tensor, default=1
        Standard deviation.

    Other Parameters
    ----------------
    alpha : float | ([C],) tensor
        Shape parameter.
    beta : float | ([C],) tensor
        Rate parameter.
    peak : float | ([C],) tensor
        Mode of the distribution.
    fwhm : float | ([C],) tensor
        Full-width at half-maximum.
    returns : [list or dict of] {"output", "mean", "std", "alpha", "beta"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """
    prm = gamma_parameters(mean=mean, std=std, **kwargs)
    alpha, beta = prm["alpha"], prm["beta"]

    backend = _backend_float(alpha, beta)
    alpha_ = torch.as_tensor(alpha, **backend)
    beta_ = torch.as_tensor(beta, **backend)
    alpha_ = alpha_.expand(shape[:1])
    beta_ = beta_.expand(shape[:1])

    output = torch.distributions.Gamma(alpha_, beta_).rsample(shape[1:])

    kwargs.setdefault("returns", "output")
    return prepare_output({"output": output, **prm}, kwargs["returns"])


def random_field_generalized(
    shape: Sequence[int],
    mean: Optional[Value] = None,
    std: Optional[Value] = None,
    beta: Value = 2,
    **kwargs
) -> Output:
    """
    Sample a random field from a Generalized Normal distribution.

    !!! note "Parameters"
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
    shape : list[int]
        Output shape, including the channel dimension (!!): (C, *spatial).
    mean : float | ([C],) tensor, default=0
        Mean.
    std : float | ([C],) tensor, default=1
        Standard deviation.
    beta : float | ([C],) tensor, default=2
        Shape parameter.

    Other Parameters
    ----------------
    peak : float | ([C],) tensor
        The mode of the distribution can be specified in place of the mean.
    fwhm : float | ([C],) tensor
        The Full-width at half maximum can be specifed in place of the std.
    alpha : float | ([C],) tensor
        The scale parameter can be specifed in place of the std.
    returns : [list or dict of] {"output", "mean", "std", "peak", "fwhm", "alpha", "beta"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """  # noqa: E501
    # https://blogs.sas.com/content/iml/2016/09/21/simulate-generalized-gaussian-sas.html

    ndim = len(shape) - 1

    # Default in `generalized_normal_parameters` is `alpha=1`, whereas
    # the default in `random_field_generalized_normal` is `std=1`.
    if kwargs.get("alpha", None) is None and kwargs.get("fwhm", None) is None:
        std = 1 if std is None else std

    kwargs["beta"], kwargs["mean"], kwargs["std"] = beta, mean, std
    prm = generalized_normal_parameters(**kwargs)

    mean, std, alpha, beta = prm["mean"], prm["std"], prm["alpha"], prm["beta"]
    backend = _backend_float(mean, std, alpha, beta, **kwargs)

    mean_ = _unsqz_spatial(mean, ndim)
    std_ = _unsqz_spatial(std, ndim)

    b = math.exp(0.5*(math.gammaln(3/beta) - math.gammaln(1/beta)))
    sign = random_field_uniform(shape, **backend) > 0.5
    output = random_field_gamma(shape, alpha=1/alpha, beta=1/b, **backend)
    output = math.mul_(output, 2 * sign - 1)
    output = math.add_(math.mul_(output, std_), mean_)

    kwargs.setdefault("returns", "output")
    return prepare_output({"output": output, **prm}, kwargs["returns"])


def random_field_like(
    name: str,
    input: Tensor,
    shape: Sequence[int],
    **kwargs
) -> Output:
    """
    Sample a random field from a probability distribution

    Parameters
    ----------
    name : {"uniform", "gaussian", "gamma", "lognormal", "generalized"}
        Distribution name.
    input : tensor
        Tensor from which to copy the data type, device and shape
    shape : list[int]
        Output shape, including the channel dimension (!!): (C, *spatial).

    Other Parameters
    ----------------
    mean : float | ([C],) tensor
        Mean.
    std : float | ([C],) tensor
        Standard deviation.
    peak : float | ([C],) tensor
        Peak.
    fwhm : float | ([C],) tensor
        Width.
    vmin, vmax, alpha, beta, mu, sigma : float | ([C],) tensor
        Distribution-specific parameters
    returns : [list or dict of] {"output", "mean", "std", "peak", "fwhm", ...}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """  # noqa: E501
    name = name.lower()
    if name == "uniform":
        return random_field_uniform_like(input, shape, **kwargs)
    if name in ("normal", "gaussian"):
        return random_field_gaussian_like(input, shape, **kwargs)
    if name == "gamma":
        return random_field_gamma_like(input, shape, **kwargs)
    if name in ("lognormal", "log-normal"):
        return random_field_lognormal_like(input, shape, **kwargs)
    if name in ("generalized", "generalised"):
        return random_field_generalized_like(input, shape, **kwargs)


def _random_field_like(
    func: Callable,
    input: Tensor,
    shape: Optional[Sequence[int]] = None,
    *args,
    **kwargs
) -> Output:
    """
    Helper to sample a random field from a distribution

    Parameters
    ----------
    func : callable
        Sampling function
    input : tensor
        Tensor from which to copy the data type, device and shape
    shape : list[int] | None
        Output shape. Same as input by default.
    *args
        `func`"s other parameters.

    Other Parameters
    ----------------
    returns : [list or dict of] {"input", "output", ...}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """
    kwargs.setdefault("returns", "output")

    # copy shape
    if shape is None:
        shape = input.shape
    shape = torch.Size(shape)
    # if pure spatial shape, copy channels
    if len(shape) == input.ndim - 1:
        shape = input.shape[:1] + shape

    # copy dtype/device
    dtype = kwargs.get("dtype", None) or input.dtype
    device = kwargs.get("device", None) or input.device
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    kwargs["dtype"] = dtype
    kwargs["device"] = device

    # sample field
    output = func(shape, *args, **kwargs)

    return returns_update(input, "input", output, kwargs["returns"])


def random_field_uniform_like(
    input: Tensor,
    shape: Optional[Sequence[int]] = None,
    vmin: Value = 0,
    vmax: Value = 1,
    **kwargs
) -> Output:
    """
    Sample a random field from a uniform distribution

    !!! note "Parameters"
        Two parameterizations can be used:
            * `(vmin, vmax)` is the distribution"s natural parameterization,
              where `vmin` is the lower bound and `vmax` is the upper bound.
            * `(mean, std)` is a moment-based parameterization, where
              `mean` is the mean of the distribution and `std` its
              standard deviation. Alternatively, the width of the distribution
              `fwhm` can be used in place of `std`.

        By default, the `(vmin, vmax)` parameterization is used. To use,
        the other one, `mean` and `std` (or `fwhm`) must be explicity set as
        keyword arguments, and neither `vmin` nor `vmax` must be used.

    Parameters
    ----------
    input : tensor
        Tensor from which to copy the data type, device and shape
    shape : list[int] | None
        Output shape. Same as input by default.
    vmin : float | ([C],) tensor
        Minimum value.
    vmax : float | ([C],) tensor
        Maximum value.

    Other Parameters
    ----------------
    mean : float | ([C],) tensor
        Mean.
    std : float | ([C],) tensor
        Standard deviation.
    fwhm : float | ([C],) tensor
        Width.
    returns : [list or dict of] {"output", "input", "vmin", "vmax", "mean", "std", "fwhm"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """  # noqa: E501
    return _random_field_like(
        random_field_uniform, input, shape, vmin, vmax, **kwargs
    )


def random_field_gaussian_like(
    input: Tensor,
    shape: Optional[Sequence[int]] = None,
    mu: Value = 0,
    sigma: Value = 1,
    **kwargs
) -> Output:
    """
    Sample a random field from a gaussian distribution

    Parameters
    ----------
    input : tensor
        Tensor from which to copy the data type, device and shape
    shape : list[int] | None
        Output shape. Same as input by default.
    mu : float | ([C],) tensor
        Mean.
    sigma : float | ([C],) tensor
        Standard deviation.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "mu", "sigma"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """
    return _random_field_like(
        random_field_gaussian, input, shape, mu, sigma, **kwargs
    )


def random_field_lognormal_like(
    input: Tensor,
    shape: Optional[Sequence[int]] = None,
    mu: Value = 0,
    sigma: Value = 1,
    **kwargs
) -> Output:
    """
    Sample a random field from a log-normal distribution

    Parameters
    ----------
    input : tensor
        Tensor from which to copy the data type, device and shape
    shape : list[int] | None
        Output shape. Same as input by default.
    mu : float | ([C],) tensor
        Mean of log.
    sigma : float | ([C],) tensor
        Standard deviation of log.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "mu", "sigma"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """
    return _random_field_like(
        random_field_lognormal, input, shape, mu, sigma, **kwargs
    )


def random_field_gamma_like(
    input: Tensor,
    shape: Optional[Sequence[int]] = None,
    mean: Optional[Value] = None,
    std: Optional[Value] = None,
    **kwargs
) -> Output:
    """
    Sample a random field from a Gamma distribution

    !!! note "Parameters"
        Two parameterizations can be used:
            * `(alpha, beta)` is the distribution"s natural parameterization,
              where `alpha` is the shape parameter and `beta` is the rate
              parameter.
            * `(mean, std)` is a moment-based parameterization, where
              `mean` is the mean of the distribution and `std` its
              standard deviation.

        By default, the `(mean, std)` parameterization is used. To use,
        the other one, `alpha` and `beta` must be explicity set as
        keyword arguments, and neither `mean` nor `std` must be used.

    Parameters
    ----------
    input : tensor
        Tensor from which to copy the data type, device and shape
    shape : list[int] | None
        Output shape. Same as input by default.
    mean : float | ([C],) tensor
        Mean.
    std : float | ([C],) tensor
        Standard deviation.

    Other Parameters
    ----------------
    alpha : float | ([C],) tensor
        Shape parameter.
    beta : float | ([C],) tensor
        Rate parameter.
    returns : [list or dict of] {"output", "mean", "std", "alpha", "beta"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """
    return _random_field_like(
        random_field_lognormal, input, shape, mean, std, **kwargs
    )


def random_field_generalized_like(
    input: Tensor,
    shape: Optional[Sequence[int]] = None,
    mean: Optional[Value] = None,
    std: Optional[Value] = None,
    beta: Value = 2,
    **kwargs
) -> Output:
    """
    Sample a random field from a Generalized Gaussian distribution

    !!! note "Parameters"
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
    input : tensor
        Tensor from which to copy the data type, device and shape
    shape : list[int] | None
        Output shape. Same as input by default.
    mean : float | ([C],) tensor, default=0
        Mean.
    std : float | ([C],) tensor, default=1
        Standard deviation.
    beta : float | ([C],) tensor, default=2
        Shape parameter.

    Other Parameters
    ----------------
    peak : float | ([C],) tensor
        The mode of the distribution can be specified in place of the mean.
    fwhm : float | ([C],) tensor
        The Full-width at half maximum can be specifed in place of the std.
    alpha : float | ([C],) tensor
        The scale parameter can be specifed in place of the std.
    returns : [list or dict of] {"output", "mean", "std", "peak", "fwhm", "alpha", "beta"}

    Returns
    -------
    output : (*shape) tensor
        Output tensor.
    """  # noqa: E501
    return _random_field_like(
        random_field_generalized, input, shape, mean, std, beta, **kwargs
    )
