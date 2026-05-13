"""
Unit tests for cornucopia.random
"""
import math
import random
import pytest
import torch
from cornucopia import random as ccrand

SEED = 12345678


# ===========================================================================
# Fixed
# ===========================================================================

class TestFixed:

    def test_scalar_value(self):
        f = ccrand.Fixed(3.14)
        assert f() == pytest.approx(3.14)

    def test_int_value(self):
        f = ccrand.Fixed(7)
        assert f() == 7

    def test_mean_alias(self):
        f = ccrand.Fixed(mean=5.5)
        assert f() == pytest.approx(5.5)

    def test_median_alias(self):
        f = ccrand.Fixed(median=2.0)
        assert f() == pytest.approx(2.0)

    def test_min_alias(self):
        f = ccrand.Fixed(min=9.0)
        assert f() == pytest.approx(9.0)

    def test_std_zero_rejects_nonzero(self):
        with pytest.raises((ValueError, TypeError)):
            ccrand.Fixed(mean=1.0, std=0.5)

    def test_no_value_raises(self):
        with pytest.raises((ValueError, TypeError)):
            ccrand.Fixed()

    def test_sample_list(self):
        f = ccrand.Fixed(5.0)
        result = f(3)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(v == pytest.approx(5.0) for v in result)

    def test_sample_tensor_shape(self):
        f = ccrand.Fixed(2.5)
        t = f([2, 3])
        assert torch.is_tensor(t)
        assert t.shape == (2, 3)
        assert torch.allclose(t, torch.as_tensor(2.5))

    def test_sample_empty_shape(self):
        f = ccrand.Fixed(1.0)
        t = f([])
        assert torch.is_tensor(t)
        assert t.shape == ()

    def test_tensor_value_scalar_output(self):
        val = torch.tensor(9.0)
        f = ccrand.Fixed(val)
        out = f()
        assert torch.is_tensor(out)

    def test_tensor_value_tensor_output(self):
        val = torch.tensor([1.0, 2.0, 3.0])
        f = ccrand.Fixed(val)
        out = f([])
        assert torch.is_tensor(out)

    def test_theta_value(self):
        f = ccrand.Fixed(4.2)
        assert "value" in f.theta
        assert f.theta.value == pytest.approx(4.2)

    def test_theta_mean_property(self):
        f = ccrand.Fixed(3.0)
        assert f.theta.mean == pytest.approx(3.0)

    def test_theta_median_property(self):
        f = ccrand.Fixed(3.0)
        assert f.theta.median == pytest.approx(3.0)

    def test_theta_min_property(self):
        f = ccrand.Fixed(3.0)
        assert f.theta.min == pytest.approx(3.0)


# ===========================================================================
# Uniform
# ===========================================================================

class TestUniform:

    def test_default_range_scalar(self):
        random.seed(SEED)
        u = ccrand.Uniform()
        for _ in range(50):
            x = u()
            assert 0.0 <= x <= 1.0

    def test_explicit_range_scalar(self):
        random.seed(SEED)
        u = ccrand.Uniform(2.0, 5.0)
        for _ in range(50):
            x = u()
            assert 2.0 <= x <= 5.0

    def test_max_only_arg(self):
        random.seed(SEED)
        u = ccrand.Uniform(3.0)
        for _ in range(50):
            x = u()
            assert 0.0 <= x <= 3.0

    def test_sample_list(self):
        random.seed(SEED)
        u = ccrand.Uniform(0.0, 1.0)
        result = u(5)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(0.0 <= x <= 1.0 for x in result)

    def test_sample_tensor_shape(self):
        torch.random.manual_seed(SEED)
        u = ccrand.Uniform(0.0, 1.0)
        t = u([4, 5])
        assert t.shape == (4, 5)
        assert (t >= 0.0).all()
        assert (t <= 1.0).all()

    def test_mean_fwhm_constructor(self):
        u = ccrand.Uniform(mean=5.0, fwhm=2.0)
        assert u.theta.min == pytest.approx(4.0)
        assert u.theta.max == pytest.approx(6.0)

    def test_mean_std_constructor(self):
        # std of Uniform(a,b) = (b-a)/sqrt(12), so fwhm = b-a = std*sqrt(12)
        # and fwhm = std * sqrt(8*ln2) per the normal-based formula
        std = 1.0
        u = ccrand.Uniform(mean=0.0, std=std)
        expected_fwhm = std * math.sqrt(8 * math.log(2))
        assert u.theta.max - u.theta.min == pytest.approx(expected_fwhm)

    def test_mean_var_constructor(self):
        u = ccrand.Uniform(mean=0.0, var=1.0)
        expected_fwhm = math.sqrt(8 * math.log(2) * 1.0)
        assert u.theta.max - u.theta.min == pytest.approx(expected_fwhm)

    def test_theta_mean(self):
        u = ccrand.Uniform(2.0, 4.0)
        assert u.theta.mean == pytest.approx(3.0)

    def test_theta_std(self):
        u = ccrand.Uniform(0.0, 2.0)
        assert u.theta.std == pytest.approx(2.0 / math.sqrt(12))

    def test_theta_var(self):
        u = ccrand.Uniform(0.0, 2.0)
        assert u.theta.var == pytest.approx(4.0 / 12.0)

    def test_theta_fwhm(self):
        u = ccrand.Uniform(1.0, 5.0)
        assert u.theta.fwhm == pytest.approx(4.0)

    def test_list_parameters(self):
        random.seed(SEED)
        u = ccrand.Uniform([0.0, 10.0], [1.0, 20.0])
        result = u()
        assert isinstance(result, list)
        assert len(result) == 2
        assert 0.0 <= result[0] <= 1.0
        assert 10.0 <= result[1] <= 20.0


# ===========================================================================
# RandInt
# ===========================================================================

class TestRandInt:

    def test_default_range(self):
        random.seed(SEED)
        r = ccrand.RandInt()
        for _ in range(50):
            x = r()
            assert x in (0, 1)
            assert isinstance(x, int)

    def test_explicit_range(self):
        random.seed(SEED)
        r = ccrand.RandInt(1, 6)
        for _ in range(100):
            x = r()
            assert 1 <= x <= 6
            assert isinstance(x, int)

    def test_max_only_arg(self):
        random.seed(SEED)
        r = ccrand.RandInt(4)
        for _ in range(50):
            x = r()
            assert 0 <= x <= 4

    def test_keyword_args(self):
        r = ccrand.RandInt(min=2, max=8)
        assert r.theta.min == 2
        assert r.theta.max == 8

    def test_sample_list(self):
        random.seed(SEED)
        r = ccrand.RandInt(0, 9)
        result = r(5)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(0 <= x <= 9 for x in result)

    def test_sample_tensor_shape(self):
        torch.random.manual_seed(SEED)
        r = ccrand.RandInt(0, 9)
        t = r([3, 3])
        assert t.shape == (3, 3)
        assert (t >= 0).all()
        assert (t <= 9).all()

    def test_unexpected_kwargs_raises(self):
        with pytest.raises(TypeError):
            ccrand.RandInt(min=0, max=5, foo=1)


# ===========================================================================
# RandKFrom
# ===========================================================================

class TestRandKFrom:

    def test_fixed_k_no_replacement(self):
        random.seed(SEED)
        r = ccrand.RandKFrom([10, 20, 30, 40, 50], k=3, replacement=False)
        result = r()
        assert len(result) == 3
        assert all(x in [10, 20, 30, 40, 50] for x in result)

    def test_no_replacement_unique(self):
        random.seed(SEED)
        r = ccrand.RandKFrom(list(range(10)), k=5, replacement=False)
        result = r()
        assert len(result) == len(set(result))  # no duplicates

    def test_with_replacement_runs(self):
        random.seed(SEED)
        r = ccrand.RandKFrom([1, 2, 3], k=5, replacement=True)
        result = r()
        assert len(result) == 5
        assert all(x in [1, 2, 3] for x in result)

    def test_random_k(self):
        random.seed(SEED)
        torch.random.manual_seed(SEED)
        r = ccrand.RandKFrom([1, 2, 3, 4, 5])
        result = r()
        assert 1 <= len(result) <= 5
        assert all(x in [1, 2, 3, 4, 5] for x in result)

    def test_too_many_without_replacement_raises(self):
        with pytest.raises(ValueError):
            ccrand.RandKFrom([1, 2, 3], k=5, replacement=False)

    def test_full_k(self):
        random.seed(SEED)
        r = ccrand.RandKFrom([1, 2, 3], k=3, replacement=False)
        result = r()
        assert sorted(result) == [1, 2, 3]

    def test_k_equals_range_replacement(self):
        random.seed(SEED)
        r = ccrand.RandKFrom([1, 2, 3], k=3, replacement=True)
        result = r()
        assert len(result) == 3


# ===========================================================================
# Normal
# ===========================================================================

class TestNormal:

    def test_default_scalar(self):
        random.seed(SEED)
        n = ccrand.Normal()
        x = n()
        assert isinstance(x, float)

    def test_explicit_params_scalar(self):
        random.seed(SEED)
        # Very small sigma → samples concentrate near mu
        n = ccrand.Normal(mu=2.0, sigma=1e-3)
        x = n()
        assert abs(x - 2.0) < 0.05

    def test_mean_std_constructor(self):
        n = ccrand.Normal(mean=5.0, std=2.0)
        assert n.theta.mu == pytest.approx(5.0)
        assert n.theta.sigma == pytest.approx(2.0)

    def test_mean_var_constructor(self):
        n = ccrand.Normal(mean=0.0, var=4.0)
        assert n.theta.sigma == pytest.approx(2.0)

    def test_mean_fwhm_constructor(self):
        fwhm = 2.0
        n = ccrand.Normal(mean=0.0, fwhm=fwhm)
        expected_sigma = fwhm / math.sqrt(8 * math.log(2))
        assert n.theta.sigma == pytest.approx(expected_sigma)

    def test_sample_list(self):
        random.seed(SEED)
        n = ccrand.Normal(mu=0.0, sigma=1.0)
        result = n(10)
        assert isinstance(result, list)
        assert len(result) == 10
        assert all(isinstance(x, float) for x in result)

    def test_sample_tensor_shape(self):
        torch.random.manual_seed(SEED)
        n = ccrand.Normal()
        t = n([3, 4])
        assert t.shape == (3, 4)

    def test_theta_mean_property(self):
        n = ccrand.Normal(mu=1.0, sigma=2.0)
        assert n.theta.mean == pytest.approx(1.0)

    def test_theta_std_property(self):
        n = ccrand.Normal(mu=0.0, sigma=2.0)
        assert n.theta.std == pytest.approx(2.0)

    def test_theta_var_property(self):
        n = ccrand.Normal(mu=0.0, sigma=3.0)
        assert n.theta.var == pytest.approx(9.0)

    def test_theta_fwhm_property(self):
        n = ccrand.Normal(mu=0.0, sigma=1.0)
        assert n.theta.fwhm == pytest.approx(math.sqrt(8 * math.log(2)))

    def test_list_params(self):
        random.seed(SEED)
        n = ccrand.Normal(mu=[0.0, 10.0], sigma=[1.0, 1.0])
        result = n()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_unexpected_kwarg_raises(self):
        with pytest.raises(TypeError):
            ccrand.Normal(mean=0.0, foo=1.0)


# ===========================================================================
# LogNormal
# ===========================================================================

class TestLogNormal:

    def test_default_positive(self):
        random.seed(SEED)
        ln = ccrand.LogNormal()
        for _ in range(50):
            x = ln()
            assert x > 0.0

    def test_explicit_params_positive(self):
        random.seed(SEED)
        ln = ccrand.LogNormal(mu=0.0, sigma=1.0)
        result = ln(50)
        assert all(x > 0.0 for x in result)

    def test_logmean_logstd_constructor(self):
        ln = ccrand.LogNormal(logmean=1.0, logstd=0.5)
        assert ln.theta.mu == pytest.approx(1.0)
        assert ln.theta.sigma == pytest.approx(0.5)

    def test_logmean_logvar_constructor(self):
        ln = ccrand.LogNormal(logmean=0.0, logvar=4.0)
        assert ln.theta.sigma == pytest.approx(2.0)

    def test_logmean_logfwhm_constructor(self):
        fwhm = 2.0
        ln = ccrand.LogNormal(logmean=0.0, logfwhm=fwhm)
        expected_sigma = fwhm / math.sqrt(8 * math.log(2))
        assert ln.theta.sigma == pytest.approx(expected_sigma)

    def test_sample_list(self):
        random.seed(SEED)
        ln = ccrand.LogNormal()
        result = ln(10)
        assert isinstance(result, list)
        assert len(result) == 10
        assert all(x > 0.0 for x in result)

    def test_sample_tensor_positive(self):
        torch.random.manual_seed(SEED)
        ln = ccrand.LogNormal()
        t = ln([2, 3])
        assert t.shape == (2, 3)
        assert (t > 0).all()

    def test_theta_logmean_property(self):
        ln = ccrand.LogNormal(mu=2.0, sigma=1.0)
        assert ln.theta.logmean == pytest.approx(2.0)

    def test_theta_logstd_property(self):
        ln = ccrand.LogNormal(mu=0.0, sigma=1.5)
        assert ln.theta.logstd == pytest.approx(1.5)

    def test_theta_median_property(self):
        ln = ccrand.LogNormal(mu=1.0, sigma=1.0)
        assert ln.theta.median == pytest.approx(math.exp(1.0))

    def test_theta_min_is_zero(self):
        ln = ccrand.LogNormal()
        assert ln.theta.min == pytest.approx(0.0)

    def test_theta_max_is_inf(self):
        ln = ccrand.LogNormal()
        assert ln.theta.max == float('inf')


# ===========================================================================
# Sampler arithmetic: TransformedSampler (sampler op scalar)
# ===========================================================================

class TestTransformedSamplers:
    """Tests for sampler combined with a fixed scalar value."""

    def test_add_scalar(self):
        f = ccrand.Fixed(3.0)
        g = f + 2.0
        assert g() == pytest.approx(5.0)

    def test_radd_scalar(self):
        f = ccrand.Fixed(3.0)
        g = 2.0 + f
        assert g() == pytest.approx(5.0)

    def test_sub_scalar(self):
        f = ccrand.Fixed(5.0)
        g = f - 2.0
        assert g() == pytest.approx(3.0)

    def test_rsub_scalar(self):
        f = ccrand.Fixed(3.0)
        g = 10.0 - f
        assert g() == pytest.approx(7.0)

    def test_mul_scalar(self):
        f = ccrand.Fixed(3.0)
        g = f * 4.0
        assert g() == pytest.approx(12.0)

    def test_rmul_scalar(self):
        f = ccrand.Fixed(3.0)
        g = 4.0 * f
        assert g() == pytest.approx(12.0)

    def test_truediv_scalar(self):
        f = ccrand.Fixed(8.0)
        g = f / 2.0
        assert g() == pytest.approx(4.0)

    def test_rtruediv_scalar(self):
        f = ccrand.Fixed(4.0)
        g = 12.0 / f
        assert g() == pytest.approx(3.0)

    def test_neg(self):
        f = ccrand.Fixed(5.0)
        g = -f
        assert g() == pytest.approx(-5.0)

    def test_pow_scalar(self):
        f = ccrand.Fixed(3.0)
        g = f ** 2
        assert g() == pytest.approx(9.0)

    def test_rpow_scalar(self):
        f = ccrand.Fixed(3.0)
        g = 2.0 ** f  # 2^3 = 8
        assert g() == pytest.approx(8.0)

    def test_exp_method(self):
        f = ccrand.Fixed(0.0)
        g = f.exp()
        assert g() == pytest.approx(1.0)

    def test_log_method(self):
        f = ccrand.Fixed(math.e)
        g = f.log()
        assert g() == pytest.approx(1.0)

    def test_chained_ops(self):
        # (3.0 + 1.0) * 2.0 = 8.0
        f = ccrand.Fixed(3.0)
        g = (f + 1.0) * 2.0
        assert g() == pytest.approx(8.0)

    def test_add_scalar_list_output(self):
        random.seed(SEED)
        f = ccrand.Uniform(0.0, 1.0)
        g = f + 10.0
        result = g(5)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(10.0 <= x <= 11.0 for x in result)

    def test_mul_scalar_tensor_output(self):
        torch.random.manual_seed(SEED)
        f = ccrand.Uniform(0.0, 1.0)
        g = f * 2.0
        t = g([3])
        assert t.shape == (3,)
        assert (t >= 0.0).all()
        assert (t <= 2.0).all()


# ===========================================================================
# Sampler arithmetic: CombinedSamplers (sampler op sampler)
# ===========================================================================

class TestCombinedSamplers:
    """Tests for two samplers combined together."""

    def test_add_samplers(self):
        a = ccrand.Fixed(3.0)
        b = ccrand.Fixed(4.0)
        g = a + b
        assert g() == pytest.approx(7.0)

    def test_sub_samplers(self):
        a = ccrand.Fixed(7.0)
        b = ccrand.Fixed(3.0)
        g = a - b
        assert g() == pytest.approx(4.0)

    def test_mul_samplers(self):
        a = ccrand.Fixed(3.0)
        b = ccrand.Fixed(4.0)
        g = a * b
        assert g() == pytest.approx(12.0)

    def test_div_samplers(self):
        a = ccrand.Fixed(12.0)
        b = ccrand.Fixed(4.0)
        g = a / b
        assert g() == pytest.approx(3.0)

    def test_add_samplers_list_output(self):
        a = ccrand.Fixed(1.0)
        b = ccrand.Fixed(2.0)
        g = a + b
        result = g(3)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(x == pytest.approx(3.0) for x in result)

    def test_min_of_samplers(self):
        a = ccrand.Fixed(3.0)
        b = ccrand.Fixed(7.0)
        g = ccrand.min(a, b)
        assert g() == pytest.approx(3.0)

    def test_max_of_samplers(self):
        a = ccrand.Fixed(3.0)
        b = ccrand.Fixed(7.0)
        g = ccrand.max(a, b)
        assert g() == pytest.approx(7.0)

    def test_sum_of_samplers(self):
        a = ccrand.Fixed(2.0)
        b = ccrand.Fixed(3.0)
        g = ccrand.sum(a, b)
        assert g() == pytest.approx(5.0)

    def test_exp_sampler(self):
        a = ccrand.Fixed(0.0)
        g = ccrand.exp(a)
        assert g() == pytest.approx(1.0)

    def test_log_sampler(self):
        a = ccrand.Fixed(1.0)
        g = ccrand.log(a)
        assert g() == pytest.approx(0.0)


# ===========================================================================
# Sampler.make factory
# ===========================================================================

class TestSamplerMake:

    def test_make_from_sampler_passthrough(self):
        f = ccrand.Fixed(5.0)
        g = ccrand.Fixed.make(f)
        assert g is f

    def test_make_from_dict(self):
        g = ccrand.Fixed.make({'value': 3.0})
        assert g() == pytest.approx(3.0)

    def test_make_from_tuple(self):
        g = ccrand.Uniform.make((1.0, 5.0))
        assert g.theta.min == pytest.approx(1.0)
        assert g.theta.max == pytest.approx(5.0)

    def test_make_from_scalar(self):
        g = ccrand.Fixed.make(7.0)
        assert g() == pytest.approx(7.0)

    def test_uniform_make_from_sampler(self):
        f = ccrand.Uniform(0.0, 1.0)
        g = ccrand.Uniform.make(f)
        assert g is f

    def test_normal_make_from_dict(self):
        g = ccrand.Normal.make({'mu': 2.0, 'sigma': 0.5})
        assert g.theta.mu == pytest.approx(2.0)
        assert g.theta.sigma == pytest.approx(0.5)


# ===========================================================================
# Parameters class
# ===========================================================================

class TestParameters:

    def test_attribute_access(self):
        f = ccrand.Fixed(3.0)
        assert f.theta.value == pytest.approx(3.0)

    def test_attribute_set(self):
        f = ccrand.Fixed(3.0)
        f.theta.value = 5.0
        assert f.theta.value == pytest.approx(5.0)

    def test_attribute_missing_raises(self):
        f = ccrand.Fixed(3.0)
        with pytest.raises(AttributeError):
            _ = f.theta.nonexistent

    def test_list_params_same_length(self):
        # Passing lists of different lengths should raise or pad
        u = ccrand.Uniform([0.0, 1.0], [2.0, 3.0])
        assert isinstance(u.theta.min, list)
        assert isinstance(u.theta.max, list)
        assert len(u.theta.min) == len(u.theta.max) == 2

    def test_sampler_attribute_access(self):
        # Attributes can also be accessed directly on the sampler
        f = ccrand.Fixed(6.0)
        # Through __getattr__
        assert f.theta.value == pytest.approx(6.0)


# ===========================================================================
# make_range
# ===========================================================================

class TestMakeRange:

    def test_single_arg_symmetric(self):
        lo, hi = ccrand.make_range(3.0)
        assert lo == pytest.approx(-3.0)
        assert hi == pytest.approx(3.0)

    def test_two_args(self):
        lo, hi = ccrand.make_range(1.0, 5.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(5.0)

    def test_with_offset(self):
        lo, hi = ccrand.make_range(2.0, offset=1.0)
        assert lo == pytest.approx(-1.0)
        assert hi == pytest.approx(3.0)

    def test_max_kwarg_only(self):
        lo, hi = ccrand.make_range(max=5.0)
        assert lo == pytest.approx(-5.0)
        assert hi == pytest.approx(5.0)

    def test_min_kwarg_only(self):
        lo, hi = ccrand.make_range(min=2.0)
        # Based on code: return vmid + vmin, vmid - vmin = (2.0, -2.0)
        # The docstring example is not shown for this case, so match code
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(-2.0)

    def test_passthrough_sampler(self):
        f = ccrand.Fixed(1.0)
        result = ccrand.make_range(f)
        assert result is f

    def test_passthrough_list(self):
        lst = [1.0, 5.0]
        result = ccrand.make_range(lst)
        assert result == lst

    def test_passthrough_tuple(self):
        tpl = (1.0, 5.0)
        result = ccrand.make_range(tpl)
        assert result == tpl

    def test_two_args_with_offset(self):
        lo, hi = ccrand.make_range(1.0, 5.0, offset=10.0)
        assert lo == pytest.approx(11.0)
        assert hi == pytest.approx(15.0)
