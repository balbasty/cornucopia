import pytest
import random
import torch
from cornucopia import (
    RandomSusceptibilityMixtureTransform,
    SusceptibilityToFieldmapTransform,
    ShimTransform,
    OptimalShimTransform,
    RandomShimTransform,
    GradientEchoTransform,
    RandomGMMGradientEchoTransform,
    SmoothLabelMap,
)

torch.autograd.set_detect_anomaly(True)

SEED = 12345678
SIZE = (2, 32, 32)
SIZE3D = (2, 32, 32, 32)


@pytest.mark.parametrize("shared", [True, False])
def test_backward_qmri_susceptibility(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(SIZE)
    x = SmoothLabelMap(nb_classes=3)(x)
    mt = torch.full([], 9.5, requires_grad=True)
    st = torch.full([], 0.01, requires_grad=True)
    mb = torch.full([], 12.5, requires_grad=True)
    sb = torch.full([], 0.1, requires_grad=True)
    f = torch.full([], 1.0, requires_grad=True)
    y = RandomSusceptibilityMixtureTransform(
        mu_tissue=mt,
        sigma_tissue=st,
        mu_bone=mb,
        sigma_bone=sb,
        fwhm=f,
        label_air=1,
        label_bone=2,
        shared=shared
    )(x)
    y.sum().backward()
    assert (
        (mt.grad is not None) and
        (st.grad is not None) and
        (mb.grad is not None) and
        (sb.grad is not None) and
        (f.grad is not None)
    ), [
        k for k, v in {'mt': mt, 'st': st, 'mb': mb, 'sb': sb, 'f': f}.items()
        if v.grad is None
    ]


def test_backward_qmri_tofieldmap():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    b = torch.full([], 3.0,         requires_grad=True)
    f = torch.full([], 42.576E6,    requires_grad=True)
    s0 = torch.full([], 0.4,        requires_grad=True)
    s1 = torch.full([], 0.4,        requires_grad=True)
    v = torch.full([], 1.0,         requires_grad=True)
    x = SmoothLabelMap(nb_classes=1)(x).bool()
    y = SusceptibilityToFieldmapTransform(
        field_strength=b,
        larmor=f,
        s0=s0,
        s1=s1,
        voxel_size=v,
    )(x)
    y.sum().backward()
    assert (
        (b.grad is not None) and
        (f.grad is not None) and
        (s0.grad is not None) and
        (s1.grad is not None) and
        (v.grad is not None)
    )


@pytest.mark.parametrize("size", [SIZE, SIZE3D])
def test_backward_qmri_shim(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(size)
    mt = torch.full([], 9.5,        requires_grad=True)
    st = torch.full([], 0.01,       requires_grad=True)
    mb = torch.full([], 12.5,       requires_grad=True)
    sb = torch.full([], 0.1,        requires_grad=True)
    fw = torch.full([], 1.0,        requires_grad=True)
    b = torch.full([], 3.0,         requires_grad=True)
    f = torch.full([], 42.576E6,    requires_grad=True)
    s0 = torch.full([], 0.4,        requires_grad=True)
    v = torch.full([], 1.0,         requires_grad=True)
    x = SmoothLabelMap(nb_classes=3)(x)
    x = RandomSusceptibilityMixtureTransform(
        mu_tissue=mt,
        sigma_tissue=st,
        mu_bone=mb,
        sigma_bone=sb,
        fwhm=fw,
        label_air=1,
        label_bone=2,
    )(x)
    x = SusceptibilityToFieldmapTransform(
        field_strength=b,
        larmor=f,
        s0=s0,
        voxel_size=v,
    )(x)
    if x.ndim == 3:
        linear = torch.randn([], requires_grad=True)
        quad = torch.randn([], requires_grad=True)
    else:
        assert x.ndim == 4
        linear = torch.randn([3], requires_grad=True)
        quad = torch.randn([2], requires_grad=True)
    y = ShimTransform(linear, quad)(x)
    y.sum().backward()
    assert (
        (mt.grad is not None) and
        (st.grad is not None) and
        (mb.grad is not None) and
        (sb.grad is not None) and
        (fw.grad is not None) and
        (b.grad is not None) and
        (f.grad is not None) and
        (s0.grad is not None) and
        (v.grad is not None)
    )


def test_backward_qmri_shim_optimal():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(SIZE)
    mt = torch.full([], 9.5,        requires_grad=True)
    st = torch.full([], 0.01,       requires_grad=True)
    mb = torch.full([], 12.5,       requires_grad=True)
    sb = torch.full([], 0.1,        requires_grad=True)
    fw = torch.full([], 1.0,        requires_grad=True)
    b = torch.full([], 3.0,         requires_grad=True)
    f = torch.full([], 42.576E6,    requires_grad=True)
    s0 = torch.full([], 0.4,        requires_grad=True)
    v = torch.full([], 1.0,         requires_grad=True)
    x = SmoothLabelMap(nb_classes=3)(x)
    x = RandomSusceptibilityMixtureTransform(
        mu_tissue=mt,
        sigma_tissue=st,
        mu_bone=mb,
        sigma_bone=sb,
        fwhm=fw,
        label_air=1,
        label_bone=2,
    )(x)
    x = SusceptibilityToFieldmapTransform(
        field_strength=b,
        larmor=f,
        s0=s0,
        voxel_size=v,
    )(x)
    y = OptimalShimTransform()(x)
    y.sum().backward()
    assert (
        (mt.grad is not None) and
        (st.grad is not None) and
        (mb.grad is not None) and
        (sb.grad is not None) and
        (fw.grad is not None) and
        (b.grad is not None) and
        (f.grad is not None) and
        (s0.grad is not None) and
        (v.grad is not None)
    )


@pytest.mark.parametrize("shared", [True, False])
def test_backward_qmri_shim_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(SIZE)
    c = torch.full([], 5.0, requires_grad=True)
    x = SmoothLabelMap(nb_classes=3)(x)
    x = RandomSusceptibilityMixtureTransform()(x)
    x = SusceptibilityToFieldmapTransform()(x)
    y = RandomShimTransform(coefficients=c, shared=shared)(x)
    y.sum().backward()
    assert c.grad is not None


def test_backward_qmri_gre():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    size = [5, *SIZE[1:]]  # channels must be MRI parameters
    x = torch.randn(size).abs_()
    x[-2] /= x[-2].max()
    x[-1] *= 0.05 / x[-1].max()
    x.requires_grad_()
    tr = torch.full([], 25e-3, requires_grad=True)
    te = torch.full([], 7e-3,  requires_grad=True)
    fa = torch.full([], 20.0,  requires_grad=True)
    b1 = torch.full([], 1.0,   requires_grad=True)
    y = GradientEchoTransform(tr=tr, te=te, alpha=fa, b1=b1)(x)
    y.sum().backward()
    assert (
        (x.grad is not None) and
        (tr.grad is not None) and
        (te.grad is not None) and
        (fa.grad is not None) and
        (b1.grad is not None)
    )


def test_backward_qmri_gre_random():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    size = [1, *SIZE[1:]]  # cannot be multi-channel
    x = torch.zeros([]).expand(size)
    tr = torch.full([], 25e-3,  requires_grad=True)
    te = torch.full([], 7e-3,   requires_grad=True)
    fa = torch.full([], 20.0,   requires_grad=True)
    b1 = torch.full([], 1.0,    requires_grad=True)
    pd = torch.full([], 1.0,    requires_grad=True)
    t1 = torch.full([], 10.0,   requires_grad=True)
    t2 = torch.full([], 100.0,  requires_grad=True)
    mt = torch.full([], 0.1,    requires_grad=True)
    s = torch.full([], 0.2,     requires_grad=True)
    f = torch.full([], 2.0,     requires_grad=True)
    x = SmoothLabelMap(nb_classes=5)(x)
    y = RandomGMMGradientEchoTransform(
        tr=tr, te=te, alpha=fa, b1=b1,
        pd=pd, t1=t1, t2=t2, mt=mt,
        sigma=s, fwhm=f,
    )(x)
    y.sum().backward()
    assert (
        (tr.grad is not None) and
        (te.grad is not None) and
        (fa.grad is not None) and
        (b1.grad is not None) and
        (t1.grad is not None) and
        (t2.grad is not None) and
        (pd.grad is not None) and
        (mt.grad is not None) and
        (s.grad is not None) and
        (f.grad is not None)
    )
