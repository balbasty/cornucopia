import pytest
import torch
import cornucopia as cc
from cornucopia.baseutils import prepare_output, Args, Kwargs, ArgsAndKwargs


class FakeFinal(cc.FinalTransform):

    def _xform(self, x):
        a = x + 1
        b = x + 2
        return prepare_output({
            "input": x, "output": a, "a": a, "b": b
        }, self.returns)


class FakeNonFinal(cc.NonFinalTransform):

    def _unroll(self, x, max_depth=float('inf')):
        return FakeFinal(**self.get_prm()).unroll(max_depth-1)


class FakeSequence(cc.SequentialTransform):

    def __init__(self, **kwargs):
        super().__init__([
            cc.IdentityTransform(),
            FakeFinal(),
            cc.IdentityTransform()
        ], **kwargs)


class FakeMaybe(cc.MaybeTransform):

    def __init__(self, **kwargs):
        super().__init__(FakeFinal(), **kwargs)


class FakeSwitch(cc.SwitchTransform):

    def __init__(self, **kwargs):
        super().__init__([FakeFinal(), cc.IdentityTransform()], **kwargs)


class FakeMappedArgs(cc.MappedTransform):

    def __init__(self, **kwargs):
        super().__init__(mapargs=[FakeFinal(), FakeFinal()])


class FakeMappedKwargs(cc.MappedTransform):

    def __init__(self, **kwargs):
        super().__init__(mapkwargs={"A": FakeFinal(), "B": FakeFinal()})


KlassesReturns = (FakeFinal, FakeNonFinal)
KlassesFinal = (*KlassesReturns, FakeSequence)
Klasses = (*KlassesFinal, FakeMaybe, FakeSwitch)


@pytest.mark.parametrize("Klass", Klasses)
def test_single_arg(Klass):
    x = torch.full([2, 8, 8], 2)
    y = Klass()(x)
    assert torch.is_tensor(y)
    if Klass in KlassesFinal:
        assert (y == 3).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_single_arg_returns(Klass):
    x = torch.full([2, 8, 8], 2)
    y = Klass(returns="input")(x)
    assert (y == 2).all()
    y = Klass(returns="output")(x)
    assert (y == 3).all()
    y = Klass(returns="a")(x)
    assert (y == 3).all()
    y = Klass(returns="b")(x)
    assert (y == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_args(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    y2, y3 = Klass()(x2, x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_args_returns(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    y2, y3 = Klass(returns="input")(x2, x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 2).all()
        assert (y3 == 3).all()
    y2, y3 = Klass(returns="output")(x2, x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 4).all()
    y2, y3 = Klass(returns="a")(x2, x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 4).all()
    y2, y3 = Klass(returns="b")(x2, x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 4).all()
        assert (y3 == 5).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_kwargs(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    y2, y3 = Klass()(A=x2, B=x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 4).all()
    y = Klass()(A=x2, B=x3)
    assert isinstance(y, Kwargs)
    if Klass in KlassesFinal:
        assert (y["A"] == 3).all()
        assert (y["B"] == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_args_kwargs(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    y2, y3 = Klass()(x2, B=x3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 4).all()
    y = Klass()(x2, B=x3)
    assert isinstance(y, ArgsAndKwargs)
    if Klass in KlassesFinal:
        assert (y[0] == 3).all()
        assert (y["B"] == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_arg_list(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    out = Klass()((x2, x3))
    assert isinstance(out, tuple)
    out = Klass()([x2, x3])
    assert isinstance(out, list)
    if Klass in KlassesFinal:
        assert (out[0] == 3).all()
        assert (out[1] == 4).all()
    y2, y3 = Klass()([x2, x3])
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_arg_dict(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    out = Klass()({"A": x2, "B": x3})
    assert isinstance(out, dict)
    if Klass in KlassesFinal:
        assert (out["A"] == 3).all()
        assert (out["B"] == 4).all()
    y2, y3 = Klass()({"A": x2, "B": x3})
    assert y2 == "A"
    assert y3 == "B"


@pytest.mark.parametrize("Klass", Klasses)
def test_kwargs_include(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    y2, y3 = Klass(include="A")(A=x2, B=x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 3).all()
    y2, y3 = Klass(include="B")(A=x2, B=x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 2).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_kwargs_exclude(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    y2, y3 = Klass(exclude="B")(A=x2, B=x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
        assert (y3 == 3).all()
    y2, y3 = Klass(exclude="A")(A=x2, B=x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y2 == 2).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_kwargs_consume(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    y2, = Klass(consume="B")(A=x2, B=x3)
    assert torch.is_tensor(y2)
    if Klass in KlassesFinal:
        assert (y2 == 3).all()
    y3, = Klass(consume="A")(A=x2, B=x3)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_dict_include(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    out = Klass(include="A")({"A": x2, "B": x3})
    assert isinstance(out, dict)
    if Klass in KlassesFinal:
        assert (out["A"] == 3).all()
        assert (out["B"] == 3).all()
    out = Klass(include="B")({"A": x2, "B": x3})
    assert isinstance(out, dict)
    if Klass in KlassesFinal:
        assert (out["A"] == 2).all()
        assert (out["B"] == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_dict_exclude(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    out = Klass(exclude="B")({"A": x2, "B": x3})
    assert isinstance(out, dict)
    if Klass in KlassesFinal:
        assert (out["A"] == 3).all()
        assert (out["B"] == 3).all()
    out = Klass(exclude="A")({"A": x2, "B": x3})
    assert isinstance(out, dict)
    if Klass in KlassesFinal:
        assert (out["A"] == 2).all()
        assert (out["B"] == 4).all()


@pytest.mark.parametrize("Klass", Klasses)
def test_dict_consume(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    out = Klass(consume="B")({"A": x2, "B": x3})
    assert isinstance(out, dict)
    if Klass in KlassesFinal:
        assert (out["A"] == 3).all()
    assert "B" not in out
    out = Klass(consume="A")({"A": x2, "B": x3})
    assert isinstance(out, dict)
    if Klass in KlassesFinal:
        assert (out["B"] == 4).all()
    assert "A" not in out


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_args_noappend(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    trf = Klass(returns=['input', 'output'], append=False)
    (x2, y2), (x3, y3) = trf(x2, x3)
    assert torch.is_tensor(x2)
    assert torch.is_tensor(x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (x2 == 2).all()
        assert (x3 == 3).all()
        assert (y2 == 3).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_args_append(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    trf = Klass(returns=['input', 'output'], append=True)
    x2, y2, x3, y3 = trf(x2, x3)
    assert torch.is_tensor(x2)
    assert torch.is_tensor(x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (x2 == 2).all()
        assert (x3 == 3).all()
        assert (y2 == 3).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_list_noappend(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    trf = Klass(returns=['input', 'output'], append=False)
    (x2, y2), (x3, y3) = trf([x2, x3])
    assert torch.is_tensor(x2)
    assert torch.is_tensor(x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (x2 == 2).all()
        assert (x3 == 3).all()
        assert (y2 == 3).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_list_append(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    trf = Klass(returns=['input', 'output'], append=True)
    x2, y2, x3, y3 = trf([x2, x3])
    assert torch.is_tensor(x2)
    assert torch.is_tensor(x3)
    assert torch.is_tensor(y2)
    assert torch.is_tensor(y3)
    if Klass in KlassesFinal:
        assert (x2 == 2).all()
        assert (x3 == 3).all()
        assert (y2 == 3).all()
        assert (y3 == 4).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_kwargs_noappend(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    trf = Klass(returns={'X': 'input', 'Y': 'output'}, append=False)
    A, B = trf(A=x2, B=x3)
    assert isinstance(A, dict)
    assert isinstance(B, dict)
    assert tuple(A.keys()) == ("X", "Y")
    assert tuple(B.keys()) == ("X", "Y")
    if Klass in KlassesFinal:
        assert (A["X"] == 2).all()
        assert (B["X"] == 3).all()
        assert (A["Y"] == 3).all()
        assert (B["Y"] == 4).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_kwargs_append(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    trf = Klass(returns={'X': 'input', 'Y': 'output'}, append=True, prefix=True)
    out = trf(A=x2, B=x3)
    assert isinstance(out, Kwargs)
    assert tuple(out.keys()) == ("A.X", "A.Y", "B.X", "B.Y")
    if Klass in KlassesFinal:
        assert (out["A.X"] == 2).all()
        assert (out["B.X"] == 3).all()
        assert (out["A.Y"] == 3).all()
        assert (out["B.Y"] == 4).all()


@pytest.mark.parametrize("Klass", KlassesReturns)
def test_kwargs_append_sep(Klass):
    x2 = torch.full([2, 8, 8], 2)
    x3 = torch.full([2, 8, 8], 3)
    trf = Klass(returns={'X': 'input', 'Y': 'output'}, append="/", prefix=True)
    out = trf(A=x2, B=x3)
    assert isinstance(out, Kwargs)
    assert tuple(out.keys()) == ("A/X", "A/Y", "B/X", "B/Y")
    if Klass in KlassesFinal:
        assert (out["A/X"] == 2).all()
        assert (out["B/X"] == 3).all()
        assert (out["A/Y"] == 3).all()
        assert (out["B/Y"] == 4).all()
