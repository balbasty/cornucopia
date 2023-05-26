__all__ = ['ElasticTransform', 'RandomElasticTransform',
           'AffineTransform', 'RandomAffineTransform',
           'AffineElasticTransform', 'RandomAffineElasticTransform',
           'MakeAffinePair',
           'ThroughSliceAffineTransform', 'RandomThroughSliceAffineTransform']

import torch
from torch.nn.functional import interpolate
import math
import random
import interpol
from .base import Transform, RandomizedTransform, prepare_output
from .random import Sampler, Uniform, RandInt, Fixed, make_range
from .utils import warps
from .utils.py import ensure_list


class ElasticTransform(Transform):
    """
    Elastic transform encoded by cubic splines.
    The number of control points is fixed but coefficients are
    randomly sampled.
    """

    def __init__(self, dmax=0.1, unit='fov', shape=5, bound='border',
                 steps=0, order=3, *, shared=True, **kwargs):
        """

        Parameters
        ----------
        dmax : [list of] float
            Max displacement per dimension
        unit : {'fov', 'vox'}
            Unit of `dmax`.
        shape : [list of] int
            Number of spline control points
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        steps : int
            Number of scaling-and-squaring integration steps
        order : int
            Spline order
        returns : [list or dict of] {'input', 'output', 'flow', 'controls'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'controls': The control points of the displacement field
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        if unit not in ('fov', 'vox'):
            raise ValueError('Unit must be one of {"fov", "vox"} '
                             f'but got "{unit}".')
        if bound not in ('zeros', 'border', 'reflection'):
            raise ValueError('Bound must be one of '
                             '{"zeros", "border", reflection"} '
                             f'but got "{bound}".')
        self.dmax = dmax
        self.unit = unit
        self.bound = bound
        self.shape = shape
        self.steps = steps
        self.order = order

    def get_parameters(self, x, fullsize=True):
        """

        Parameters
        ----------
        x : (C, *shape) tensor
        fullsize : bool

        Returns
        -------
        warp : (C, D, *shape) tensor, if `fullsize`
            Dense, exponentiated warp
        control : (C, D, *self.shape)
            Spline control points

        """
        batch, *fullshape = x.shape
        ndim = len(fullshape)
        smallshape = ensure_list(self.shape, ndim)
        dmax = ensure_list(self.dmax, ndim)
        backend = dict(dtype=x.dtype, device=x.device)
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        if self.unit == 'fov':
            dmax = [d * f for d, f in zip(dmax, fullshape)]
        t = torch.rand([batch, ndim, *smallshape], **backend)
        for d in range(ndim):
            t[:, d].sub_(0.5).mul_(2*dmax[d])
        if not fullsize:
            return t
        if self.order == 1:
            mode = ('trilinear' if len(fullshape) == 3 else
                    'bilinear' if len(fullshape) == 2 else
                    'linear')
            ft = interpolate(t[None], fullshape, mode=mode, align_corners=True)[0]
        else:
            ft = interpol.resize(t, shape=fullshape, interpolation=self.order,
                                 prefilter=False)
        if self.steps:
            ft = warps.exp_velocity(ft.movedim(1, -1), self.steps).movedim(-1, 1)
        return ft, t

    def apply_transform(self, x, parameters):
        flow, controls = parameters
        y = warps.apply_flow(x[:, None], flow.movedim(1, -1),
                             padding_mode=self.bound)[:, 0]
        return prepare_output(
            dict(input=x, output=y, flow=flow, controls=controls),
            self.returns)


class RandomElasticTransform(RandomizedTransform):
    """
    Elastic Transform with random parameters.
    """

    def __init__(self, dmax=0.15, shape=10, unit='fov', bound='border',
                 steps=0, order=3, *, shared=True, **kwargs):
        """

        Parameters
        ----------
        dmax : Sampler or float
            Sampler or Upper bound for maximum displacement
        shape : Sampler or int
            Sampler or Upper bound for number of control points
        unit : {'fov', 'vox'}
            Unit of `dmax`
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        order : int
            Spline order
        returns : [list or dict of] {'input', 'output', 'flow', 'controls'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'controls': The control points of the displacement field
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(
            ElasticTransform,
            dict(dmax=Uniform.make(make_range(0, dmax)),
                 shape=RandInt.make(make_range(2, shape)),
                 unit=unit,
                 bound=bound,
                 steps=steps,
                 order=order,
                 **kwargs),
            shared=shared)

    def get_parameters(self, x):
        ndim = x.dim() - 1
        sample = self.sample
        keys_nd = ['shape']
        self.sample = {
            key: value(ndim)
            if isinstance(value, Sampler) and key in keys_nd else value
            for key, value in sample.items()
        }
        sub = super().get_parameters(x)
        self.sample = sample
        return sub


class AffineTransform(Transform):
    """
    Apply an affine transform encoded by translations, rotations,
    shears and zooms.

    The affine matrix is defined as:
        `A = T @ Rx @ Ry @ Rz @ Sx @ Sy Sz @ Z`
    with the center of the field of view used as center of rotation.
    (A is a matrix so the transforms are applied right to left)
    """

    def __init__(self, translations=0, rotations=0, shears=0, zooms=0,
                 unit='fov', bound='border', *, shared=True, **kwargs):
        """

        Parameters
        ----------
        translations : [list of] float
            Translation (per X/Y/Z)
        rotations : [list of] float
            Rotations (about Z/Y/X), in deg
        shears : [list of] float
            Translation (about Z/Y/Z)
        zooms : [list of] float
            Zoom about 1 (per X/Y/Z)
        unit : {'fov', 'vox'}
            Unit of `translations`.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'matrix': The affine matrix
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        if unit not in ('fov', 'vox'):
            raise ValueError('Unit must be one of {"fov", "vox"} '
                             f'but got "{unit}".')
        if bound not in ('zeros', 'border', 'reflection'):
            raise ValueError('Bound must be one of '
                             '{"zeros", "border", reflection"} '
                             f'but got "{bound}".')
        self.translations = translations
        self.rotations = rotations
        self.zooms = zooms
        self.shears = shears
        self.unit = unit
        self.bound = bound

    def get_parameters(self, x, fullsize=True):
        batch, *fullshape = x.shape
        ndim = len(fullshape)
        backend = dict(dtype=x.dtype, device=x.device)
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()

        rotations = ensure_list(self.rotations, ndim * (ndim - 1) // 2)
        shears = ensure_list(self.shears, ndim * (ndim - 1) // 2)
        translations = ensure_list(self.translations, ndim)
        zooms = ensure_list(self.zooms, ndim)
        offsets = [(n-1)/2 for n in fullshape]

        if self.unit == 'fov':
            translations = [t * n for t, n in zip(translations, fullshape)]
        rotations = [r * math.pi / 180 for r in rotations]

        rotations = torch.as_tensor(rotations, **backend)
        shears = torch.as_tensor(shears, **backend)
        translations = torch.as_tensor(translations, **backend)
        zooms = torch.as_tensor(zooms, **backend)
        offsets = torch.as_tensor(offsets, **backend)

        I = torch.eye(ndim+1, **backend)
        O = I.clone()
        O[:ndim, -1] = -offsets
        Z = I.clone()
        Z.diagonal(0, -1, -2)[:-1].copy_(1 + zooms)
        T = torch.eye(ndim+1, **backend)
        T[:ndim, -1] = translations

        A = O               # origin at center of FOV
        A = Z @ A           # zoom
        if ndim == 2:
            S = I.clone()
            S[0, 1] = S[1, 0] = shears[0]
            A = S @ A       # shear
            R = I.clone()
            R[0, 0] = R[1, 1] = rotations[0].cos()
            R[0, 1] = rotations[0].sin()
            R[1, 0] = -R[0, 1]
            A = R @ A       # rotation
        elif ndim == 3:
            Sz = I.clone()
            Sz[0, 1] = Sz[1, 0] = shears[0]
            Sy = I.clone()
            Sy[0, 2] = Sz[2, 0] = shears[1]
            Sx = I.clone()
            Sx[1, 2] = Sz[2, 1] = shears[2]
            A = Sx @ Sy @ Sz @ A       # shear
            Rz = I.clone()
            Rz[0, 0] = Rz[1, 1] = rotations[0].cos()
            Rz[0, 1] = rotations[0].sin()
            Rz[1, 0] = -Rz[0, 1]
            Ry = I.clone()
            Ry[0, 0] = Ry[2, 2] = rotations[1].cos()
            Ry[0, 2] = rotations[1].sin()
            Ry[2, 0] = -Ry[0, 2]
            Rx = I.clone()
            Rx[1, 1] = Rx[2, 2] = rotations[2].cos()
            Rx[1, 2] = rotations[2].sin()
            Rx[2, 1] = -Rx[1, 2]
            A = Rx @ Ry @ Rz @ A       # rotation
        A = O.inverse() @ A
        A = T @ A

        if not fullsize:
            return A
        t = warps.affine_flow(A, fullshape).movedim(-1, 0)
        return t, A

    def apply_transform(self, x, parameters):
        flow, matrix = parameters
        y = warps.apply_flow(x[:, None], flow.movedim(0, -1),
                             padding_mode=self.bound)[:, 0]
        return prepare_output(
            dict(input=x, output=y, flow=flow, matrix=matrix),
            self.returns)


class RandomAffineTransform(RandomizedTransform):
    """
    Affine Transform with random parameters.
    """

    def __init__(self,
                 translations=0.1,
                 rotations=15,
                 shears=0.012,
                 zooms=0.15,
                 unit='fov',
                 bound='border',
                 *,
                 shared=True,
                 **kwargs):
        """

        Parameters
        ----------
        translations : Sampler or [list of] float
            Sampler or Upper bound for translation (per X/Y/Z)
        rotations : Sampler or [list of] float
            Sampler or Upper bound for rotations (about Z/Y/X), in deg
        shears : Sampler or [list of] float
            Sampler or Upper bound for shears (about Z/Y/Z)
        zooms : Sampler or [list of] float
            Sampler or Upper bound for zooms about 1 (per X/Y/Z)
        unit : {'fov', 'vox'}
            Unit of `translations`.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'matrix': The affine matrix
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(
            AffineTransform,
            dict(translations=Uniform.make(make_range(translations)),
                 rotations=Uniform.make(make_range(rotations)),
                 shears=Uniform.make(make_range(shears)),
                 zooms=Uniform.make(make_range(zooms)),
                 unit=unit,
                 bound=bound,
                 **kwargs),
            shared=shared)

    def get_parameters(self, x):
        ndim = x.dim() - 1
        sample = self.sample
        affine_keys = ['translations', 'rotations', 'shears', 'zooms']
        self.sample = {
            key: value(ndim)
            if isinstance(value, Sampler) and key in affine_keys else value
            for key, value in sample.items()
        }
        sub = super().get_parameters(x)
        self.sample = sample
        return sub


class AffineElasticTransform(Transform):
    """
    Affine + Elastic [+ Patch] transform.
    """

    def __init__(self, dmax=0.1, shape=5, steps=0,
                 translations=0, rotations=0, shears=0, zooms=0,
                 unit='fov', bound='border', patch=None, order=3,
                 *, shared=True, **kwargs):
        """

        Parameters
        ----------
        dmax : [list of] float
            Max displacement per dimension
        shape : [list of] int
            Number of spline control points
        steps : int
            Number of scaling-and-squaring integration steps
        translations : [list of] float
            Translation (per X/Y/Z)
        rotations : [list of] float
            Rotations (about Z/Y/X), in deg
        shears : [list of] float
            Translation (about Z/Y/Z)
        zooms : [list of] float
            Zoom about 1 (per X/Y/Z)
        unit : {'fov', 'vox'}
            Unit of `translations` and `dmax`.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        patch : [list of] int
            Size of random patch to extract
        order : int
            Spline order
        returns : [list or dict of] {'input', 'output', 'flow', 'controls', 'matrix'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'control': The control points of the nonlinear field
            - 'matrix': The affine matrix
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        self.patch = patch
        self.steps = steps
        self.affine = AffineTransform(
            translations, rotations, shears, zooms, unit, bound, shared=shared)
        self.elastic = ElasticTransform(
            dmax,  unit, shape, bound, steps, order, shared=shared)

    def get_parameters(self, x):
        """

        Parameters
        ----------
        x : (C, *shape) tensor

        Returns
        -------
        warp : (C, D, *shape) tensor
        control : (C, D, *self.shape) tensor
        affine : (D+1, D+1) tensor

        """
        ndim = x.ndim - 1
        fullshape = x.shape[1:]
        backend = dict(dtype=x.dtype, device=x.device)
        if not x.is_floating_point():
            backend['dtype'] = torch.get_default_dtype()

        # get transformation parameters
        A = self.affine.get_parameters(x, fullsize=False)           # (D+1, D+1)
        if self.steps:
            # request the blown up and exponentiated field
            eslasticflow, control = self.elastic.get_parameters(x, fullsize=True)   # (C, D, *shape)
        else:
            # request only the spline control points
            control = self.elastic.get_parameters(x, fullsize=False)
            eslasticflow = None

        # 1) start from identity
        patchshape = ensure_list(self.patch or fullshape, ndim)
        flow = warps.identity(patchshape, **backend)  # (*shape, D)

        if self.patch:
            # 1.b) randomly sample patch location and add offset
            patch_origin = [random.randint(0, s-p)
                            for s, p in zip(fullshape, self.patch)]
            flow += torch.as_tensor(patch_origin, **backend)

        # 2.) apply affine transform
        flow = A[:ndim, :ndim].matmul(flow.unsqueeze(-1)).squeeze(-1)
        flow = flow.add_(A[:ndim, -1])

        # 3) compose with elastic transform
        if eslasticflow is not None:
            # we sample into the blown up elastic flow,
            # which has the size of the full image
            flow = warps.apply_flow(
                eslasticflow, flow,
                padding_mode='zeros',
                has_identity=True,
            ).add_(flow.movedim(-1, 0))
        else:
            # we sample into the spline control points
            # and must rescale the sampling coordinates accordingly
            smallshape = control.shape[2:]
            scale = [(s0-1)/(s1-1) for s0, s1 in zip(smallshape, fullshape)]
            scale = torch.as_tensor(scale, **backend)
            if self.elastic.order == 1:
                # we can use pytorch
                flow = warps.apply_flow(
                    control, flow * scale,
                    padding_mode='zeros',
                    has_identity=True,
                ).add_(flow.movedim(-1, 0))
            else:
                # we must use torch-interpol
                flow = interpol.grid_pull(
                    control, flow * scale,
                    bound='zero',
                    interpolation=self.elastic.order,
                ).add_(flow.movedim(-1, 0))
        return flow, control, A

    def apply_transform(self, x, parameters):
        flow, controls, affine = parameters
        y = warps.apply_flow(x[:, None], flow.movedim(1, -1),
                             padding_mode=self.elastic.bound,
                             has_identity=True)[:, 0]
        return prepare_output(
            dict(input=x, output=y, flow=flow, controls=controls, matrix=affine),
            self.returns)


class RandomAffineElasticTransform(RandomizedTransform):
    """
    Random Affine + Elastic transform.
    """

    def __init__(self, dmax=0.1, shape=5, steps=0,
                 translations=0.1, rotations=15, shears=0.012, zooms=0.15,
                 unit='fov', bound='border', patch=None, order=3,
                 *, shared=True, **kwargs):
        """

        Parameters
        ----------
        dmax : Sampler or float
            Sampler or Upper bound for maximum displacement
        shape : Sampler or int
            Sampler or Upper bounds for number of control points
        steps : int
            Number of scaling-and-squaring integration steps
        translations : Sampler or [list of] float
            Sampler or Upper bound for translation (per X/Y/Z)
        rotations : Sampler or [list of] float
            Sampler or Upper bound for rotations (about Z/Y/X), in deg
        shears : Sampler or [list of] float
            Sampler or Upper bound for shears (about Z/Y/Z)
        zooms : Sampler or [list of] float
            Sampler or Upper bound for zooms about 1 (per X/Y/Z)
        unit : {'fov', 'vox'}
            Unit of `translations`.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        patch : [list of] int
            Size of random patch to extract
        order : int
            Spline order
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(
            AffineElasticTransform,
            dict(dmax=Uniform.make(make_range(0, dmax)),
                 shape=RandInt.make(make_range(2, shape)),
                 translations=Uniform.make(make_range(translations)),
                 rotations=Uniform.make(make_range(rotations)),
                 shears=Uniform.make(make_range(shears)),
                 zooms=Uniform.make(make_range(zooms)),
                 unit=unit,
                 bound=bound,
                 steps=steps,
                 patch=patch,
                 order=order,
                 **kwargs),
            shared=shared)

    def get_parameters(self, x):
        ndim = x.dim() - 1
        sample = self.sample
        affine_keys = ['shape', 'translations', 'rotations', 'shears', 'zooms']
        self.sample = {
            key: value(ndim)
            if isinstance(value, Sampler) and key in affine_keys else value
            for key, value in sample.items()
        }
        sub = super().get_parameters(x)
        self.sample = sample
        return sub


class MakeAffinePair(Transform):
    """
    Generate a pair made of the same image transformed in two different ways.

    This Transform returns a tuple: (transformed_input, true_transform),
    where transformed_input has the same layout as the input and
    true_transform is a dictionary with keys 'flow' and 'affine'.
    """

    def __init__(self, transform=None, *, returns=('left', 'right'), **kwargs):
        """

        Parameters
        ----------
        transform : RandomAffineTransform, default=`RandomAffineTransform()`
            An instantiated transform.
        returns : [list or dict of] {'left', 'right', 'flow', 'matrix'}

            - 'input': Input image
            - 'left': First transformed image
            - 'right': Second transformed image
            - 'flow': Displacement field that warps right to left
            - 'matrix': Affine matrix that warps right to left
                        (i.e., maps left coordinates to right coordinates)
        """
        super().__init__(shared=True, returns=returns, **kwargs)
        self.subtransform = transform or RandomAffineTransform()

    def get_parameters(self, x):
        t1 = self.subtransform.get_parameters(x)
        t2 = self.subtransform.get_parameters(x)
        p1, p2 = t1.get_parameters(x), t2.get_parameters(x)
        return t1, p1, t2, p2

    def apply_transform(self, x, parameters):
        t1, p1, t2, p2 = parameters
        x1 = t1.apply_transform(x, p1)
        x2 = t2.apply_transform(x, p2)
        mat1, mat2 = p1[1], p2[1]
        mat12 = mat2.inverse() @ mat1
        flow12 = warps.affine_flow(mat12, x.shape[1:]).movedim(-1, 0)
        return prepare_output(
            dict(input=x, left=x1, right=x2, flow=flow12, matrix=mat12),
            self.returns)


class ThroughSliceAffineTransform(Transform):
    """Each slice samples the 3D volume using a different transform"""

    def __init__(self, translations=0, rotations=0, shears=0, zooms=0,
                 slice=-1, unit='fov', bound='border',
                 *, shared=True, **kwargs):
        """

        Parameters
        ----------
        translations : list of [list of] float
            Translation per slice (per X/Y/Z)
        rotations : list of [list of] float
            Rotations per slice (about Z/Y/X), in deg
        shears : list of [list of] float
            Translation per slice (about Z/Y/Z)
        zooms : list of [list of] float
            Zoom about 1 per slice (per X/Y/Z)
        slice : {0, 1, 2}
            Slice direction
        unit : {'fov', 'vox'}
            Unit of `translations`.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}

            - 'input': First transformed image
            - 'output': Second transformed image
            - 'flow': Displacement field
            - 'matrix': Stacked affine matrices (one per slice)
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        if unit not in ('fov', 'vox'):
            raise ValueError('Unit must be one of {"fov", "vox"} '
                             f'but got "{unit}".')
        if bound not in ('zeros', 'border', 'reflection'):
            raise ValueError('Bound must be one of '
                             '{"zeros", "border", reflection"} '
                             f'but got "{bound}".')
        self.translations = translations
        self.rotations = rotations
        self.zooms = zooms
        self.shears = shears
        self.unit = unit
        self.bound = bound
        self.slice = slice

    def get_parameters(self, x, fullsize=True):
        batch, *fullshape = x.shape
        ndim = len(fullshape)
        slice = self.slice - ndim if self.slice >= 0 else self.slice
        nb_slice = fullshape[slice]
        backend = dict(dtype=x.dtype, device=x.device)
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()

        rotations = [ensure_list(r, ndim * (ndim - 1) // 2) for r in self.rotations]
        shears = [ensure_list(s, ndim * (ndim - 1) // 2) for s in self.shears]
        translations = [ensure_list(t, ndim) for t in self.translations]
        zooms = [ensure_list(z, ndim) for z in self.zooms]
        offsets = [(n-1)/2 for n in fullshape]

        if self.unit == 'fov':
            translations = [[t1 * n for t1, n in zip(t, fullshape)] for t in translations]
        rotations = [[r1 * math.pi / 180 for r1 in r] for r in rotations]

        rotations = torch.as_tensor(rotations, **backend)
        shears = torch.as_tensor(shears, **backend)
        translations = torch.as_tensor(translations, **backend)
        zooms = torch.as_tensor(zooms, **backend)
        offsets = torch.as_tensor(offsets, **backend)

        I = torch.eye(ndim+1, **backend).expand([nb_slice, ndim+1, ndim+1])
        O = torch.eye(ndim+1, **backend)
        O[:ndim, -1] = -offsets
        Z = I.clone()
        Z.diagonal(0, -1, -2)[:, :-1].copy_(1 + zooms)
        T = I.clone()
        T[:, :ndim, -1] = translations

        A = O               # origin at center of FOV
        A = Z.matmul(A)     # zoom
        if ndim == 2:
            S = I.clone()
            S[:, 0, 1] = S[:, 1, 0] = shears[:, 0]
            A = S.matmul(A)       # shear
            R = I.clone()
            R[:, 0, 0] = R[:, 1, 1] = rotations[:, 0].cos()
            R[:, 0, 1] = rotations[:, 0].sin()
            R[:, 1, 0] = -R[:, 0, 1]
            A = R.matmul(A)       # rotation
        elif ndim == 3:
            Sz = I.clone()
            Sz[:, 0, 1] = Sz[:, 1, 0] = shears[:, 0]
            Sy = I.clone()
            Sy[:, 0, 2] = Sz[:, 2, 0] = shears[:, 1]
            Sx = I.clone()
            Sx[:, 1, 2] = Sz[:, 2, 1] = shears[:, 2]
            A = Sx.matmul(Sy).matmul(Sz).matmul(A)       # shear
            Rz = I.clone()
            Rz[:, 0, 0] = Rz[:, 1, 1] = rotations[:, 0].cos()
            Rz[:, 0, 1] = rotations[:, 0].sin()
            Rz[:, 1, 0] = -Rz[:, 0, 1]
            Ry = I.clone()
            Ry[:, 0, 0] = Ry[:, 2, 2] = rotations[:, 1].cos()
            Ry[:, 0, 2] = rotations[:, 1].sin()
            Ry[:, 2, 0] = -Ry[:, 0, 2]
            Rx = I.clone()
            Rx[:, 1, 1] = Rx[:, 2, 2] = rotations[:, 2].cos()
            Rx[:, 1, 2] = rotations[:, 2].sin()
            Rx[:, 2, 1] = -Rx[:, 1, 2]
            A = Rx.matmul(Ry).matmul(Rz).matmul(A)       # rotation
        A = O.inverse().matmul(A)
        A = T.matmul(A)

        if not fullsize:
            return A
        t = warps.identity(fullshape, dtype=A.dtype, device=A.device)
        t = t.movedim(-1, 0).movedim(slice, -1).movedim(0, -1)
        t = A[:, :-1, :-1].matmul(t.unsqueeze(-1)).squeeze(-1).add_(A[:, :-1, -1])
        t = t.movedim(-1, 0).movedim(-1, slice).movedim(0, -1)
        t = warps.sub_identity_(t).movedim(-1, 0)
        return t, A

    def apply_transform(self, x, parameters):
        flow, matrix = parameters
        y = warps.apply_flow(x[None], flow.movedim(0, -1)[None],
                             padding_mode=self.bound)[0]
        return prepare_output(
            dict(input=x, output=y, flow=flow, matrix=matrix),
            self.returns)


class RandomThroughSliceAffineTransform(RandomizedTransform):
    """
    Slicewise3DAffineTransform with random parameters.
    """

    def __init__(self, translations=0.1, rotations=15,
                 shears=0, zooms=0, slice=-1, shots=2, nodes=8,
                 unit='fov', bound='border', *, shared=True, **kwargs):
        """

        Parameters
        ----------
        translations : Sampler or [list of] float
            Sampler or Upper bound for translation (per X/Y/Z)
        rotations : Sampler or [list of] float
            Sampler or Upper bound for rotations (about Z/Y/X), in deg
        shears : Sampler or [list of] float
            Sampler or Upper bound for shears (about Z/Y/Z)
        zooms : Sampler or [list of] float
            Sampler or Upper bound for zooms about 1 (per X/Y/Z)
        slice : int, optional
            Slice direction. If None, a random slice direction is selected.
        shots : int
            Number of interleaved sweeps.
            Typically, two interleaved sequences of slices are acquired
            to avoid slice cross talk.
        nodes : Sampler or int, optional
            Sampler or Upper bound for the number of nodes in the
            motion trajectory (encoded by cubic splines).
            If None, independent motions are sampled for each slice.
        unit : {'fov', 'vox'}
            Unit of `translations`.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}

            - 'input': First transformed image
            - 'output': Second transformed image
            - 'flow': Displacement field
            - 'matrix': Stacked affine matrices (one per slice)
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(
            ThroughSliceAffineTransform,
            dict(translations=Uniform.make(make_range(translations)),
                 rotations=Uniform.make(make_range(rotations)),
                 shears=Uniform.make(make_range(shears)),
                 zooms=Uniform.make(make_range(zooms)),
                 unit=unit,
                 bound=bound,
                 slice=slice,
                 **kwargs),
            shared=shared)
        if nodes:
            nodes = RandInt.make(make_range(0, nodes))
        self.nodes = nodes
        self.shots = shots

    def get_parameters(self, x):
        ndim = x.ndim - 1
        sample, self.sample = self.sample, dict(self.sample)

        # get slice direction
        if self.sample['slice'] is None:
            self.sample['slice'] = RandInt(0, ndim)
        if isinstance(self.sample['slice'], Sampler):
            self.sample['slice'] = self.sample['slice']()

        # get number of independent motions
        nb_slices = x.shape[1:][self.sample['slice']]
        nodes = self.nodes
        if isinstance(nodes, Sampler):
            nodes = nodes()
        nodes = min(nodes or nb_slices, nb_slices)

        # sample parameters per slice per XYZ
        self.sample = {
            key: [val(ndim) for _ in range(nodes)]
            if isinstance(val, Sampler) else val
            for key, val in self.sample.items()
        }

        # cubic interpolation of motion parameters
        if nodes < nb_slices:
            def node2slice(x):
                x = torch.as_tensor(x, dtype=torch.float32).T  # [3, N]
                x = interpol.resize(x, shape=[nb_slices],
                                    interpolation=3, bound='replicate',
                                    prefilter=False).T  # [S, 3]
                y = torch.empty_like(x)
                for i in range(self.shots):
                    y[i::self.shots] = x[:len(y[i::self.shots])]
                    x = x[len(y[i::self.shots]):]
                return y.tolist()
            affine_keys = ['translations', 'rotations', 'shears', 'zooms']
            self.sample = {
                key: node2slice(val) if key in affine_keys else val
                for key, val in self.sample.items()
            }

        out = super().get_parameters(x)
        self.sample = sample
        return out
