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
from .base import NonFinalTransform, FinalTransform
from .baseutils import prepare_output, return_requires
from .random import Sampler, Uniform, RandInt, make_range
from .utils import warps
from .utils.py import ensure_list, cast_like


class ElasticTransform(NonFinalTransform):
    """
    Elastic transform encoded by cubic splines.
    The number of control points is fixed but coefficients are
    randomly sampled.
    """

    class Final(FinalTransform):
        """Final (deterministic) elastic transform"""

        def __init__(self, flow=None, controls=None, steps=0, order=3,
                     bound='border', **kwargs):
            """
            Parameters
            ----------
            flow : (C, D, *spatial) tensor
                Flow field
            control : (C, D, *shape) tensor
                Spline control points
            steps : int
                Number of scaling and squaring steps
            order : int
                Spline order
            bound : str
                Boundary condition
            """
            super().__init__(**kwargs)
            self.flow = flow
            self.controls = controls
            self.steps = steps
            self.order = order
            self.bound = bound

        def make_flow(self, control, fullshape):
            """Upsample the control points to the final full size

            Parameters
            ----------
            control : (C, D, *shape) tensor
                Spline control points
            fullshape : list[int]
                Target shape

            Returns
            -------
            flow : (C, D, *fullshape) tensor
                Upampled flow field

            """
            if self.order == 1:
                mode = ('trilinear' if len(fullshape) == 3 else
                        'bilinear' if len(fullshape) == 2 else
                        'linear')
                flow = interpolate(
                    control[None], fullshape, mode=mode, align_corners=True
                )[0]
            else:
                flow = interpol.resize(
                    control, shape=fullshape, interpolation=self.order,
                    prefilter=False
                )
            if self.steps:
                flow = warps.exp_velocity(
                    flow.movedim(1, -1), self.steps
                ).movedim(-1, 1)
            return flow

        def apply(self, x):
            """Deform the input tensor

            Parameters
            ----------
            x : (C, *spatial) tensor
                Input tensor

            Returns
            -------
            out : [dict or list of] tensor
                The tensors returned by this function depend on the
                value of `self.returns`. See `ElasticTransform`.

            """
            flow = cast_like(self.flow, x)
            controls = cast_like(self.controls, x)
            required = return_requires(self.returns)
            if flow is None and ('flow' in required or 'output' in required):
                flow = self.make_flow(controls, x.shape[1:])
            y = None
            if 'output' in required:
                y = warps.apply_flow(
                    x[:, None], flow.movedim(1, -1), padding_mode=self.bound
                )[:, 0]
            return prepare_output(
                dict(input=x, output=y, flow=flow, controls=controls),
                self.returns
            )

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

        Keyword Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'controls'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'controls': The control points of the displacement field
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        if unit not in ('fov', 'vox'):
            raise ValueError(
                'Unit must be one of {"fov", "vox"} but got "{unit}".'
            )
        if bound not in ('zeros', 'border', 'reflection'):
            raise ValueError(
                'Bound must be one of {"zeros", "border", reflection"} '
                f'but got "{bound}".'
            )
        self.dmax = dmax
        self.unit = unit
        self.bound = bound
        self.shape = shape
        self.steps = steps
        self.order = order

    def make_final(self, x, max_depth=float('inf'), flow=True):
        """
        Generate a deterministic transform with constant parameters

        Parameters
        ----------
        x : (C, *spatial) tensor
            Tensor to deform
        max_depth : int
            Maximum number of transforms to unroll
        flow : bool
            Precompute the upsampled flow field

        Returns
        -------
        xform : ElasticTransform.Final
            Final transform with parameters
            - `flow : (C, D, *spatial) tensor`, the upsampled flow field
            - `control : (C, D, *shape) tensor`, the spline control points

        """
        if max_depth == 0:
            return self
        batch, *fullshape = x.shape
        if 'channels' in self.shared:
            batch = 1
        ndim = len(fullshape)
        smallshape = ensure_list(self.shape, ndim)
        dmax = ensure_list(self.dmax, ndim)
        backend = dict(dtype=x.dtype, device=x.device)
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        if self.unit == 'fov':
            dmax = [d * f for d, f in zip(dmax, fullshape)]
        controls = torch.rand([batch, ndim, *smallshape], **backend)
        for d in range(ndim):
            controls[:, d].sub_(0.5).mul_(2*dmax[d])
        if flow:
            if self.order == 1:
                mode = ('trilinear' if len(fullshape) == 3 else
                        'bilinear' if len(fullshape) == 2 else
                        'linear')
                flow = interpolate(
                    controls[None], fullshape, mode=mode, align_corners=True
                )[0]
            else:
                flow = interpol.resize(
                    controls, shape=fullshape, interpolation=self.order,
                    prefilter=False
                )
            if self.steps:
                flow = warps.exp_velocity(
                    flow.movedim(1, -1), self.steps
                ).movedim(-1, 1)
        else:
            flow = None
        return self.Final(
            flow, controls, self.steps, self.order, self.bound,
            **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomElasticTransform(NonFinalTransform):
    """
    Elastic Transform with random parameters.
    """

    def __init__(self, dmax=0.1, shape=5, unit='fov', bound='border',
                 steps=0, order=3, *, shared=True, shared_flow=None, **kwargs):
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

        Keyword Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'controls'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'controls': The control points of the displacement field
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Whether to share random parameters across tensors and/or channels
        shared_flow : {'channels', 'tensors', 'channels+tensors', '', None}
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """
        super().__init__(shared=shared, **kwargs)
        self.dmax = Uniform.make(make_range(0, dmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.unit = unit
        self.bound = bound
        self.steps = steps
        self.order = order
        self.shared_flow = shared_flow

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        ndim = x.ndim - 1

        dmax, shape, order = self.dmax, self.shape, self.order
        if isinstance(dmax, Sampler):
            dmax = dmax()
        if isinstance(shape, Sampler):
            shape = shape(ndim)
        if isinstance(order, Sampler):
            order = order()
        shared_flow = self.shared_flow
        if shared_flow is None:
            shared_flow = self.shared

        return ElasticTransform(
            dmax=dmax, shape=shape, order=order,
            unit=self.unit, bound=self.bound, steps=self.steps,
            shared=shared_flow,
        ).make_final(x, max_depth-1)


class AffineTransform(NonFinalTransform):
    """
    Apply an affine transform encoded by translations, rotations,
    shears and zooms.

    The affine matrix is defined as:
        `A = T @ Rx @ Ry @ Rz @ Sx @ Sy Sz @ Z`
    with the center of the field of view used as center of rotation.
    (A is a matrix so the transforms are applied right to left)
    """

    class Final(FinalTransform):
        """Apply an affine transform encoded by an affine matrix"""

        def __init__(self, flow=None, matrix=None, bound='border', **kwargs):
            """
            Parameters
            ----------
            flow : ([C], D, *spatial) tensor
                Flow field
            matrix : ([C], D+1, D+1) tensor
                Matrix
            """
            super().__init__(**kwargs)
            self.flow = flow
            self.matrix = matrix
            self.bound = bound

        def make_flow(self, matrix, shape):
            """Build affine flow from matrix

            Parameters
            ----------
            """
            return warps.affine_flow(matrix, shape).movedim(-1, 0)

        def apply(self, x):
            flow = cast_like(self.flow, x)
            matrix = cast_like(self.matrix, x)
            required = return_requires(self.returns)
            if flow is None and ('flow' in required or 'output' in required):
                flow = self.make_flow(matrix, x.shape[1:])
            y = None
            if 'output' in required:
                y = warps.apply_flow(x[:, None], flow.movedim(0, -1),
                                     padding_mode=self.bound)[:, 0]
            return prepare_output(
                dict(input=x, output=y, flow=flow, matrix=matrix),
                self.returns
            )

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

        Keyword Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'matrix': The affine matrix
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        if unit not in ('fov', 'vox'):
            raise ValueError(
                'Unit must be one of {"fov", "vox"} but got "{unit}".'
            )
        if bound not in ('zeros', 'border', 'reflection'):
            raise ValueError(
                'Bound must be one of {"zeros", "border", reflection"} '
                f'but got "{bound}".'
            )
        self.translations = translations
        self.rotations = rotations
        self.zooms = zooms
        self.shears = shears
        self.unit = unit
        self.bound = bound

    def make_final(self, x, max_depth=float('inf'), flow=True):
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

        E = torch.eye(ndim+1, **backend)
        # offset to center
        F = E.clone()
        F[:ndim, -1] = -offsets
        # zooms
        Z = E.clone()
        Z.diagonal(0, -1, -2)[:-1].copy_(1 + zooms)
        # translations
        T = torch.eye(ndim+1, **backend)
        T[:ndim, -1] = translations

        A = F               # origin at center of FOV
        A = Z @ A           # zoom
        if ndim == 2:
            S = E.clone()
            S[0, 1] = S[1, 0] = shears[0]
            A = S @ A       # shear
            R = E.clone()
            R[0, 0] = R[1, 1] = rotations[0].cos()
            R[0, 1] = rotations[0].sin()
            R[1, 0] = -R[0, 1]
            A = R @ A       # rotation
        elif ndim == 3:
            Sz = E.clone()
            Sz[0, 1] = Sz[1, 0] = shears[0]
            Sy = E.clone()
            Sy[0, 2] = Sz[2, 0] = shears[1]
            Sx = E.clone()
            Sx[1, 2] = Sz[2, 1] = shears[2]
            A = Sx @ Sy @ Sz @ A       # shear
            Rz = E.clone()
            Rz[0, 0] = Rz[1, 1] = rotations[0].cos()
            Rz[0, 1] = rotations[0].sin()
            Rz[1, 0] = -Rz[0, 1]
            Ry = E.clone()
            Ry[0, 0] = Ry[2, 2] = rotations[1].cos()
            Ry[0, 2] = rotations[1].sin()
            Ry[2, 0] = -Ry[0, 2]
            Rx = E.clone()
            Rx[1, 1] = Rx[2, 2] = rotations[2].cos()
            Rx[1, 2] = rotations[2].sin()
            Rx[2, 1] = -Rx[1, 2]
            A = Rx @ Ry @ Rz @ A       # rotation
        A = F.inverse() @ A
        A = T @ A

        if flow:
            flow = warps.affine_flow(A, fullshape).movedim(-1, 0)
        else:
            flow = None
        return self.Final(
            flow, A, self.bound, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomAffineTransform(NonFinalTransform):
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
                 shared_matrix=None,
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

        Keyword Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'matrix': The affine matrix
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Whether to share random parameters across tensors and/or channels
        shared_matrix : {'channels', 'tensors', 'channels+tensors', '', None}
            Whether to share matrices across tensors and/or channels.
            By default: same as `shared`
        """
        super().__init__(shared=shared, **kwargs)
        self.translations = Uniform.make(make_range(translations))
        self.rotations = Uniform.make(make_range(rotations))
        self.shears = Uniform.make(make_range(shears))
        self.zooms = Uniform.make(make_range(zooms))
        self.unit = unit
        self.bound = bound
        self.shared_matrix = shared_matrix

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        ndim = x.ndim - 1

        translations = self.translations
        rotations = self.rotations
        shears = self.shears
        zooms = self.zooms
        unit = self.unit
        bound = self.bound
        shared_matrix = self.shared_matrix
        if isinstance(translations, Sampler):
            translations = translations(ndim)
        if isinstance(rotations, Sampler):
            rotations = rotations(ndim)
        if isinstance(shears, Sampler):
            shears = shears(ndim)
        if isinstance(zooms, Sampler):
            zooms = zooms(ndim)
        if shared_matrix is None:
            shared_matrix = self.shared

        return AffineTransform(
            translations, rotations, shears, zooms,
            unit=unit, bound=bound, shared=shared_matrix,
            **self.get_prm()
        ).make_final(x, max_depth-1)


class AffineElasticTransform(NonFinalTransform):
    """
    Affine + Elastic [+ Patch] transform.
    """

    class Final(FinalTransform):
        def __init__(self, flow, controls, affine, bound='border', **kwargs):
            """
            Parameters
            ----------
            flow : (C, D, *spatial) tensor
            controls : (C, D, *spatial) tensor
            affine : ([C], D+1, D+1) tensor
            bound : str
            """
            super().__init__(**kwargs)
            self.flow = flow
            self.controls = controls
            self.affine = affine
            self.bound = bound

        def apply(self, x):
            flow = cast_like(self.flow, x)
            controls = cast_like(self.controls, x)
            affine = cast_like(self.affine, x)
            y = warps.apply_flow(x[:, None], flow.movedim(1, -1),
                                 padding_mode=self.bound,
                                 has_identity=True)[:, 0]
            return prepare_output(
                dict(input=x, output=y, flow=flow,
                     controls=controls, matrix=affine),
                self.returns)

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

        Keyword Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'controls', 'matrix'}
            - 'input': The input image
            - 'output': The deformed image
            - 'flow': The displacement field
            - 'control': The control points of the nonlinear field
            - 'matrix': The affine matrix
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same transform to all images/channels
        """  # noqa: E501
        super().__init__(shared=shared, **kwargs)
        self.patch = patch
        self.steps = steps
        self.affine = AffineTransform(
            translations, rotations, shears, zooms, unit, bound, shared=shared)
        self.elastic = ElasticTransform(
            dmax,  unit, shape, bound, steps, order, shared=shared)

    def make_final(self, x, max_depth=float('inf')):
        """

        Parameters
        ----------
        x : (C, *shape) tensor
        max_depth : int

        Returns
        -------
        warp : (C, D, *shape) tensor
        controls : (C, D, *self.shape) tensor
        affine : (D+1, D+1) tensor

        """
        ndim = x.ndim - 1
        fullshape = x.shape[1:]
        backend = dict(dtype=x.dtype, device=x.device)
        if not x.is_floating_point():
            backend['dtype'] = torch.get_default_dtype()

        # get transformation parameters
        A = self.affine.make_final(x, flow=False).matrix  # (D+1, D+1)
        if self.steps:
            # request the blown up and exponentiated field
            xform = self.elastic.make_final(x, flow=True)
            eslasticflow, controls = xform.flow, xform.controls
            # (C, D, *shape)
        else:
            # request only the spline control points
            controls = self.elastic.make_final(x, flow=False).controls
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
            smallshape = controls.shape[2:]
            scale = [(s0-1)/(s1-1) for s0, s1 in zip(smallshape, fullshape)]
            scale = torch.as_tensor(scale, **backend)
            if self.elastic.order == 1:
                # we can use pytorch
                flow = warps.apply_flow(
                    controls, flow * scale,
                    padding_mode='zeros',
                    has_identity=True,
                ).add_(flow.movedim(-1, 0))
            else:
                # we must use torch-interpol
                flow = interpol.grid_pull(
                    controls, flow * scale,
                    bound='zero',
                    interpolation=self.elastic.order,
                ).add_(flow.movedim(-1, 0))

        return self.Final(
            flow, controls, A, self.elastic.bound, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomAffineElasticTransform(NonFinalTransform):
    """
    Random Affine + Elastic transform.
    """

    def __init__(
        self,
        dmax=0.1,
        shape=5,
        steps=0,
        translations=0.1,
        rotations=15,
        shears=0.012,
        zooms=0.15,
        unit='fov',
        bound='border',
        patch=None,
        order=3,
        *,
        shared=True,
        shared_flow=None,
        **kwargs
    ):
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

        Keyword Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same hyperparameters to all images/channels
        shared_flow : {'channels', 'tensors', 'channels+tensors', '', None}
            Apply the same random flow to all images/channels.
            Default: same as shared
        """
        super().__init__(shared=shared, **kwargs)
        self.dmax = Uniform.make(make_range(0, dmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.translations = Uniform.make(make_range(translations))
        self.rotations = Uniform.make(make_range(rotations))
        self.shears = Uniform.make(make_range(shears))
        self.zooms = Uniform.make(make_range(zooms))
        self.unit = unit
        self.bound = bound
        self.steps = steps
        self.patch = patch
        self.order = order
        self.shared_flow = shared_flow

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self

        ndim = x.ndim - 1
        dmax = self.dmax
        shape = self.shape
        translations = self.translations
        rotations = self.rotations
        shears = self.shears
        zooms = self.zooms
        unit = self.unit
        bound = self.bound
        steps = self.steps
        patch = self.patch
        order = self.order
        shared_flow = self.shared_flow

        if isinstance(dmax, Sampler):
            dmax = dmax()
        if isinstance(shape, Sampler):
            shape = shape(ndim)
        if isinstance(translations, Sampler):
            translations = translations(ndim)
        if isinstance(rotations, Sampler):
            rotations = rotations(ndim)
        if isinstance(shears, Sampler):
            shears = shears(ndim)
        if isinstance(zooms, Sampler):
            zooms = zooms(ndim)
        if isinstance(order, Sampler):
            order = order(ndim)
        if shared_flow is None:
            shared_flow = self.shared

        return AffineElasticTransform(
            dmax=dmax,
            shape=shape,
            translations=translations,
            rotations=rotations,
            shears=shears,
            zooms=zooms,
            unit=unit,
            bound=bound,
            steps=steps,
            patch=patch,
            order=order,
            shared=shared_flow,
            **self.get_prm(),
        ).make_final(x, max_depth-1)


class MakeAffinePair(NonFinalTransform):
    """
    Generate a pair made of the same image transformed in two different ways.

    This Transform returns a tuple: (transformed_input, true_transform),
    where transformed_input has the same layout as the input and
    true_transform is a dictionary with keys 'flow' and 'affine'.
    """

    class Final(FinalTransform):
        def __init__(self, left, right, **kwargs):
            super().__init__(**kwargs)
            self.left = left
            self.right = right

        def apply(self, x):
            x1 = self.left.apply(x)
            x2 = self.right.apply(x)
            mat1, mat2 = self.left.matrix, self.right.matrix
            mat1, mat2 = cast_like(mat1, x), cast_like(mat2, x)
            mat12 = mat2.inverse() @ mat1
            flow12 = warps.affine_flow(mat12, x.shape[1:]).movedim(-1, 0)
            return prepare_output(
                dict(input=x, left=x1, right=x2, flow=flow12, matrix=mat12),
                self.returns)

    def __init__(self, transform=None, *, returns=('left', 'right'), **kwargs):
        """

        Parameters
        ----------
        transform : RandomAffineTransform, default=`RandomAffineTransform()`
            An instantiated transform.

        Keyword Parameters
        ------------------
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

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        left = self.subtransform.make_final(x)
        right = self.subtransform.make_final(x)
        return self.Final(
            left, right, **self.get_prm
        ).make_final(x, max_depth-1)


class ThroughSliceAffineTransform(NonFinalTransform):
    """Each slice samples the 3D volume using a different transform"""

    class Final(FinalTransform):
        def __init__(self, flow, matrix, bound='border', **kwargs):
            super().__init__(**kwargs)
            self.flow = flow
            self.matrix = matrix
            self.bound = bound

        def apply(self, x):
            flow = cast_like(self.flow, x)
            matrix = cast_like(self.matrix, x)
            y = warps.apply_flow(x[None], flow.movedim(0, -1)[None],
                                 padding_mode=self.bound)[0]
            return prepare_output(
                dict(input=x, output=y, flow=flow, matrix=matrix),
                self.returns
            )

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

        Keyword Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}

            - 'input': First transformed image
            - 'output': Second transformed image
            - 'flow': Displacement field
            - 'matrix': Stacked affine matrices (one per slice)
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        if unit not in ('fov', 'vox'):
            raise ValueError(
                'Unit must be one of {"fov", "vox"} but got ' + f'{unit}".'
            )
        if bound not in ('zeros', 'border', 'reflection'):
            raise ValueError(
                'Bound must be one of {"zeros", "border", reflection"} '
                f'but got "{bound}".'
            )
        self.translations = translations
        self.rotations = rotations
        self.zooms = zooms
        self.shears = shears
        self.unit = unit
        self.bound = bound
        self.slice = slice

    def make_final(self, x, max_depth=float('inf'), flow=True):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth, flow=True)

        fullshape = x.shape[1:]
        ndim = len(fullshape)
        slice = self.slice - ndim if self.slice >= 0 else self.slice
        nb_slice = fullshape[slice]
        backend = dict(dtype=x.dtype, device=x.device)
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()

        rotations = [
            ensure_list(r, ndim * (ndim - 1) // 2) for r in self.rotations
        ]
        shears = [
            ensure_list(s, ndim * (ndim - 1) // 2) for s in self.shears
        ]
        translations = [
            ensure_list(t, ndim) for t in self.translations
        ]
        zooms = [
            ensure_list(z, ndim) for z in self.zooms
        ]
        offsets = [(n-1)/2 for n in fullshape]

        if self.unit == 'fov':
            translations = [
                [t1 * n for t1, n in zip(t, fullshape)] for t in translations
            ]
        rotations = [
            [r1 * math.pi / 180 for r1 in r] for r in rotations
        ]

        rotations = torch.as_tensor(rotations, **backend)
        shears = torch.as_tensor(shears, **backend)
        translations = torch.as_tensor(translations, **backend)
        zooms = torch.as_tensor(zooms, **backend)
        offsets = torch.as_tensor(offsets, **backend)

        E = torch.eye(ndim+1, **backend).expand([nb_slice, ndim+1, ndim+1])
        F = torch.eye(ndim+1, **backend)
        F[:ndim, -1] = -offsets
        Z = E.clone()
        Z.diagonal(0, -1, -2)[:, :-1].copy_(1 + zooms)
        T = E.clone()
        T[:, :ndim, -1] = translations

        A = F               # origin at center of FOV
        A = Z.matmul(A)     # zoom
        if ndim == 2:
            S = E.clone()
            S[:, 0, 1] = S[:, 1, 0] = shears[:, 0]
            A = S.matmul(A)       # shear
            R = E.clone()
            R[:, 0, 0] = R[:, 1, 1] = rotations[:, 0].cos()
            R[:, 0, 1] = rotations[:, 0].sin()
            R[:, 1, 0] = -R[:, 0, 1]
            A = R.matmul(A)       # rotation
        elif ndim == 3:
            Sz = E.clone()
            Sz[:, 0, 1] = Sz[:, 1, 0] = shears[:, 0]
            Sy = E.clone()
            Sy[:, 0, 2] = Sz[:, 2, 0] = shears[:, 1]
            Sx = E.clone()
            Sx[:, 1, 2] = Sz[:, 2, 1] = shears[:, 2]
            A = Sx.matmul(Sy).matmul(Sz).matmul(A)       # shear
            Rz = E.clone()
            Rz[:, 0, 0] = Rz[:, 1, 1] = rotations[:, 0].cos()
            Rz[:, 0, 1] = rotations[:, 0].sin()
            Rz[:, 1, 0] = -Rz[:, 0, 1]
            Ry = E.clone()
            Ry[:, 0, 0] = Ry[:, 2, 2] = rotations[:, 1].cos()
            Ry[:, 0, 2] = rotations[:, 1].sin()
            Ry[:, 2, 0] = -Ry[:, 0, 2]
            Rx = E.clone()
            Rx[:, 1, 1] = Rx[:, 2, 2] = rotations[:, 2].cos()
            Rx[:, 1, 2] = rotations[:, 2].sin()
            Rx[:, 2, 1] = -Rx[:, 1, 2]
            A = Rx.matmul(Ry).matmul(Rz).matmul(A)       # rotation
        A = F.inverse().matmul(A)
        A = T.matmul(A)

        if flow:
            flow = warps.identity(fullshape, dtype=A.dtype, device=A.device)
            flow = flow.movedim(-1, 0).movedim(slice, -1).movedim(0, -1)
            flow = A[:, :-1, :-1].matmul(flow.unsqueeze(-1)).squeeze(-1)
            flow += A[:, :-1, -1]
            flow = flow.movedim(-1, 0).movedim(-1, slice).movedim(0, -1)
            flow = warps.sub_identity_(flow).movedim(-1, 0)
        else:
            flow = None

        return self.Final(
            flow, A, self.bound, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomThroughSliceAffineTransform(NonFinalTransform):
    """
    Slicewise3DAffineTransform with random parameters.
    """

    def __init__(self, translations=0.1, rotations=15,
                 shears=0, zooms=0, slice=-1, shots=2, nodes=8,
                 unit='fov', bound='border',
                 *, shared=True, shared_matrix=None, **kwargs):
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

        Keyword Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'flow', 'matrix'}

            - 'input': First transformed image
            - 'output': Second transformed image
            - 'flow': Displacement field
            - 'matrix': Stacked affine matrices (one per slice)
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same parameters to all images/channels
        shared_matrix : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same affine matrix to all images/channels.
            Default: same as `shared`.
        """
        super().__init__(shared=shared, **kwargs)
        self.translations = Uniform.make(make_range(translations))
        self.rotations = Uniform.make(make_range(rotations))
        self.shears = Uniform.make(make_range(shears))
        self.zooms = Uniform.make(make_range(zooms))
        self.unit = unit
        self.bound = bound
        self.slice = slice
        if nodes:
            nodes = RandInt.make(make_range(0, nodes))
        self.nodes = nodes
        self.shots = shots
        self.share_matrix = shared_matrix

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth, flow=True)

        ndim = x.ndim - 1

        # get slice direction
        slice = self.slice
        if slice is None:
            slice = RandInt(0, ndim)
        if isinstance(slice, Sampler):
            slice = slice()

        # get number of independent motions
        nb_slices = x.shape[1:][slice]
        nodes = self.nodes
        if isinstance(nodes, Sampler):
            nodes = nodes()
        nodes = min(nodes or nb_slices, nb_slices)

        # sample parameters per slice per XYZ
        translations = self.translations
        rotations = self.rotations
        shears = self.shears
        zooms = self.zooms
        if isinstance(translations, Sampler):
            translations = translations(ndim)
        if isinstance(rotations, Sampler):
            rotations = rotations((ndim*(ndim-1))//2)
        if isinstance(shears, Sampler):
            shears = shears((ndim*(ndim-1))//2)
        if isinstance(zooms, Sampler):
            zooms = zooms(ndim)

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
            translations = node2slice(translations)
            rotations = node2slice(rotations)
            shears = node2slice(shears)
            zooms = node2slice(zooms)

        shared_matrix = self.shared_matrix
        if shared_matrix is None:
            shared_matrix = self.shared

        return ThroughSliceAffineTransform(
            translations, rotations, shears, zooms,
            slice=slice, unit=self.unit, bound=self.bound,
            shared=shared_matrix, **self.get_prm()
        ).make_final(x, max_depth-1)
