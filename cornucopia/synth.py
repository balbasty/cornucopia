"""
SynthShape generator

??? reference
    1.  Billot, B., Greve, D., Van Leemput, K., Fischl, B., Iglesias, J.E.
        and Dalca, A.V., 2020.
        [**A learning strategy for contrast-agnostic MRI segmentation.**](http://proceedings.mlr.press/v121/billot20a/billot20a.pdf)
        In _Proceedings of the Third Conference on Medical Imaging with Deep Learning_,
        PMLR 121, pp. 75-93.

            @inproceedings{billot2020learning,
                title       = {A Learning Strategy for Contrast-agnostic MRI Segmentation},
                author      = {Billot, Benjamin and Greve, Douglas N. and Van Leemput, Koen and Fischl, Bruce and Iglesias, Juan Eugenio and Dalca, Adrian},
                booktitle   = {Proceedings of the Third Conference on Medical Imaging with Deep Learning},
                pages       = {75--93},
                year        = {2020},
                editor      = {Arbel, Tal and Ben Ayed, Ismail and de Bruijne, Marleen and Descoteaux, Maxime and Lombaert, Herve and Pal, Christopher},
                volume      = {121},
                series      = {Proceedings of Machine Learning Research},
                month       = {06--08 Jul},
                publisher   = {PMLR},
                pdf         = {http://proceedings.mlr.press/v121/billot20a/billot20a.pdf},
                url         = {https://proceedings.mlr.press/v121/billot20a.html}
            }

    2.  Billot, B., Robinson, E., Dalca, A.V. and Iglesias, J.E., 2020.
        [**Partial volume segmentation of brain MRI scans of any resolution and contrast.**](https://arxiv.org/abs/2004.10221)
        In _Medical Image Computing and Computer Assisted Intervention-MICCAI 2020:
        23rd International Conference_, Lima, Peru, October 4-8, 2020,
        Proceedings, Part VII 23 (pp. 177-187). Springer International Publishing.

            @inproceedings{billot2020partial,
            title         = {Partial volume segmentation of brain MRI scans of any resolution and contrast},
            author        = {Billot, Benjamin and Robinson, Eleanor and Dalca, Adrian V and Iglesias, Juan Eugenio},
            booktitle     = {Medical Image Computing and Computer Assisted Intervention--MICCAI 2020: 23rd International Conference, Lima, Peru, October 4--8, 2020, Proceedings, Part VII 23},
            pages         = {177--187},
            year          = {2020},
            organization  = {Springer},
            url           = {https://arxiv.org/abs/2004.10221}
            }

    3.  Hoffmann, M., Billot, B., Greve, D.N., Iglesias, J.E., Fischl, B.
        and Dalca, A.V., 2021.
        [**SynthMorph: learning contrast-invariant registration without acquired images.**](https://arxiv.org/pdf/2004.10282)
        _IEEE transactions on medical imaging_, 41(3), pp.543-558.

            @article{hoffmann2021synthmorph,
            title     = {SynthMorph: learning contrast-invariant registration without acquired images},
            author    = {Hoffmann, Malte and Billot, Benjamin and Greve, Douglas N and Iglesias, Juan Eugenio and Fischl, Bruce and Dalca, Adrian V},
            journal   = {IEEE transactions on medical imaging},
            volume    = {41},
            number    = {3},
            pages     = {543--558},
            year      = {2021},
            publisher = {IEEE},
            url       = {https://arxiv.org/pdf/2004.10282}
            }

    4.  Billot, B., Greve, D.N., Puonti, O., Thielscher, A., Van Leemput, K.,
        Fischl, B., Dalca, A.V. and Iglesias, J.E., 2023.
        [**SynthSeg: Segmentation of brain MRI scans of any contrast and resolution
        without retraining.**](https://www.sciencedirect.com/science/article/pii/S1361841523000506)
        _Medical image analysis_, 86, p.102789.

            @article{billot2023synthseg,
            title     = {SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining},
            author    = {Billot, Benjamin and Greve, Douglas N and Puonti, Oula and Thielscher, Axel and Van Leemput, Koen and Fischl, Bruce and Dalca, Adrian V and Iglesias, Juan Eugenio and others},
            journal   = {Medical image analysis},
            volume    = {86},
            pages     = {102789},
            year      = {2023},
            publisher = {Elsevier},
            url       = {https://www.sciencedirect.com/science/article/pii/S1361841523000506}
            }

"""  # noqa: E501
__all__ = [
    'SynthFromLabelTransform',
    'ApplySynthFromLabelTransform',
    'IntensityTransform'
]

# stdlib
import random as pyrandom
from numbers import Number
from math import inf

# dependencies
import torch
import typing_extensions as tx
from torch import Tensor

# internals
from .baseutils import Kwargs, Returned, prepare_output
from .base import Transform, FinalTransform, NonFinalTransform
from .special import (
    SequentialTransform,
    RandomizedTransform,
    SwitchTransform,
    IdentityTransform,
)
from .labels import (
    RandomGaussianMixtureTransform,
    RelabelTransform,
    OneHotTransform,
)
from .intensity import (
    RandomMulFieldTransform,
    RandomGammaTransform,
    QuantileTransform,
)
from .psf import (
    RandomSmoothTransform,
    RandomLowResSliceTransform,
    RandomLowResTransform,
)
from .noise import (
    RandomChiNoiseTransform,
    GFactorTransform,
)
from .geometric import RandomAffineElasticTransform
from .random import Sampler, Uniform, RandInt, LogNormal
from .io import LoadTransform
from . import typing as cct


class IntensityTransform(SequentialTransform):
    """Common intensity augmentation for MRI and related images

    The arguments control the *range* of the distributions from which
    the transform parameters are sampled.

    It is also possible to directly provide the probability distribution
    from which to sample the parametes. In this case, it **must** be a
    [`Sampler`][cornucopia.random.Sampler] instance.

    Setting any argument to `False` disables the corresponding transform
    entirely.

    ??? reference
        Billot, B., Greve, D.N., Puonti, O., Thielscher, A., Van Leemput, K.,
        Fischl, B., Dalca, A.V. and Iglesias, J.E., 2023.
        [**SynthSeg: Segmentation of brain MRI scans of any contrast and resolution
        without retraining.**](https://www.sciencedirect.com/science/article/pii/S1361841523000506)
        _Medical image analysis_, 86, p.102789.

            @article{billot2023synthseg,
              title     = {SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining},
              author    = {Billot, Benjamin and Greve, Douglas N and Puonti, Oula and Thielscher, Axel and Van Leemput, Koen and Fischl, Bruce and Dalca, Adrian V and Iglesias, Juan Eugenio and others},
              journal   = {Medical image analysis},
              volume    = {86},
              pages     = {102789},
              year      = {2023},
              publisher = {Elsevier},
              url       = {https://www.sciencedirect.com/science/article/pii/S1361841523000506}
            }
    """  # noqa: E501

    def __init__(
        self,
        bias: cct.SamplerOrBoundOrBool[int] = 7,
        bias_strength: cct.SamplerOrBoundOrBool[float] = 0.5,
        gamma: cct.SamplerOrBoundOrBool[float] = 0.6,
        motion_fwhm: cct.SamplerOrBoundOrBool[float] = 3,
        resolution: cct.SamplerOrBoundOrBool[float] = 8,
        snr: cct.SamplerOrBoundOrBool[float] = 10,
        gfactor: cct.SamplerOrBoundOrBool[int] = 5,
        order: cct.SamplerOrBoundOrBool[int] = 3,
        **kwargs
    ):
        """
        Parameters
        ----------
        bias : Sampler | int | {False}
            The sampled value controls the smoothness of the intensity
            bias field (smaller values yield smoother fields).
            If an `int`, sample from `RandInt(2, value)`.
        bias_strength : Sampler | float in (0..1)
            The maximum magnitude of the bias field (about 1).
            If a `float`, sample from `Uniform(0, value)`.
            The minimum and maximum values of the bias field will be
            `1 - bias_strength` and `1 + bias_strength`.
        gamma : Sampler | float | {False}
            The Gamma transform squeezes intensities such that the contrast
            to noise ratio is decreased (positive values lead to less
            decreased contrast, positive values lead to increased contrast).
            If a `float`, sample the gamma exponent from `LogNormal(0, value)`.
        motion_fwhm : Sampler | float | {False}
            A blur can be perform to model the point spread function or
            motion-related smearing. The amount of smoothing is encoded by
            the full-width at half-maximum (FWHM) of the underlying
            Gaussian kernel.
            If a `float`, sample the FWHM from `Uniform(0, value)`.
        resolution : Sampler | float | {False}
            Thick-slice or isotropic low-resolution (LR) images are randomly
            applied. and their (through-slice or iso) resolution is
            controlled here. It is defined as a proportion of the
            high-resolution voxel size (i.e., a resolution of `4` mean
            that the LR voxel size will be four times as large as the
            input voxel size)
            If a `float`, sampled form `Uniform(0, value)`.
        snr : Sampler | float | {False}
            The amount of noise added is encoded by the signal-to-noise ratio
            (SNR) of the noisy image (larger sampled values yield less
            noisy images).
            If a `float`, the value is a lower bound for SNR (no image
            will have a poorer SNR than this). The noise variance is
            then sampled from `Uniform(0, 1/snr)`.
        gfactor : Sampler | int | {False}
            The g-factor is a smooth field that locally scales the noise
            variance. The `gfactor` argument controls the smoothness of
            the g-factor field.
            If an `int`, sample from `RandInt(2, value)`.
        order : {1..7}
            Spline order of the bias/g-factor fields (1 is much faster)
        """
        steps = []

        if bias:
            if not isinstance(bias, Sampler):
                bias = RandInt(2, bias)
            if not isinstance(bias_strength, Sampler):
                bias_strength = Uniform(0, bias_strength)
            bias = RandomMulFieldTransform(
                bias, vmax=bias_strength, symmetric=1, order=order)
            steps += [bias]

        if gamma:
            if not isinstance(gamma, Sampler):
                gamma = LogNormal(0, gamma)
            gamma = RandomGammaTransform(gamma)
            steps += [gamma]

        if motion_fwhm:
            if not isinstance(motion_fwhm, Sampler):
                motion_fwhm = Uniform(0, motion_fwhm)
            smooth = RandomSmoothTransform(motion_fwhm)
            steps += [smooth]

        if snr:
            noise_sd = 1 / snr
            if not isinstance(noise_sd, Sampler):
                noise_sd = Uniform(0, noise_sd)
            noise1 = RandomChiNoiseTransform(noise_sd)
            if gfactor:
                if not isinstance(gfactor, Sampler):
                    gfactor = RandInt(2, gfactor)
                noise = RandomizedTransform(
                    GFactorTransform,
                    dict(noise=noise1, shape=gfactor, order=order),
                )
            else:
                noise = noise1
        else:
            noise = None

        if resolution:
            if not isinstance(resolution, Sampler):
                resolution = Uniform(1, resolution)
            lowres2d = RandomLowResSliceTransform(resolution, noise=noise)
            lowres3d = RandomLowResTransform(resolution, noise=noise)
            lowres = SwitchTransform([lowres2d, lowres3d])
            steps += [lowres]
        elif snr:
            steps += [noise]

        # Quantile transform
        # Maps intensity percentiles (pmin, pmax) to intensities (vmin, vmax).
        # If `clamp`, then clip values inside (vmin, vmax).
        # Default: pmin=0.01, pmax=0.99, vmin=0, vmax=1, clamp=False
        steps += [QuantileTransform()]
        super().__init__(steps, **kwargs)


# typing
_LabelGrouping = tx.Union[
    tx.Tuple[int, ...],
    tx.Tuple[tx.Tuple[int, ...], ...]
]
_GeomStr = tx.Literal['affine', 'elastic', 'affine+elastic', '']
_Geom = tx.Union[_GeomStr, bool]


class ApplySynthFromLabelTransform(FinalTransform):
    """
    Apply the sequence of tranformations that synthesizes an image from
    a label map.
    """

    def __init__(
        self,
        gmm: Transform,
        deform: tx.Optional[Transform] = None,
        intensity: tx.Optional[Transform] = None,
        load: tx.Optional[Transform] = None,
        preproc: tx.Optional[Transform] = None,
        postproc: tx.Optional[Transform] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        gmm : Transform
            The Gaussian mixture model (GMM) transform that generates
            an image from a label map.
        deform : Transform | None
            The deformation transform that deforms the label map before
            generation.
        intensity : Transform | None
            The intensity transform that augments the generated image.
        load : Transform | None
            The transform that loads the input label map from disk.
        preproc : Transform | None
            The transform that preprocesses the labels.
            Only labels returned by `preproc` are used for generation.
        postproc : Transform | None
            The transform that postprocesses the labels.
            Only labels returned by `postproc` are used as targets.

        Other Parameters
        ----------------
        returns : [(list | dict) of] {'input', 'deformed', 'generators', 'target', 'label', 'image', 'output'}
            Tensors to return

            - `'input'`: the input label map (after loading, if `load` is provided)
            - `'deformed'`: the deformed label map (after `deform`, if provided)
            - `'generators'`: the generator maps (after `preproc`, if provided)
            - `'target'`, `'label'`: the target label map (after `postproc`, if provided)
            - `'image'`, `'output'`: the generated image (after `intensity`, if provided)

        """  # noqa: E501
        super().__init__(**kwargs)
        self.load = load or IdentityTransform()
        self.preproc = preproc or IdentityTransform()
        self.postproc = postproc or IdentityTransform()
        self.deform = deform or IdentityTransform()
        self.intensity = intensity or IdentityTransform()
        self.gmm = gmm or IdentityTransform()

    @property
    def is_final(self) -> bool:
        return (
            self.load.is_final and
            self.preproc.is_final and
            self.postproc.is_final and
            self.deform.is_final and
            self.intensity.is_final and
            self.gmm.is_final
        )

    def _unroll(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0 or self.is_final:
            return self

        layers = dict(
            load=self.load,
            preproc=self.preproc,
            postproc=self.postproc,
            deform=self.deform,
            intensity=self.intensity,
            gmm=self.gmm,
        )

        while not layers['load'].is_final and max_depth > 0:
            layers['load'] = layers['load'].unroll(x, max_depth)
            max_depth -= 1
        if max_depth == 0:
            return type(self)(**layers, **self.get_prm())

        lab = layers['load'](x)
        while not layers['deform'].is_final and max_depth > 0:
            layers['deform'] = layers['deform'].unroll(lab, max_depth)
            max_depth -= 1
        if max_depth == 0:
            return type(self)(**layers, **self.get_prm())

        dfm = layers['deform'](lab)
        while not layers['preproc'].is_final and max_depth > 0:
            layers['preproc'] = layers['preproc'].unroll(dfm, max_depth)
            max_depth -= 1
        if max_depth == 0:
            return type(self)(**layers, **self.get_prm())

        gen = layers['preproc'](dfm)
        while not layers['gmm'].is_final and max_depth > 0:
            layers['gmm'] = layers['gmm'].unroll(gen, max_depth)
            max_depth -= 1
        if max_depth == 0:
            return type(self)(**layers, **self.get_prm())

        img = layers['gmm'](gen)
        while not layers['intensity'].is_final and max_depth > 0:
            layers['intensity'] = layers['intensity'].unroll(img, max_depth)
            max_depth -= 1
        if max_depth == 0:
            return type(self)(**layers, **self.get_prm())

        while not layers['postproc'].is_final and max_depth > 0:
            layers['postproc'] = layers['postproc'].unroll(dfm, max_depth)
            max_depth -= 1
        if max_depth == 0:
            return type(self)(**layers, **self.get_prm())

        return type(self)(**layers, **self.get_prm()).unroll(x, max_depth-1)

    def _xform(self, lab: Tensor) -> Returned:
        lab = self.load(lab)
        dfm = self.deform(lab)
        gen = self.preproc(dfm)
        img = self.gmm(gen)
        img = self.intensity(img)
        tgt = self.postproc(dfm)
        return prepare_output(
            dict(input=lab, deformed=dfm, generators=gen,
                    target=tgt, label=tgt, image=img, output=img),
            self.returns,
        )


class SynthFromLabelTransform(NonFinalTransform):
    """
    Synthesize an MRI from an existing label map

    !!! example
        ```python
        # if inputs are preloaded label tensors (default)
        synth = SynthFromLabelTransform()

        # if inputs are filenames
        synth = SynthFromLabelTransform(from_disk=True)

        # memory-efficient patch-synthesis
        synth = SynthFromLabelTransform(patch=64)

        img, lab = synth(input)
        ```

    ??? reference
        Billot, B., Greve, D.N., Puonti, O., Thielscher, A., Van Leemput, K.,
        Fischl, B., Dalca, A.V. and Iglesias, J.E., 2023.
        [**SynthSeg: Segmentation of brain MRI scans of any contrast and resolution
        without retraining.**](https://www.sciencedirect.com/science/article/pii/S1361841523000506)
        _Medical image analysis_, 86, p.102789.

            @article{billot2023synthseg,
              title     = {SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining},
              author    = {Billot, Benjamin and Greve, Douglas N and Puonti, Oula and Thielscher, Axel and Van Leemput, Koen and Fischl, Bruce and Dalca, Adrian V and Iglesias, Juan Eugenio and others},
              journal   = {Medical image analysis},
              volume    = {86},
              pages     = {102789},
              year      = {2023},
              publisher = {Elsevier},
              url       = {https://www.sciencedirect.com/science/article/pii/S1361841523000506}
            }

    """  # noqa: E501

    Final = Next = ApplySynthFromLabelTransform
    """The transform type returned by `unroll`, `next` and `final`."""

    def __init__(
        self,
        *,
        patch: tx.Optional[cct.NumberOrSequence[int]] = None,
        from_disk: bool = False,
        one_hot: bool = False,
        synth_labels: tx.Optional[_LabelGrouping] = None,
        synth_labels_maybe: tx.Mapping[_LabelGrouping, float] = None,
        target_labels: tx.Optional[_LabelGrouping] = None,
        order: cct.SamplerOrBoundOrBool[int] = 3,
        geom: _Geom = True,
        translations: cct.SamplerOrBoundOrBool[float] = 0.1,
        rotation: cct.SamplerOrBoundOrBool[float] = 15,
        shears: cct.SamplerOrBoundOrBool[float] = 0.012,
        zooms: cct.SamplerOrBoundOrBool[float] = 0.15,
        elastic: cct.SamplerOrBoundOrBool[float] = 0.05,
        elastic_nodes: cct.SamplerOrBound[int] = 10,
        elastic_steps: cct.SamplerOrBound[int] = 0,
        bound: cct.TorchBound = 'border',
        gmm_fwhm: cct.SamplerOrBoundOrBool[float] = 10,
        bias: cct.SamplerOrBoundOrBool[int] = 7,
        bias_strength: cct.SamplerOrBound[float] = 0.5,
        gamma: cct.SamplerOrBoundOrBool[float] = 0.6,
        motion_fwhm: cct.SamplerOrBoundOrBool[float] = 3,
        resolution: cct.SamplerOrBoundOrBool[float] = 8,
        snr: cct.SamplerOrBoundOrBool[float] = 10,
        gfactor: cct.SamplerOrBoundOrBool[int] = 5,
        sample_in_background: bool = False,
        dtype: tx.Optional[torch.dtype] = None,
        device: tx.Optional[cct.TorchDevice] = None,
        returns: tx.Union[cct.ReturnsT] = Kwargs(image='image', label='label')
    ) -> None:
        """

        Parameters
        ----------
        patch : [list of] int
            If provided, patches of this size are extracted. Note that
            patches are extracted *after* application of the geometric
            transforms (altough both operations are combined efficiently).
        from_disk : bool
            Assume inputs are filenames and load from disk
        one_hot : bool, default=False
            Return one-hot labels. Else return a label map.
        synth_labels : tuple of [tuple of] int
            List of labels to use for synthesis.
            If multiple labels are grouped in a sublist, they share the
            same intensity in the GMM. All labels not listed are assumed
            background. For example, this option can be used to ensure
            that symmetric structures share the same intensity.
        synth_labels_maybe : dict[tuple of [tuple of] int, float]
            List of labels to sometimes use for synthesis, and their
            probability of being sampled. This options allow groups parts
            of the anatomy to be hidden in a random subset of images. This
            can be used to e.g. model the presence of skull-stripped images.
        target_labels : tuple of [tuple of] int
            List of target labels.
            If multiple labels are grouped in a sublist, they are fused.
            All labels not listed are assumed background.
            The final label map is relabeled in the order provided,
            starting from 1 (background is 0).
            This option can be used to predict a coarser set of labels,
            than those used for synthesis.
        order : int
            Spline order of the elastic and bias fields (1 is much faster)

        Other Parameters
        ----------------
        geom : {'affine', 'elastic', 'affine+elastic', ''} | bool
            Which geometric transforms to apply.
            `True` (default) corresponds to `affine+elastic`.
            `False` corresponds to no geometric transform.

            !!! addedin "![v0.5](https://img.shields.io/badge/v0.5-green) \
                Added `geom` argument."
        translations : Sampler | float | {False}
            Distribution from which random translations (in percentage of
            field-of-view) are sampled.
            If a `float`, sample from `Uniform(-value, value)`.
        rotation : Sampler | float | {False}
            Distribution from which random rotations (in degree) are sampled.
            If a `float`, sample from `Uniform(-value, value)`.
        shears : Sampler | float | {False}
            Distribution from which random shears are sampled.
            If a `float`, sample from `Uniform(-value, value)`.
        zooms : Sampler | float | {False}
            Distribution from which random zooms (about one) are sampled.
            If a `float`, sample from `Uniform(-value, value)`.
            The zoom effectively applied is 1 plus the sampled value
            (i.e., the zoom is sampled from `Uniform(1-value, 1+value)`).
        elastic : Sampler | float | {False}
            Distribution from which the maximum of the displacement magnitude
            (in proportion of the field-of-view) is sampled.
            If a `float`, sample from `Uniform(0, value)`.
        elastic_nodes : Sampler | int
            The sampled value controls the smoothness of the displacement
            field (smaller values yield smoother fields).
            If a `float`, sample from `RandInt(2, value)`.
        elastic_steps : Sampler | int
            Number of scaling-and-squaring integration steps.
            Scaling-and-squaring ensure that the elastic field is
            diffeomorphic (one-to-one and onto).
            If 0, the field is not integrated, which is faster but may
            result in image foldings.
        bound : {'zeros', 'border', 'reflection'}
            Padding mode when sampling outside the field-of-view.

        Other Parameters
        ----------------
        gmm_fwhm : Sampler | float | {False}
            In contrast with the SynthSeg paper, we perform an
            edge-preserving smoothing after intensities are sampled, in
            order to mimic texture. This parameter controls the width
            of the smoothing kernel.
            If a `float`, sample from `Uniform(0, value)`.
        bias : Sampler | int | {False}
            The sampled value controls the smoothness of the intensity
            bias field (smaller values yield smoother fields).
            If a `float`, sample from `RandInt(2, value)`.
        bias_strength : Sampler | (0..1)
            The maximum magnitude of the bias field (about 1).
            If a `float`, sample from `Uniform(0, value)`.
            The minimum and maximum values of the bias field will be
            `1 - bias_strength` and `1 + bias_strength`.
        gamma : Sampler | float | {False}
            The Gamma transform squeezes intensities such that the contrast
            to noise ratio is decreased (positive values lead to less
            decreased contrast, positive values lead to increased contrast).
            If a `float`, sample the gamma exponent from `LogNormal(0, value)`.
        motion_fwhm : Sampler | float | {False}
            A blur can be perform to model the point spread function or
            motion-related smearing. The amount of smoothing is encoded by
            the full-width at half-maximum (FWHM) of the underlying
            Gaussian kernel.
            If a `float`, sample the FWHM from `Uniform(0, value)`.
        resolution : Sampler | float | {False}
            Thick-slice or isotropic low-resolution (LR) images are randomly
            applied. and their (through-slice or iso) resolution is
            controlled here. It is defined as a proportion of the
            high-resolution voxel size (i.e., a resolution of `4` mean
            that the LR voxel size will be four times as large as the
            input voxel size)
            If a `float`, sampled form `Uniform(0, value)`.
        snr : Sampler | float | {False}
            The amount of noise added is encoded by the signal-to-noise ratio
            (SNR) of the noisy image (larger sampled values yield less
            noisy images).
            If a `float`, the value is a lower bound for SNR (no image
            will have a poorer SNR than this). The noise variance is
            then sampled from `Uniform(0, 1/snr)`.
        gfactor : Sampler | int | {False}
            The g-factor is a smooth field that locally scales the noise
            variance. The sampled value controls the smoothness of
            the g-factor field.
            If a `float`, sample from `RandInt(2, value)`.
        sample_in_background : bool
            If True, sample a Gaussian in the background class.
            Otherwise, keep it zeros.

        Other Parameters
        ----------------
        returns : [(list | dict) of] {'input', 'deformed', 'generators', 'target', 'label', 'image', 'output'}
            Tensors to return

            - `'input'`: the input label map (after loading, if `load` is provided)
            - `'deformed'`: the deformed label map (after `deform`, if provided)
            - `'generators'`: the generator maps (after `preproc`, if provided)
            - `'target'`, `'label'`: the target label map (after `postproc`, if provided)
            - `'image'`, `'output'`: the generated image (after `intensity`, if provided)
        """  # noqa: E501
        super().__init__(shared=False, returns=returns)
        self.load = (
            LoadTransform(dtype='long') if from_disk else IdentityTransform()
        )
        self.synth_labels = synth_labels
        self.synth_labels_maybe = synth_labels_maybe
        if one_hot:
            postproc = OneHotTransform()
            if target_labels:
                postproc = RelabelTransform(target_labels) + postproc
        elif target_labels:
            postproc = RelabelTransform(target_labels)
        else:
            postproc = IdentityTransform()
        self.postproc_labels = postproc
        if not geom:
            elastic = translations = rotation = shears = zooms = 0
        elif geom == 'affine':
            elastic = 0
        elif geom == 'elastic':
            translations = rotation = shears = zooms = 0
        self.deform = RandomAffineElasticTransform(
            0 if elastic in (None, False) else elastic or 0,
            elastic_nodes,
            order=order,
            translations=translations or 0,
            rotations=rotation or 0,
            shears=shears or 0,
            zooms=zooms or 0,
            patch=patch,
            steps=elastic_steps,
            bound=bound,
            dtype=dtype,
            device=device,
            nearest_if_label=True,
        )
        self.gmm = RandomGaussianMixtureTransform(
            fwhm=gmm_fwhm or 0,
            background=None if sample_in_background else 0,
        )
        self.intensity = IntensityTransform(
            bias=bias,
            bias_strength=bias_strength,
            gamma=gamma,
            motion_fwhm=motion_fwhm,
            resolution=resolution,
            snr=snr,
            gfactor=gfactor,
            order=order,
        )

    def _unroll(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self

        # sample labels to use for synthesis
        synth_labels = list(self.synth_labels or [])
        if self.synth_labels_maybe:
            for labels, prob in self.synth_labels_maybe.items():
                if pyrandom.random() > (1 - prob):
                    if isinstance(labels, Number):
                        synth_labels += [labels]
                    else:
                        synth_labels += list(labels)
        if synth_labels:
            preproc = RelabelTransform(synth_labels)
        else:
            preproc = IdentityTransform()

        return self.Next(
            self.gmm,
            self.deform,
            self.intensity,
            self.load,
            preproc,
            self.postproc_labels,
            **self.get_prm(),
        ).unroll(x, max_depth-1)
