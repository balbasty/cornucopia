"""SynthShape generator

References
----------
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
__all__ = ['SynthFromLabelTransform', 'IntensityTransform']

from .baseutils import Kwargs, prepare_output
from .base import FinalTransform, NonFinalTransform
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
from numbers import Number
import random as pyrandom


class IntensityTransform(SequentialTransform):
    """Common intensity augmentation for MRI and related images

    The arguments control the *range* of the distributions from which
    the transform parameters are sampled.

    It is also possible to directly provide the probability distribution
    from which to sample the parametes. In this case, it **must** be a
    `cc.random.Sampler` instance.

    Setting any argument to `False` disables the corresponding transform
    entirely.

    !!! reference
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

    def __init__(self,
                 bias=7,
                 bias_strength=0.5,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 snr=10,
                 gfactor=5,
                 order=3,
                 **kwargs):
        """
        Parameters
        ----------
        bias : int or Sampler or False
            The sampled value controls the smoothness of the intensity
            bias field (smaller values yield smoother fields).
            If a `float`, sample from `RandInt(2, value)`.
        bias_strength : (0..1) or Sampler
            The maximum magnitude of the bias field (about 1).
            If a `float`, sample from `Uniform(0, value)`.
            The minimum and maximum values of the bias field will be
            `1 - bias_strength` and `1 + bias_strength`.
        gamma : float or Sampler or False
            The Gamma transform squeezes intensities such that the contrast
            to noise ratio is decreased (positive values lead to less
            decreased contrast, positive values lead to increased contrast).
            If a `float`, sample the gamma exponent from `LogNormal(0, value)`.
        motion_fwhm : float or Sampler or False
            A blur can be perform to model the point spread function or
            motion-related smearing. The amount of smoothing is encoded by
            the full-width at half-maximum (FWHM) of the underlying
            Gaussian kernel.
            If a `float`, sample the FWHM from `Uniform(0, value)`.
        resolution : float or Sampler or False
            Thick-slice or isotropic low-resolution (LR) images are randomly
            applied. and their (through-slice or iso) resolution is
            controlled here. It is defined as a proportion of the
            high-resolution voxel size (i.e., a resolution of `4` mean
            that the LR voxel size will be four times as large as the
            input voxel size)
            If a `float`, sampled form `Uniform(0, value)`.
        snr : float or Sampler or False
            The amount of noise added is encoded by the signal-to-noise ratio
            (SNR) of the noisy image (larger sampled values yield less
            noisy images).
            If a `float`, the value is a lower bound for SNR (no image
            will have a poorer SNR than this). The noise variance is
            then sampled from `Uniform(0, 1/snr)`.
        gfactor : int or Sampler or False
            The g-factor is a smooth field that locally scales the noise
            variance. The `gfactor` argument controls the smoothness of
            the g-factor field.
            If a `float`, sample from `RandInt(2, value)`.
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
                    dict(noise=noise1, shape=gfactor),
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

    !!! reference
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

    class Final(FinalTransform):

        def __init__(self, gmm, deform=None, intensity=None,
                     load=None, preproc=None, postproc=None, **kwargs):
            super().__init__(**kwargs)
            self.load = load or IdentityTransform()
            self.preproc = preproc or IdentityTransform()
            self.postproc = postproc or IdentityTransform()
            self.deform = deform or IdentityTransform()
            self.intensity = intensity or IdentityTransform()
            self.gmm = gmm or IdentityTransform()

        @property
        def is_final(self):
            return (
                self.load.is_final and
                self.preproc.is_final and
                self.postproc.is_final and
                self.deform.is_final and
                self.intensity.is_final and
                self.gmm.is_final
            )

        def make_final(self, x, max_depth=float('inf')):
            if max_depth == 0 or self.is_final:
                return self
            return type(self)(
                self.gmm.make_final(x, 1),
                self.deform.make_final(x, 1),
                self.intensity.make_final(x, 1),
                self.load.make_final(x, 1),
                self.preproc.make_final(x, 1),
                self.postproc.make_final(x, 1),
                **self.get_prm(),
            ).make_final(x, max_depth-1)

        def apply(self, lab):
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

    def __init__(self,
                 patch=None,
                 from_disk=False,
                 one_hot=False,
                 synth_labels=None,
                 synth_labels_maybe=None,
                 target_labels=None,
                 rotation=15,
                 shears=0.012,
                 zooms=0.15,
                 elastic=0.05,
                 elastic_nodes=10,
                 elastic_steps=0,
                 gmm_fwhm=10,
                 bias=7,
                 bias_strength=0.5,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 snr=10,
                 gfactor=5,
                 order=3,
                 returns=Kwargs(image='image', label='label')):
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
            background. For example, this option can be ued to ensure
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
        rotation : float or Sampler or False
            Distribution from which random rotations (in degree) are sampled.
            If a `float`, sample from `Uniform(-value, value)`.
        shears : float or Sampler or False
            Distribution from which random shears are sampled.
            If a `float`, sample from `Uniform(-value, value)`.
        zooms : float or Sampler or False
            Distribution from which random zooms (about one) are sampled.
            If a `float`, sample from `Uniform(-value, value)`.
            The zoom effectively applied is 1 plus the sampled value
            (i.e., the zoom is sampled from `Uniform(1-value, 1+value)`).
        elastic : float or Sampler or False
            Distribution from which the maximum of the displacement magnitude
            (in proportion of the field-of-view) is sampled.
            If a `float`, sample from `Uniform(0, value)`.
        elastic_nodes : int or Sampler
            The sampled value controls the smoothness of the displacement
            field (smaller values yield smoother fields).
            If a `float`, sample from `RandInt(2, value)`.
        elastic_steps : int or Sampler
            Number of scaling-and-squaring integration steps.
            Scaling-and-squaring ensure that the elastic field is
            diffeomorphic (one-to-one and onto).
            If 0, the field is not integrated, which is faster but may
            result in image foldings.

        Other Parameters
        ----------------
        gmm_fwhm : float or Sampler or False
            In contrast with the SynthSeg paper, we perform an
            edge-preserving smoothing after intensities are sampled, in
            order to mimic texture. This parameter controls the width
            of the smoothing kernel.
            If a `float`, sample from `Uniform(0, value)`.
        bias : int or Sampler or False
            The sampled value controls the smoothness of the intensity
            bias field (smaller values yield smoother fields).
            If a `float`, sample from `RandInt(2, value)`.
        bias_strength : (0..1) or Sampler
            The maximum magnitude of the bias field (about 1).
            If a `float`, sample from `Uniform(0, value)`.
            The minimum and maximum values of the bias field will be
            `1 - bias_strength` and `1 + bias_strength`.
        gamma : float or Sampler or False
            The Gamma transform squeezes intensities such that the contrast
            to noise ratio is decreased (positive values lead to less
            decreased contrast, positive values lead to increased contrast).
            If a `float`, sample the gamma exponent from `LogNormal(0, value)`.
        motion_fwhm : float or Sampler or False
            A blur can be perform to model the point spread function or
            motion-related smearing. The amount of smoothing is encoded by
            the full-width at half-maximum (FWHM) of the underlying
            Gaussian kernel.
            If a `float`, sample the FWHM from `Uniform(0, value)`.
        resolution : float or Sampler or False
            Thick-slice or isotropic low-resolution (LR) images are randomly
            applied. and their (through-slice or iso) resolution is
            controlled here. It is defined as a proportion of the
            high-resolution voxel size (i.e., a resolution of `4` mean
            that the LR voxel size will be four times as large as the
            input voxel size)
            If a `float`, sampled form `Uniform(0, value)`.
        snr : float or Sampler or False
            The amount of noise added is encoded by the signal-to-noise ratio
            (SNR) of the noisy image (larger sampled values yield less
            noisy images).
            If a `float`, the value is a lower bound for SNR (no image
            will have a poorer SNR than this). The noise variance is
            then sampled from `Uniform(0, 1/snr)`.
        gfactor : int or Sampler or False
            The g-factor is a smooth field that locally scales the noise
            variance. The sampled value controls the smoothness of
            the g-factor field.
            If a `float`, sample from `RandInt(2, value)`.
        """
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
        self.deform = RandomAffineElasticTransform(
            elastic or 0,
            elastic_nodes,
            order=order,
            rotations=rotation or 0,
            shears=shears or 0,
            zooms=zooms or 0,
            patch=patch,
            steps=elastic_steps,
        )
        self.gmm = RandomGaussianMixtureTransform(
            fwhm=gmm_fwhm or 0,
            background=0,
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

    def make_final(self, x, max_depth=float('inf')):
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

        return self.Final(
            self.gmm,
            self.deform,
            self.intensity,
            self.load,
            preproc,
            self.postproc_labels,
            **self.get_prm(),
        ).make_final(x, max_depth-1)
