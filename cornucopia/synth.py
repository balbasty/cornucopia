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
    In _Medical Image Computing and Computer Assisted Intervention–MICCAI 2020:
    23rd International Conference_, Lima, Peru, October 4–8, 2020,
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

"""
__all__ = ['SynthFromLabelTransform', 'IntensityTransform']

from .base import SequentialTransform, RandomizedTransform, SwitchTransform, Transform
from .labels import RandomGaussianMixtureTransform, RelabelTransform, OneHotTransform
from .intensity import MultFieldTransform, GammaTransform, QuantileTransform
from .psf import SmoothTransform, LowResSliceTransform, LowResTransform
from .noise import ChiNoiseTransform, GFactorTransform
from .geometric import RandomAffineElasticTransform
from .random import Uniform, RandInt, Fixed
from .io import LoadTransform
import random as pyrandom


class IntensityTransform(SequentialTransform):
    """Common intensity augmentation for MRI and related images

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
    """

    def __init__(self,
                 bias=7,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 snr=10,
                 gfactor=5,
                 order=3):
        """
        Parameters
        ----------
        bias : int
            Upper bound for the number of control points of the bias field
        gamma : float
            Upper bound for the exponent of the Gamma transform
        motion_fwhm : float
            Upper bound of the FWHM of the global (PSF/motion) smoothing kernel
        resolution : float
            Upper bound for the inter-slice spacing (in voxels)
        snr : float
            Lower bound for the signal-to-noise ratio
        gfactor : int
            Upper bound for the number of control points of the g-factor map
        order : int
            Spline order of the bias field (1 is much faster)
        """
        noise_sd = 255 / snr

        bias = RandomizedTransform(MultFieldTransform, RandInt(2, bias), dict(order=order))
        gamma = RandomizedTransform(GammaTransform, Uniform(0, gamma))
        smooth = RandomizedTransform(SmoothTransform, Uniform(0, motion_fwhm))
        noise1 = RandomizedTransform(ChiNoiseTransform, Uniform(0, noise_sd))
        noise = RandomizedTransform(GFactorTransform, [Fixed(noise1),
                                                       RandInt(2, gfactor)])
        lowres2d = RandomizedTransform(LowResSliceTransform,
                                       dict(resolution=Uniform(1, resolution),
                                            noise=Fixed(noise)))
        lowres3d = RandomizedTransform(LowResTransform,
                                       dict(resolution=Uniform(1, resolution),
                                            noise=Fixed(noise)))
        lowres = SwitchTransform([lowres2d, lowres3d])
        rescale = QuantileTransform()

        super().__init__([bias, gamma, smooth, lowres, rescale])


class SynthFromLabelTransform(Transform):
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
    """

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
                 steps=0,
                 gmm_fwhm=10,
                 bias=7,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 snr=10,
                 gfactor=5,
                 order=3):
        """

        Parameters
        ----------
        patch : [list of] int
            Shape of the patches to extact
        from_disk : bool
            Assume inputs are filenames and load from disk
        one_hot : bool, default=False
            Return one-hot labels. Else return a label map.
        synth_labels : tuple of [tuple of] int
            List of labels to use for synthesis.
            If multiple labels are grouped in a sublist, they share the
            same intensity in the GMM. All labels not listed are assumed
            background.
        synth_labels_maybe : dict[tuple of [tuple of] int, float]
            List of labels to sometimes use for synthesis, and their
            probability of being sampled.
        target_labels : tuple of [tuple of] int
            List of target labels.
            If multiple labels are grouped in a sublist, they are fused.
            All labels not listed are assumed background.
            The final label map is relabeled in the order provided,
            starting from 1 (background is 0).
        order : int
            Spline order of the elastic and bias fields (1 is much faster)

        Other Parameters
        ----------------
        rotation : float
            Upper bound for rotations, in degree.
        shears : float
            Upper bound for shears
        zooms : float
            Upper bound for zooms (about one)
        elastic : float
            Upper bound for elastic displacements, in percent of the FOV.
        elastic_nodes : int
            Upper bound for number of control points in the elastic field.
        steps : int
            Number of scaling-and-squaring integration steps

        Other Parameters
        ----------------
        gmm_fwhm : float
            Upper bound for the FWHM of the intra-tissue smoothing kernel
        bias : int
            Upper bound for the number of control points of the bias field
        gamma : float
            Upper bound for the exponent of the Gamma transform
        motion_fwhm : float
            Upper bound of the FWHM of the global (PSF/motion) smoothing kernel
        resolution : float
            Upper bound for the inter-slice spacing (in voxels)
        snr : float
            Lower bound for the signal-to-noise ratio
        gfactor : int
            Upper bound for the number of control points of the g-factor map
        """
        super().__init__(shared=False)
        self.load = LoadTransform(dtype='long') if from_disk else None
        self.synth_labels = synth_labels
        self.synth_labels_maybe = synth_labels_maybe
        if one_hot:
            postproc = OneHotTransform()
            if target_labels:
                postproc = RelabelTransform(target_labels) + postproc
        elif target_labels:
            postproc = RelabelTransform(target_labels)
        else:
            postproc = None
        self.postproc_labels = postproc
        self.deform = RandomAffineElasticTransform(
            elastic, elastic_nodes, order=order, steps=steps,
            rotations=rotation, shears=shears, zooms=zooms, patch=patch)
        self.gmm = RandomGaussianMixtureTransform(fwhm=gmm_fwhm, background=0)
        self.intensity = IntensityTransform(
            bias, gamma, motion_fwhm, resolution, snr, gfactor, order)

    def get_parameters(self, x):
        parameters = dict()
        synth_labels = list(self.synth_labels or [])
        if self.synth_labels_maybe:
            for labels, prob in self.synth_labels_maybe.items():
                if pyrandom.random() > (1 - prob):
                    synth_labels += list(labels)
        if synth_labels:
            parameters['preproc'] = RelabelTransform(synth_labels)
        parameters['gmm'] = self.gmm.get_parameters(x)
        parameters['deform'] = self.deform.get_parameters(x)
        return parameters

    def apply_transform(self, lab, parameters=None):
        parameters = parameters or {}
        load = self.load or (lambda x: x)
        preproc = parameters.get('preproc', lambda x: x)
        gmm = parameters.get('gmm', lambda x: x)
        deform = parameters.get('deform', lambda x: x)
        postproc = self.postproc_labels or (lambda x: x)

        lab = load(lab)
        lab = deform(lab)
        img = gmm(preproc(lab))
        img = self.intensity(img)
        lab = postproc(lab)
        return img, lab


