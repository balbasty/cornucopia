"""SynthShape generator

References
----------
..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
      MRI Scans of any Contrast and Resolution"
      Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
      Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
      2021
      https://arxiv.org/abs/2107.09559
..[2] "Learning image registration without images"
      Malte Hoffmann, Benjamin Billot, Juan Eugenio Iglesias,
      Bruce Fischl, Adrian V. Dalca
      ISBI 2021
      https://arxiv.org/abs/2004.10282
..[3] "Partial Volume Segmentation of Brain MRI Scans of any Resolution
       and Contrast"
      Benjamin Billot, Eleanor D. Robinson, Adrian V. Dalca, Juan Eugenio Iglesias
      MICCAI 2020
      https://arxiv.org/abs/2004.10221
..[4] "A Learning Strategy for Contrast-agnostic MRI Segmentation"
      Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl,
      Juan Eugenio Iglesias*, Adrian V. Dalca*
      MIDL 2020
      https://arxiv.org/abs/2003.01995
"""
__all__ = ['SynthFromLabelTransform', 'IntensityTransform']

from .base import SequentialTransform, RandomizedTransform, SwitchTransform, Transform
from .labels import RandomGaussianMixtureTransform, RelabelTransform, OneHotTransform
from .intensity import MultFieldTransform, GammaTransform
from .psf import SmoothTransform, LowResSliceTransform, LowResTransform
from .noise import ChiNoiseTransform, GFactorTransform
from .geometric import RandomAffineElasticTransform
from .random import Uniform, RandInt, Fixed
from .io import LoadTransform
import random as pyrandom


class IntensityTransform(SequentialTransform):
    """Common intensity augmentation for MRI and related images

    References
    ----------
    ..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
          MRI Scans of any Contrast and Resolution"
          Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
          Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
          2021
          https://arxiv.org/abs/2107.09559
    """

    def __init__(self,
                 bias=7,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 snr=10,
                 gfactor=5):
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
        """
        noise_sd = 255 / snr

        bias = RandomizedTransform(MultFieldTransform, RandInt(2, bias))
        gamma = RandomizedTransform(GammaTransform, Uniform(0, gamma))
        smooth = RandomizedTransform(SmoothTransform, Uniform(0, motion_fwhm))
        noise1 = RandomizedTransform(ChiNoiseTransform, Uniform(0, noise_sd))
        noise = RandomizedTransform(GFactorTransform, [Fixed(noise1),
                                                       RandInt(2, gfactor)])
        lowres2d = RandomizedTransform(LowResSliceTransform,
                                       dict(resolution=Uniform(0, resolution),
                                            noise=Fixed(noise)))
        lowres3d = RandomizedTransform(LowResTransform,
                                       dict(resolution=Uniform(0, resolution),
                                            noise=Fixed(noise)))
        lowres = SwitchTransform([lowres2d, lowres3d])

        super().__init__([bias, gamma, smooth, lowres])


class SynthFromLabelTransform(Transform):
    """
    Synthesize an MRI from an existing label map

    Examples
    --------
    ::

        # if inputs are preloaded label tensors (default)
        synth = SynthFromLabelTransform()

        # if inputs are filenames
        synth = SynthFromLabelTransform(from_disk=True)

        # memory-efficient patch-synthesis
        synth = SynthFromLabelTransform(patch=64)

        img, lab = synth(input)

    References
    ----------
    ..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
          MRI Scans of any Contrast and Resolution"
          Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
          Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
          2021
          https://arxiv.org/abs/2107.09559
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
                 elastic=0.15,
                 elastic_nodes=10,
                 gmm_fwhm=10,
                 bias=7,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 snr=10,
                 gfactor=5):
        """

        Parameters
        ----------
        patch : [list of] int, optional
            Shape of the patches to extact
        from_disk : bool, default=False
            Assume inputs are filenames and load from disk
        one_hot : bool, default=False
            Return one-hot labels. Else return a label map.
        synth_labels : tuple of [tuple of] int, optional
            List of labels to use for synthesis.
            If multiple labels are grouped in a sublist, they share the
            same intensity in the GMM. All labels not listed are assumed
            background.
        synth_labels_maybe : dict(tuple of [tuple of] int -> float), optional
            List of labels to sometimes use for synthesis, and their
            probability of being sampled.
        target_labels : tuple of [tuple of] int, optional
            List of target labels.
            If multiple labels are grouped in a sublist, they are fused.
            All labels not listed are assumed background.
            The final label map is relabeled in the order provided,
            starting from 1 (background is 0).

        Geometric Parameters
        --------------------
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

        Intensity Parameters
        --------------------
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
            elastic, elastic_nodes,
            rotations=rotation, shears=shears, zooms=zooms, patch=patch)
        self.gmm = RandomGaussianMixtureTransform(fwhm=gmm_fwhm, background=0)
        self.intensity = IntensityTransform(
            bias, gamma, motion_fwhm, resolution, snr, gfactor)

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


