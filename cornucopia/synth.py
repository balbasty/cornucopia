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
from .base import SequentialTransform, RandomizedTransform, SwitchTransform
from .labels import RandomGaussianMixtureTransform
from .intensity import MultFieldTransform, GammaTransform
from .psf import SmoothTransform, LowResSliceTransform, LowResTransform
from .noise import ChiNoiseTransform, GFactorTransform
from .random import Uniform, RandInt, Fixed


__all__ = ['SynthTransform']


class SynthTransform(SequentialTransform):
    """Synthesize a contrast and imaging artefacts from a label map"""

    def __init__(self,
                 gmm_fwhm=10,
                 bias=7,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 noise=48,
                 gfactor=5):

        gmm = RandomGaussianMixtureTransform(fwhm=gmm_fwhm)
        bias = RandomizedTransform(MultFieldTransform, RandInt(2, bias))
        gamma = RandomizedTransform(GammaTransform, Uniform(0, gamma))
        smooth = RandomizedTransform(SmoothTransform, Uniform(0, motion_fwhm))
        noise1 = RandomizedTransform(ChiNoiseTransform, Uniform(0, noise))
        noise = RandomizedTransform(GFactorTransform, [Fixed(noise1),
                                                       RandInt(2, gfactor)])
        lowres2d = RandomizedTransform(LowResSliceTransform,
                                       dict(resolution=Uniform(0, resolution),
                                            noise=Fixed(noise)))
        lowres3d = RandomizedTransform(LowResTransform,
                                       dict(resolution=Uniform(0, resolution),
                                            noise=Fixed(noise)))
        lowres = SwitchTransform([lowres2d, lowres3d])

        super().__init__([gmm, bias, gamma, smooth, lowres])
