"""Flexible transforms for pre-processing and augmentation

Example on how to use this machinery to generate within-subject
image pairs with a random affine deformation between them::

    # It's easy to randomize deterministic parameters of a transform
    rand_chi = trf.RandomizedTransform(
    trf.ChiNoiseTransform, dict(sigma=trf.Uniform(0.01, 0.3)))

    # SequentialTransform is equivalent to monai's Compose
    # MappedTransform make sit possible to process only some inputs (the
    # others are kept untouched)
    demo_transformer = trf.SequentialTransform([
        trf.MappedTransform(img=trf.ToTensorTransform(3, dtype=torch.float32),
                            seg=trf.ToTensorTransform(3, dtype=torch.int64)),
        trf.MappedTransform(seg=trf.OneHotTransform([1])),
        trf.MappedTransform(img=trf.QuantileTransform()),
        trf.MakeAffinePair(trf.RandomAffineTransform()),
        trf.MappedTransform(img=trf.MultFieldTransform()),
        trf.MappedTransform(img=rand_chi),
    ])

    # Apply the workflow
    img_and_seg, flow_and_mat = demo_transformer(dict(img=dat, seg=lab))

    # Unpile everything
    img1, img2 = img_and_seg['img']
    seg1, seg2 = img_and_seg['seg']
    flow = flow_and_mat['flow'

"""

# TODO:
#   [x] Make it a standalone package?
#   [x] Move samplers in their own file
#   [x] Add IO transforms (that transform filenames in tensors)
#   [ ] Better deal with separable/shared transforms
#       [ ] Add a SharedTransform class (like Randomized) that does the heavy
#           lifting
#       [ ] By default (non shared), let Transforms handle multi-channel
#           data (currently we loop across channels in the base class)

from . import random                        # noqa: F401
from . import ctx                           # noqa: F401
from .ctx import batch                      # noqa: F401
from .base import *                         # noqa: F401,F403
from .special import *                      # noqa: F401,F403
from .contrast import *                     # noqa: F401,F403
from .geometric import *                    # noqa: F401,F403
from .intensity import *                    # noqa: F401,F403
from .io import *                           # noqa: F401,F403
from .fov import *                          # noqa: F401,F403
from .kspace import *                       # noqa: F401,F403
from .labels import *                       # noqa: F401,F403
from .noise import *                        # noqa: F401,F403
from .psf import *                          # noqa: F401,F403
from .qmri import *                         # noqa: F401,F403
from .synth import *                        # noqa: F401,F403
from .utils.patch import patch_apply        # noqa: F401,F403

from . import _version
__version__ = _version.get_versions()['version']
