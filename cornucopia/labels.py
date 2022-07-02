from .base import Transform


__all__ = ['OneHotTransform']


class OneHotTransform(Transform):

    def __init__(self, label_map=None, label_ref=None, keep_background=True,
                 dtype=None):
        """

        Parameters
        ----------
        label_map : list or [list of] int
            Map one-hot classes to [list of] labels or label names
            (!! Should not include the background class !!)
        label_ref : dict[int] -> str
            Map label values to label names
        keep_background : bool
            If True, the first one-hot class is the background class,
            and the one hot tensor sums to one.
        dtype : torch.dtype, Use a different dtype for the one-hot
        """
        super().__init__()
        self.label_map = label_map
        self.label_ref = label_ref
        self.keep_background = keep_background
        self.dtype = dtype

    def get_parameters(self, x):
        def get_key(map, value):
            if isinstance(map, dict):
                for k, v in map:
                    if v == value:
                        return k
            else:
                for k, v in enumerate(map):
                    if v == value:
                        return k
            raise ValueError(f'Cannot find "{value}"')

        label_map = self.label_map
        if label_map is None:
            label_map = x.unique(sorted=True)
            if label_map[0] == 0:
                label_map = label_map[1:]
            return label_map.tolist()
        label_ref = self.label_ref
        if label_ref is not None:
            new_label_map = []
            for label in label_map:
                if isinstance(label, (list, tuple)):
                    label = [get_key(label_ref, l) for l in label]
                else:
                    label = get_key(label_ref, label)
                new_label_map.append(label)
            return new_label_map
        return label_map

    def transform_with_parameters(self, x, parameters):
        if len(x) != 1:
            raise ValueError('Cannot one-hot multi-channel tensors')
        x = x[0]

        lmax = len(parameters) + self.keep_background
        y = x.new_zeros([lmax, *x.shape], dtype=self.dtype)

        for new_l, old_l in enumerate(parameters):
            new_l += self.keep_background
            if isinstance(old_l, (list, tuple)):
                for old_l1 in old_l:
                    y[new_l, x == old_l1] = 1
            else:
                y[new_l, x == old_l] = 1

        if self.keep_background:
            y[0] = 1 - y[1:].sum(0)

        return y
