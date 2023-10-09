import copy
import numpy as np


class GeneArray:
    def __init__(self, sizes: dict[str, int]) -> None:
        self.sizes = sizes
        self.arr = np.zeros(sum(sizes.values()), dtype=np.float16)
        self.keys = sorted(list(sizes.keys()))
        self.key2i = {key: i for (i, key) in enumerate(self.keys)}
        self.ids = [0] + list(np.cumsum([sizes[key] for key in self.keys]))
        self.is_set = {key: False for key in self.keys}

    def __getitem__(self, key: str) -> np.ndarray:
        assert key in self.sizes
        assert self.is_set[key]
        i = self.key2i[key]
        value = self.arr[self.ids[i]: self.ids[i + 1]]
        if len(value) == 1:
            return value[0]
        else:
            return value

    def __setitem__(self, key: str, value: np.ndarray) -> np.ndarray:
        i = self.key2i[key]
        # Check length
        l = self.ids[i + 1] - self.ids[i]
        if l == 1:
            assert np.array(value).shape in [(), (1,)]
        else:
            assert np.array(value).shape == (l,)
        # Set
        self.arr[self.ids[i]: self.ids[i + 1]] = value
        self.is_set[key] = True
        return self.arr[self.ids[i]: self.ids[i + 1]]

    def asarray(self):
        assert all(self.is_set.values())
        return self.arr.copy()

    def set_array(self, arr):
        self.arr = arr
        self.is_set = {key: True for key in self.is_set}
        return self

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.arr)
