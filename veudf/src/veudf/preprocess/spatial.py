"""Conservative triangle-box queries shared by edge and vertex surface proposals."""

from dataclasses import dataclass
import numpy as np


@dataclass
class BoxNode:
    lo: np.ndarray
    hi: np.ndarray
    children: tuple = ()
    ids: object = None


class TriangleBoxes:
    def __init__(self, vertices, faces):
        tri = vertices[faces]
        self.lo, self.hi = tri.min(1), tri.max(1)
        self.centers = (self.lo + self.hi) * 0.5
        self.root = self._build(np.arange(len(faces)))

    def _build(self, ids):
        lo, hi = self.lo[ids].min(0), self.hi[ids].max(0)
        if len(ids) <= 64:
            return BoxNode(lo, hi, ids=ids)
        axis = np.ptp(self.centers[ids], axis=0).argmax()
        order = ids[np.argsort(self.centers[ids, axis], kind="stable")]
        middle = len(order) // 2
        return BoxNode(
            lo, hi, (self._build(order[:middle]), self._build(order[middle:]))
        )

    def query(self, lo, hi):
        found, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if np.any(node.lo > hi) or np.any(node.hi < lo):
                continue
            if node.children:
                stack.extend(node.children)
            else:
                ids = node.ids
                keep = (self.lo[ids] <= hi).all(1) & (self.hi[ids] >= lo).all(1)
                found.append(ids[keep])
        return np.sort(np.concatenate(found)) if found else np.empty(0, np.int64)
