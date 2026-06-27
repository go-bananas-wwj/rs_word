from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from rs_words.data_engine.patch_bank import Patch
from rs_words.glyph import Stroke


class RiverMatcher:
    def __init__(
        self,
        chamfer_weight: float = 1.0,
        hu_weight: float = 0.3,
        direction_weight: float = 0.2,
    ):
        self.chamfer_weight = chamfer_weight
        self.hu_weight = hu_weight
        self.direction_weight = direction_weight

    def _edges(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, 50, 150)

    def _chamfer(self, stroke_mask: np.ndarray, patch: Patch) -> float:
        stroke_edges = cv2.Canny((stroke_mask > 0).astype(np.uint8) * 255, 50, 150)
        patch_edges = self._edges(patch.image)
        if patch_edges.sum() == 0:
            return 1e6
        dt = cv2.distanceTransform(255 - patch_edges, cv2.DIST_L2, 5).astype(np.float32)
        pts = stroke_edges > 0
        if pts.sum() == 0:
            return 1e6
        return float(dt[pts].mean())

    def _hu_distance(self, stroke_mask: np.ndarray, patch: Patch) -> float:
        se = cv2.Canny((stroke_mask > 0).astype(np.uint8) * 255, 50, 150)
        pe = self._edges(patch.image)
        sc, _ = cv2.findContours(se, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pc, _ = cv2.findContours(pe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not sc or not pc:
            return 1.0
        return cv2.matchShapes(max(sc, key=cv2.contourArea), max(pc, key=cv2.contourArea), cv2.CONTOURS_MATCH_I1, 0.0)

    def _direction_distance(self, stroke_mask: np.ndarray, patch: Patch) -> float:
        def angle(mask: np.ndarray) -> float:
            ys, xs = np.where(mask > 0)
            if len(xs) < 2:
                return 0.0
            cov = np.cov(xs, ys)
            eigvals, eigvecs = np.linalg.eigh(cov)
            vec = eigvecs[:, np.argmax(eigvals)]
            return np.degrees(np.arctan2(vec[1], vec[0]))

        a1 = abs(angle(stroke_mask))
        gray = cv2.cvtColor(patch.image, cv2.COLOR_RGB2GRAY)
        a2 = abs(angle((gray > 0).astype(np.uint8)))
        diff = abs(a1 - a2)
        if diff > 90:
            diff = 180 - diff
        return diff / 90.0

    def score(self, stroke: Stroke, patch: Patch) -> float:
        h, w = patch.image.shape[:2]
        resized = cv2.resize((stroke.mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return (
            self.chamfer_weight * self._chamfer(resized, patch)
            + self.hu_weight * self._hu_distance(resized, patch)
            + self.direction_weight * self._direction_distance(resized, patch)
        )

    def match(self, stroke: Stroke, bank: "PatchBank", k: int = 5) -> List[Tuple[Patch, float]]:
        scored = [(patch, self.score(stroke, patch)) for patch in bank.patches]
        scored.sort(key=lambda x: x[1])
        return scored[:k]
