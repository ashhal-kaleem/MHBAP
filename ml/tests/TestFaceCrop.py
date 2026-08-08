"""
Unit tests for the new face cropping logic in SessionRunner.
"""
from __future__ import annotations

import unittest
import numpy as np

from ml.SessionRunner import _crop_face


class TestFaceCrop(unittest.TestCase):

    def setUp(self):
        # 100x100 fake frame
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_crop_face_none_frame(self):
        self.assertIsNone(_crop_face(None, (0, 0, 1, 1)))

    def test_crop_face_none_bbox(self):
        self.assertIsNone(_crop_face(self.frame, None))

    def test_crop_face_normal(self):
        # Center 50x50 box: x from 25 to 75, y from 25 to 75
        bbox = (0.25, 0.25, 0.75, 0.75)
        # padding is 0.25 by default. Box width = 50. Padding = int(50*0.25) = 12
        # Expected crop: x from 25-12=13 to 75+12=87. y from 13 to 87.
        # Shape should be 87-13 = 74
        crop = _crop_face(self.frame, bbox)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (74, 74, 3))

    def test_crop_face_boundary_clamp(self):
        # Box touching top left with size that after padding stays >=64px
        # Using a larger bbox (0.0, 0.0, 0.6, 0.6) -> width 60, padding 15, resulting crop 75x75
        bbox = (0.0, 0.0, 0.6, 0.6)
        crop = _crop_face(self.frame, bbox)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (75, 75, 3))

        # Box touching bottom right with same size
        bbox2 = (0.4, 0.4, 1.0, 1.0)
        crop2 = _crop_face(self.frame, bbox2)
        self.assertIsNotNone(crop2)
        self.assertEqual(crop2.shape, (75, 75, 3))

    def test_crop_face_degenerate(self):
        # Width/height < 8 pixels should return None
        # Width = 0
        self.assertIsNone(_crop_face(self.frame, (0.5, 0.5, 0.5, 0.5)))
        # Width = 5 (with 0.25 padding, total width = 5 + 1 + 1 = 7)
        self.assertIsNone(_crop_face(self.frame, (0.50, 0.50, 0.55, 0.55)))


if __name__ == "__main__":
    unittest.main()
