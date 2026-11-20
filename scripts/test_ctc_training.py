"""Checks for scientific bookkeeping; fixtures here are not training data."""
import unittest
import numpy as np
import torch
from train_ctc_detector import (TRAIN, VALID, Detector, centers, counts, crop,
                                decode, heatmap, normalize)


class Checks(unittest.TestCase):
    def test_disjoint_split(self):
        self.assertFalse(set(TRAIN) & set(VALID))
        self.assertGreater(min(VALID)-max(TRAIN), 10)

    def test_matching_is_one_to_one_and_physical(self):
        truth = np.array([[0., 0., 0.]])
        self.assertEqual(counts(np.array([[0., 0., 0.], [0., 0., 1.]]), truth), [1, 1, 0])
        self.assertEqual(counts(np.array([[4., 0., 0.]]), truth), [0, 1, 1])
        self.assertEqual(counts(np.array([[0., 0., 4.]]), truth), [1, 0, 0])
        self.assertEqual(counts(np.empty((0, 3)), truth), [0, 0, 1])

    def test_centroid_keeps_label_identity(self):
        mask = np.zeros((3, 6, 6), dtype=np.uint16)
        mask[1, 1:3, 1:3] = 7
        mask[2, 5, 5] = 99
        np.testing.assert_allclose(centers(mask), [[1, 1.5, 1.5], [2, 5, 5]])

    def test_downsample_coordinates(self):
        point = np.array([[2., 40.5, 40.5]])
        h = heatmap((5, 50, 50), point)
        pred = decode(h, .5)
        np.testing.assert_allclose(pred, point)

    def test_forward_and_crop(self):
        a = normalize(np.ones((5, 81, 83), dtype=np.uint16))
        self.assertTrue(np.isfinite(a).all())
        target = np.zeros_like(a)
        x, y = crop(a, target, np.empty((0, 3)), np.random.default_rng(42))
        self.assertEqual(x.shape, (5, 40, 41))
        net = Detector()
        out = net(torch.from_numpy(x)[None, None])
        self.assertEqual(tuple(out.shape), (1, 1, 5, 40, 41))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_shallow_crop_does_not_invent_negative_slices(self):
        a = np.ones((5, 70, 70), dtype=np.float32)
        x, y = crop(a, a.copy(), np.empty((0, 3)), np.random.default_rng(1))
        self.assertEqual(x.shape, (5, 64, 64))
        np.testing.assert_array_equal(x, y)


if __name__ == '__main__':
    unittest.main()
