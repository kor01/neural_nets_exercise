import unittest
from scipy import optimize
import numpy as np
from vanilla_impl import manual_grad


class TestSoftmaxXent(unittest.TestCase):

  def test_body(self):

    y = np.random.randint(0, 10, size=(16,))
    idx = 0

    def forward_func(x):
      batch = np.random.rand(16, 10)
      batch[idx, :] = x
      sm = manual_grad.softmax(batch)
      ret = manual_grad.sparse_xent(sm, y)[idx]
      return ret

    def backward_func(x):
      batch = np.random.rand(16, 10)
      batch[idx, :] = x
      sm = manual_grad.softmax(batch)
      ret = manual_grad.sparse_softmax_xent_grad(sm, y)[idx, :]
      return ret

    for _ in range(20):
      idx = np.random.randint(0, 16)
      start_point = np.random.rand(10)
      error = optimize.check_grad(
        forward_func, backward_func, start_point)
      self.assertLess(error, 1e-4)


class TestLinearLayer(unittest.TestCase):

  def test_body(self):
    pass

