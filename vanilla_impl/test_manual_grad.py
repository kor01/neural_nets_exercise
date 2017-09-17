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


class TestLinear(unittest.TestCase):

  def test_w(self):

    x = np.random.randn(16, 128)

    b = np.random.rand(73)

    def forward_func(w):
      w = w.reshape(128, 73)
      hidden = manual_grad.xw_plus_b(x, w, b)
      ret = manual_grad.xw_plus_b(hidden, reduce_param, 0).mean(axis=0)[0]
      return ret

    def backward_func(w):
      _ = w.reshape(128, 73)
      initial_grad = np.ones((16, 1), dtype=np.float32)
      grad = manual_grad.xw_plus_b_grad_x(initial_grad, reduce_param)
      ret = manual_grad.xw_plus_b_grad_w(grad, x)
      ret = ret.mean(axis=0)
      return ret.reshape((-1,))

    for _ in range(20):
      reduce_param = np.random.rand(73, 1)
      start_point = np.random.randn(128, 73).reshape((-1,))
      error = optimize.check_grad(
        forward_func, backward_func, start_point)
      self.assertLess(error, 1e-4)

  def test_b(self):

    w = np.random.randn(128, 73)
    x = np.random.randn(16, 128)
    reduce_param = np.random.rand(73, 1)

    def forward_func(b):
      hidden = manual_grad.xw_plus_b(x, w, b)
      ret = manual_grad.xw_plus_b(hidden, reduce_param, 0).mean(axis=0)[0]
      return ret

    def backward_func(b):
      initial_grad = np.ones((16, 1), dtype=np.float32)
      grad = manual_grad.xw_plus_b_grad_x(initial_grad, reduce_param)
      return grad.mean(axis=0)

    for _ in range(20):
      reduce_param = np.random.rand(73, 1)
      start_point = np.random.randn(73)
      error = optimize.check_grad(forward_func, backward_func, start_point)
      self.assertLess(error, 1e-4)


class TestSigmoid(unittest.TestCase):

  def test_sigmoid(self):
    reduce_param = np.random.rand(73, 1)
    idx = 0

    def forward_func(x):
      batch = np.random.rand(16, 73)
      batch[idx, :] = x
      ret = manual_grad.sigmoid(batch)
      ret = manual_grad.xw_plus_b(ret, reduce_param, 0)
      return ret[idx]

    def backward_func(x):
      batch = np.random.rand(16, 73)
      batch[idx, :] = x
      initial_grad = np.ones((16, 1), dtype=np.float32)
      grad = manual_grad.xw_plus_b_grad_x(initial_grad, reduce_param)
      sig = manual_grad.sigmoid(batch)
      grad = manual_grad.sigmoid_grad(sig, grad)
      return grad[idx]

    for _ in range(20):
      start_point = np.random.randn(73)
      idx = np.random.randint(0, 16)
      error = optimize.check_grad(forward_func, backward_func, start_point)
      self.assertLess(error, 1e-4)



