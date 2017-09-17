import unittest
from scipy import optimize
import numpy as np
from vanilla_impl import manual_grad
from vanilla_impl import two_layer_mlp

batch_size = 16

np.random.seed(0)

net = two_layer_mlp.MLPNet(256, 128, 10, 0.001)
x = np.random.randn(batch_size, 256)
y = np.random.randint(0, 10, size=batch_size)


class TestMLPNet(unittest.TestCase):

  def test_output_grad(self):

    def forward_func(outputs):
      outputs = outputs.reshape(batch_size, 128)
      softmax = manual_grad.softmax(outputs)
      xent = manual_grad.sparse_xent(softmax, y)
      return xent.sum(axis=0)

    def backward_func(outputs):
      outputs = outputs.reshape(batch_size, 128)
      softmax = manual_grad.softmax(outputs)
      output_grad = manual_grad.sparse_softmax_xent_grad(
        softmax, y)
      return output_grad.reshape((-1,))

    for _ in range(4):
      start_point = np.random.randn(batch_size, 128).reshape((-1,))
      error = optimize.check_grad(forward_func, backward_func, start_point)
      self.assertLess(error, 1e-4)

  def test_output_weight_grad(self):

    net.forward(x, y)

    def forward_func(w):
      w = w.reshape((256, 10))
      outputs = manual_grad.xw_plus_b(x, w, 0)
      softmax = manual_grad.softmax(outputs)
      xent = manual_grad.sparse_xent(softmax, y)
      return xent.mean(axis=0)

    def backward_func(w):
      w = w.reshape((256, 10))
      outputs = manual_grad.xw_plus_b(x, w, 0)
      softmax = manual_grad.softmax(outputs)
      outputs_grad = manual_grad.sparse_softmax_xent_grad(
        softmax, y)
      w_grad = manual_grad.xw_plus_b_grad_w(outputs_grad, x)
      return w_grad.mean(axis=0).reshape((-1,))

    for _ in range(4):
      start_point = np.random.randn(256, 10).reshape((-1,))
      error = optimize.check_grad(
        forward_func, backward_func, start_point)
      self.assertLess(error, 1e-4)

  def test_net_output_weight_grad(self):

    def forward_func(w):
      net.output_weight = w.reshape(
        net.output_weight.shape).astype('float32')
      net.forward(x, y)
      ret = net.xent.mean(axis=0)
      valid_ret = forward_func_valid(w)
      diff = np.abs(ret - valid_ret).sum()
      assert diff < 1e-5
      return ret

    def forward_func_valid(w):
      w = w.reshape((128, 10))
      outputs = manual_grad.xw_plus_b(net.hidden_1_act, w, 0)
      softmax = manual_grad.softmax(outputs)
      xent = manual_grad.sparse_xent(softmax, y)
      return xent.mean(axis=0)

    def backward_func(w):
      net.output_weight = w.reshape(
        net.output_weight.shape).astype('float32')
      net.forward(x, y)
      net.backward()
      ret = net.output_weight_grad.mean(axis=0).reshape((-1,))
      valid_ret = backward_func_valid(w)
      diff = np.abs(valid_ret - ret).sum()
      assert diff < 1e-5
      return ret

    def backward_func_valid(w):
      w = w.reshape((128, 10))
      outputs = manual_grad.xw_plus_b(net.hidden_1_act, w, 0)
      softmax = manual_grad.softmax(outputs)
      outputs_grad = manual_grad.sparse_softmax_xent_grad(
        softmax, y)
      w_grad = manual_grad.xw_plus_b_grad_w(
        outputs_grad, net.hidden_1_act)
      return w_grad.mean(axis=0).reshape((-1,))

    for _ in range(10):
      start_point = np.random.randn(128, 10).reshape((-1,)).astype('float32')
      error = optimize.check_grad(
        forward_func_valid,
        backward_func_valid, start_point)

      grad_diff = backward_func_valid(start_point) - backward_func(start_point)
      max_diff = np.abs(grad_diff).max()
      forward_diff = forward_func(start_point) - forward_func_valid(start_point)
      max_forward_diff = np.abs(forward_diff).max()

      print('max_forward_diff', max_forward_diff)
      print('max_diff:', max_diff)
      print('valid error:', error)
      self.assertLess(error, 1e-4)
      error = optimize.check_grad(
        forward_func, backward_func, start_point, epsilon=1e-5)
      print('output_weight', error)

  def test_hidden_weight_1_grad(self):

    def forward_func(w):
      w = w.reshape(net.hidden_1_weight.shape)
      net.hidden_1_weight[:] = w
      net.forward(x, y)
      return net.xent.mean(axis=0)

    def backward_func(w):
      w = w.reshape(net.hidden_1_weight.shape)
      net.hidden_1_weight[:] = w
      net.forward(x, y)
      net.backward()
      ret = net.hidden_weight_1_grad.mean(axis=0)
      return ret.reshape((-1,))

    for _ in range(1):
      start_point = np.random.randn(
        *net.hidden_1_weight.shape).astype('float32').reshape((-1,))
      error = optimize.check_grad(
        forward_func, backward_func, start_point, epsilon=1e-4)
      self.assertLess(error, 1e-4)
      print('hidden_weight_1:', error)

  def test_bias_1_grad(self):

    def forward_func(b):
      net.hidden_1_bias[:] = b
      net.forward(x, y)
      return net.xent.mean(axis=0)

    def backward_func(b):
      net.hidden_1_bias[:] = b
      net.forward(x, y)
      net.backward()
      return net.hidden_bias_1_grad.mean(axis=0)

    for _ in range(10):
      start_point = np.random.randn(
        *net.hidden_1_bias.shape).astype('float32').reshape((-1,))

      error = optimize.check_grad(
        forward_func, backward_func, start_point, epsilon=1e-4)
      print('bias1_error', error)
      self.assertLess(error, 1e-4)

  def test_hidden_0_weight_grad(self):

    def forward_func(w):
      net.hidden_0_weight[:] = w.reshape(net.hidden_0_weight.shape)
      net.forward(x, y)
      return net.xent.mean(axis=0)

    def backward_func(w):
      net.hidden_0_weight[:] = w.reshape(net.hidden_0_weight.shape)
      net.forward(x, y)
      net.backward()
      return net.hidden_weight_0_grad.mean(axis=0).reshape((-1,))

    for _ in range(2):
      start_point = np.random.randn(
        *net.hidden_0_weight.shape).astype('float32').reshape((-1,))
      error = optimize.check_grad(
        forward_func, backward_func, start_point, epsilon=1e-4)
      self.assertLess(error, 1e-5)
      print('hidden_0_weight_error', error)

  def test_hidden_0_bias_grad(self):

    def forward_func(w):
      net.hidden_0_bias[:] = w
      net.forward(x, y)
      return net.xent.mean(axis=0)

    def backward_func(w):
      net.hidden_0_bias[:] = w
      net.forward(x, y)
      net.backward()
      return net.hidden_bias_0_grad.mean(axis=0)

    for _ in range(2):
      start_point = np.random.randn(
        *net.hidden_0_bias.shape).astype('float32').reshape((-1,))
      error = optimize.check_grad(
        forward_func, backward_func, start_point)
      print('bias_0_error:', error)
