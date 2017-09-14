import numpy as np


def softmax(x):
  exp = np.exp(x)
  return exp / exp.sum(axis=-1, keepdims=True)


def softmax_grad(sm, y):
  sm_y = sm[:, y]
  ret = - sm * sm_y
  ret[:, y] += sm_y
  return ret


def sparse_softmax_xent_grad(y, xent, sm):
  return (-1.0 / xent) * softmax_grad(sm, y)


def sparse_xent(x, y):
  return -np.log(x[:, y])


def xw_plus_b(x, w, b):
  return np.matmul(x, w) + b


def xw_plus_b_grad_w(grad, x):
  x = np.expand_dims(x, axis=-1)
  grad = np.expand_dims(grad, axis=-2)
  return x * grad


def xw_plus_b_grad_x(grad, w):
  return np.matmul(w, grad)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sigmoid_grad(sig, grad):
  return sig * (1 - sig) * grad


def relu(x):
  mask = np.cast((x > 0), 'float32')
  return x * mask


def relu_grad(x):
  mask = np.cast((x > 0), 'float32')
  return mask
