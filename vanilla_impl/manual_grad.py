import numpy as np


def softmax(x):
  exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return exp / exp.sum(axis=-1, keepdims=True)


def softmax_grad(sm, y):
  batch_size = sm.shape[0]
  sm_y = sm[[range(batch_size), y]]
  sm_y_expand = np.expand_dims(sm_y, axis=-1)
  ret = - sm * sm_y_expand
  ret[[range(batch_size), y]] += sm_y
  return ret


def sparse_softmax_xent_grad(sm, y):
  batch_size = sm.shape[0]
  sm_y = sm[[range(batch_size), y]]
  sm_y = np.expand_dims(sm_y, axis=-1)
  ret = (-1.0 / sm_y) * softmax_grad(sm, y)
  return ret


def sparse_xent(x, y):
  batch_size = x.shape[0]
  return -np.log(x[range(batch_size), y])


def xw_plus_b(x, w, b):
  return np.matmul(x, w) + b


def xw_plus_b_grad_w(grad, x):
  x = np.expand_dims(x, axis=-1)
  grad = np.expand_dims(grad, axis=-2)
  return x * grad


def xw_plus_b_grad_x(grad, w):
  return np.matmul(grad, w.T)


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
