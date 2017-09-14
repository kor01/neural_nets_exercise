import numpy as np
from vanilla_impl import manual_grad


# noinspection PyAttributeOutsideInit
class MLPNet(object):

  def __init__(self, feature_dim, hidden_size,
               num_classes, learning_rate):

    self.global_step = 0
    self.learning_rate = learning_rate
    self.hidden_0_weight = np.random.normal(
      0, 1 / (np.sqrt(feature_dim) * np.sqrt(hidden_size)),
      size=(feature_dim, hidden_size)).astype('float32')
    self.hidden_0_bias = np.zeros((hidden_size,), dtype=np.float32)

    self.hidden_1_weight = np.random.normal(
      0, 1 / hidden_size,
      size=(hidden_size, hidden_size)).astype('float32')

    self.hidden_1_bias = np.zeros((hidden_size,), dtype=np.float32)
    self.output_weight = np.random.normal(
      0, 1 / np.sqrt(hidden_size * num_classes),
      size=(hidden_size, num_classes)).astype('float32')
    self.feature_dim = feature_dim
    self.hidden_size = hidden_size
    self.num_classes = num_classes

  def forward(self, x, y):
    # forward pass
    self.x, self.y = x, y
    self.hidden_0 = manual_grad.xw_plus_b(
      x, self.hidden_0_weight, self.hidden_0_bias)
    self.hidden_0_act = manual_grad.sigmoid(self.hidden_0)
    self.hidden_1 = manual_grad.xw_plus_b(
      self.hidden_0_act, self.hidden_1_weight, self.hidden_1_bias)
    self.hidden_1_act = manual_grad.sigmoid(self.hidden_1)
    self.output = manual_grad.xw_plus_b(self.hidden_1_act, self.output_weight, 0)
    self.sm = manual_grad.softmax(self.output)
    self.xent = manual_grad.sparse_xent(self.sm, self.y)

  def backward(self):
    self.output_grad = manual_grad.sparse_softmax_xent_grad(self.y, self.xent, self.sm)
    self.output_weight_grad = manual_grad.xw_plus_b_grad_w(self.output_grad, self.x)
    self.hidden_1_act_grad = manual_grad.xw_plus_b_grad_x(self.output_grad, self.output_weight)
    self.hidden_1_grad = manual_grad.sigmoid_grad(self.hidden_1_act, self.hidden_1_act_grad)
    self.hidden_weight_1_grad = manual_grad.xw_plus_b_grad_w(self.hidden_1_grad, self.hidden_0_act)
    self.hidden_bias_1_grad = self.hidden_1_grad
    self.hidden_0_act_grad = manual_grad.xw_plus_b_grad_x(self.hidden_1_grad, self.hidden_1_weight)
    self.hidden_0_grad = manual_grad.sigmoid_grad(self.hidden_0_act, self.hidden_0_act_grad)
    self.hidden_weight_0_grad = manual_grad.xw_plus_b_grad_w(self.hidden_0_grad, self.x)
    self.hidden_bias_0_grad = self.hidden_0_grad

  def update(self):
    self.output_weight -= self.learning_rate * self.output_weight_grad.mean(axis=0)
    self.hidden_1_weight -= self.learning_rate * self.hidden_weight_1_grad.mean(axis=0)
    self.hidden_1_bias -= self.learning_rate * self.hidden_bias_0_grad.mean(axis=0)
    self.hidden_0_weight -= self.learning_rate * self.hidden_weight_0_grad.mean(axis=0)
    self.hidden_0_bias -= self.learning_rate * self.hidden_bias_0_grad.mean(axis=0)
    self.global_step += 1

