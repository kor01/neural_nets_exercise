import numpy as np


def sigmoid(x):
  return 1.0 / (1.0 - np.exp(-x))


def sigmoid_prime(x):
  sig = sigmoid(x)
  return sig * (1 - sig)


num_batches = 16
batch_size = 4
feature_dim = 16
hidden_size = 128

X = np.random.randn(num_batches, batch_size, feature_dim)

Y = np.random.randint(low=0, high=2, size=(num_batches, batch_size))

hidden_weight = np.random.normal(0, 1 / (np.sqrt(feature_dim) * np.sqrt(hidden_size)),
                                 size=(feature_dim, hidden_size)).astype('float32')
hidden_bias = np.zeros((hidden_size,), dtype=np.float32)
output_weight = np.random.normal(0, 1 / np.sqrt(hidden_size), size=(hidden_size, 1)).astype('float32')

learning_rate = 0.01

batch_idx = 0

global_step = 0

while True:
  batch_idx = batch_idx % num_batches
  x, y = X[batch_idx, :, :], Y[batch_idx, :]

  # forward process

  hidden = np.matmul(x, hidden_weight) + hidden_bias
  hidden_act = sigmoid(hidden)

  # noinspection PyTypeChecker
  output = np.matmul(hidden_act, output_weight)
  output_act = sigmoid(output)

  # xentropy
  loss = -y * np.log(output_act) - (1 - y) * np.log(1 - output_act)

  # backward process

  # dloss / doutput
  dl_output_act = -y * 1.0 / output_act + (1 - y) * 1.0 / (1 - output_act)

  # d_loss / d_output = d_loss / d_output_act * d_output_act / d_output
  dl_output = dl_output_act * output_act * (1 - output_act)

  # dl_output_weight = dl_output * d_output / d_output_weight
  dl_output_weight = dl_output * hidden_act

  dl_hidden_act = dl_output * output_weight

  # dl_hidden = dl_hidden_act * sigmoid_prime(hidden)
  dl_hidden = dl_hidden_act * hidden * (1 - hidden)

  # dl_hidden_weight = tilt(x, hidden_dim) * dl_hidden
  dl_hidden_weight = np.expand_dims(x, axis=1) * np.expand_dims(dl_hidden, axis=2)
  dl_hidden_bias = dl_hidden

  # update weights

  output_weight -= np.mean(dl_output_weight, axis=0)
  hidden_weight -= np.mean(dl_hidden_weight, axis=0)
  hidden_bias -= np.mean(dl_hidden_bias, axis=0)

  global_step += 1
  print('global_step [%d] loss = %f' % (global_step, loss))
