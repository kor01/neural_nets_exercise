import tensorflow as tf



def two_layer_mlp_net(inputs, hidden_size, num_classes):

  hidden_0 = tf.layers.dense(
    inputs, hidden_size, activation=tf.nn.sigmoid)

  hidden_1 = tf.layers.dense(
    hidden_0, hidden_size, activation=tf.nn.sigmoid)

  outputs = tf.layers.dense(hidden_1, num_classes, use_bias=False)

  logits = tf.nn.softmax(outputs)

  return logits


def xent_loss(softmax, labels):
  ret = tf.reduce_sum(-tf.log(softmax) * labels, axis=-1)
  return tf.reduce_mean(ret, axis=0)


def sgd_optimize(loss):
  opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
  variables = tf.trainable_variables()
  grad = tf.gradients(loss, variables)
  pairs = list(zip(grad, variables))
  train_op = opt.apply_gradients(pairs)
  return train_op, pairs
