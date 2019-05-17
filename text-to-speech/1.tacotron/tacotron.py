import tensorflow as tf
from utils import *


def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    step = tf.cast(global_step + 1, dtype = tf.float32)
    return (
        init_lr
        * warmup_steps ** 0.5
        * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
    )


def prenet(inputs, num_units = None, is_training = True, scope = 'prenet'):
    if num_units is None:
        num_units = [embed_size, embed_size // 2]
    with tf.variable_scope(scope):
        outputs = tf.layers.dense(
            inputs,
            units = num_units[0],
            activation = tf.nn.relu,
            name = 'dense1',
        )
        #         outputs = tf.layers.dropout(
        #             outputs,
        #             rate = dropout_rate,
        #             training = is_training,
        #             name = 'dropout1',
        #         )
        outputs = tf.layers.dense(
            outputs,
            units = num_units[1],
            activation = tf.nn.relu,
            name = 'dense2',
        )
    #         outputs = tf.layers.dropout(
    #             outputs,
    #             rate = dropout_rate,
    #             training = is_training,
    #             name = 'dropout2',
    #         )
    return outputs


def bn(
    inputs, is_training = True, activation_fn = None, scope = 'bn', reuse = None
):
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis = 1)
            inputs = tf.expand_dims(inputs, axis = 2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis = 1)

        outputs = tf.contrib.layers.batch_norm(
            inputs = inputs,
            center = True,
            scale = True,
            updates_collections = None,
            is_training = is_training,
            scope = scope,
            fused = True,
            reuse = reuse,
        )
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis = [1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis = 1)
    else:
        outputs = tf.contrib.layers.batch_norm(
            inputs = inputs,
            center = True,
            scale = True,
            updates_collections = None,
            is_training = is_training,
            scope = scope,
            reuse = reuse,
            fused = False,
        )
    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs


def highwaynet(inputs, num_units = None, scope = 'highwaynet'):
    if not num_units:
        num_units = inputs.get_shape()[-1]
    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs, units = num_units, activation = tf.nn.relu, name = 'dense1'
        )
        T = tf.layers.dense(
            inputs,
            units = num_units,
            activation = tf.nn.sigmoid,
            bias_initializer = tf.constant_initializer(-1.0),
            name = 'dense2',
        )
        outputs = H * T + inputs * (1.0 - T)
    return outputs


def conv1d_banks(inputs, K = 16, is_training = True, scope = 'conv1d_banks'):
    with tf.variable_scope(scope):
        outputs = tf.layers.conv1d(inputs, embed_size // 2, 1, padding = 'SAME')
        for k in range(2, K + 1):
            with tf.variable_scope('num_{}'.format(k)):
                output = tf.layers.conv1d(
                    inputs, embed_size // 2, k, padding = 'SAME'
                )
                outputs = tf.concat((outputs, output), -1)
        # outputs = bn(outputs, is_training, tf.nn.relu)
    return outputs


class Tacotron:
    def __init__(self, reuse = None):
        self.X = tf.placeholder(tf.int32, (None, None))
        lookup_table = tf.get_variable(
            'lookup_table',
            dtype = tf.float32,
            shape = [len(vocab), embed_size],
            initializer = tf.truncated_normal_initializer(
                mean = 0.0, stddev = 0.01
            ),
        )
        embedded = tf.nn.embedding_lookup(lookup_table, self.X)
        self.Y = tf.placeholder(tf.float32, (None, None, n_mels * resampled))
        self.decoder_inputs = tf.concat(
            (tf.zeros_like(self.Y[:, :1, :]), self.Y[:, :-1, :]), 1
        )
        self.decoder_inputs = self.decoder_inputs[:, :, -n_mels:]
        self.Z = tf.placeholder(
            tf.float32, (None, None, fourier_window_size // 2 + 1)
        )
        self.training = tf.placeholder(tf.bool, None)
        with tf.variable_scope('encoder', reuse = reuse):
            prenet_out_encoder = prenet(embedded, is_training = self.training)
            enc = conv1d_banks(
                prenet_out_encoder,
                K = decoder_num_banks,
                is_training = self.training,
            )
            enc = tf.layers.max_pooling1d(
                enc, pool_size = 2, strides = 1, padding = 'same'
            )
            enc = tf.layers.conv1d(
                enc,
                embed_size // 2,
                3,
                name = 'encoder-conv1-1',
                padding = 'SAME',
            )
            # enc = bn(enc, self.training, tf.nn.relu, scope = 'encoder-conv1-1')
            enc = tf.layers.conv1d(
                enc,
                embed_size // 2,
                3,
                name = 'encoder-conv1-2',
                padding = 'SAME',
            )
            # enc = bn(enc, self.training, scope = 'encoder-conv1-2')
            enc += prenet_out_encoder
            for i in range(num_highwaynet_blocks):
                enc = highwaynet(
                    enc,
                    num_units = embed_size // 2,
                    scope = 'encoder-highwaynet-{}'.format(i),
                )
            with tf.variable_scope('encoder-gru', reuse = reuse):
                cell = tf.contrib.rnn.GRUCell(embed_size // 2)
                cell_bw = tf.contrib.rnn.GRUCell(embed_size // 2)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell, cell_bw, enc, dtype = tf.float32
                )
                self.memory = tf.concat(outputs, 2)
                self.output = tf.layers.dense(self.memory, len(vocab))
                self.X_seq_len = tf.count_nonzero(self.X, 1, dtype = tf.int32)
                masks = tf.sequence_mask(
                    self.X_seq_len,
                    tf.reduce_max(self.X_seq_len),
                    dtype = tf.float32,
                )
                self.seq_loss = tf.contrib.seq2seq.sequence_loss(
                    logits = self.output, targets = self.X, weights = masks
                )
        with tf.variable_scope('decoder-1', reuse = reuse):
            prenet_out_decoder1 = prenet(
                self.decoder_inputs, is_training = self.training
            )
            with tf.variable_scope('attention-decoder-1', reuse = reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    embed_size, self.memory
                )
                decoder_cell = tf.contrib.rnn.GRUCell(embed_size)
                cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell,
                    attention_mechanism,
                    embed_size,
                    alignment_history = True,
                )
                outputs_attention, state_attention = tf.nn.dynamic_rnn(
                    cell_with_attention, prenet_out_decoder1, dtype = tf.float32
                )
            self.alignments = tf.transpose(
                state_attention.alignment_history.stack(), [1, 2, 0]
            )
            with tf.variable_scope('decoder1-gru-1'):
                cell = tf.contrib.rnn.GRUCell(embed_size)
                outputs, _ = tf.nn.dynamic_rnn(
                    cell, outputs_attention, dtype = tf.float32
                )
                outputs_attention += outputs
            with tf.variable_scope('decoder1-gru-2'):
                cell = tf.contrib.rnn.GRUCell(embed_size)
                outputs, _ = tf.nn.dynamic_rnn(
                    cell, outputs_attention, dtype = tf.float32
                )
                outputs_attention += outputs
            self.Y_hat = tf.layers.dense(outputs_attention, n_mels * resampled)
        with tf.variable_scope('decoder-2', reuse = reuse):
            out_decoder2 = tf.reshape(
                self.Y_hat, [tf.shape(self.Y_hat)[0], -1, n_mels]
            )
            dec = conv1d_banks(
                out_decoder2, K = decoder_num_banks, is_training = self.training
            )
            dec = tf.layers.max_pooling1d(
                dec, pool_size = 2, strides = 1, padding = 'same'
            )
            dec = tf.layers.conv1d(
                dec,
                embed_size // 2,
                3,
                name = 'decoder-conv1-1',
                padding = 'SAME',
            )
            # dec = bn(dec, self.training, tf.nn.relu, scope = 'decoder-conv1-1')
            dec = tf.layers.conv1d(
                dec,
                embed_size // 2,
                3,
                name = 'decoder-conv1-2',
                padding = 'SAME',
            )
            # dec = bn(dec, self.training, scope = 'decoder-conv1-2')
            dec = tf.layers.dense(dec, embed_size // 2)
            for i in range(4):
                dec = highwaynet(
                    dec,
                    num_units = embed_size // 2,
                    scope = 'decoder-highwaynet-{}'.format(i),
                )
            with tf.variable_scope('decoder-gru', reuse = reuse):
                cell = tf.contrib.rnn.GRUCell(embed_size // 2)
                cell_bw = tf.contrib.rnn.GRUCell(embed_size // 2)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell, cell_bw, dec, dtype = tf.float32
                )
                outputs = tf.concat(outputs, 2)
            self.Z_hat = tf.layers.dense(outputs, 1 + fourier_window_size // 2)
        self.loss1 = tf.reduce_mean(tf.abs(self.Y - self.Y_hat))
        self.loss2 = tf.reduce_mean(tf.abs(self.Z - self.Z_hat))
        self.loss = self.loss1 + self.loss2 + self.seq_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(
            self.loss
        )

        self.global_step = tf.Variable(
            0, name = 'global_step', trainable = False
        )
        self.lr = learning_rate_decay(1e-3, global_step = self.global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.gvs = optimizer.compute_gradients(self.loss)
        self.clipped = []
        for grad, var in self.gvs:
            grad = tf.clip_by_norm(grad, 5.0)
            self.clipped.append((grad, var))
        self.optimizer = optimizer.apply_gradients(
            self.clipped, global_step = self.global_step
        )
