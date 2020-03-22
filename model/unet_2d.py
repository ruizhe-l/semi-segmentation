from __future__ import print_function, division, absolute_import

import sklearn
import numpy as np
import tensorflow as tf
import tf_eager.util as U

is_bn = True
is_dice_loss = True
norm_test = False


layers = tf.keras.layers
initializers = tf.keras.initializers

class _Residual(tf.keras.Model):
    def __init__(self, features, name_scope):
        super(_Residual, self).__init__(name=name_scope)
        self.output_dims = features
        stddev = np.sqrt(2 / features)
        # self.conv = layers.Conv2D(features, 1, padding='SAME', use_bias=True,
        #                             kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
        #                             name='conv')
        # self.bn = layers.BatchNormalization(name='bn')
    
    def call(self, x1, x2, train_phase=False):
        # x = self.conv(x1)
        # x = self.bn(x, training=train_phase)
        if x1.shape[-1] < x2.shape[-1]:
            x = tf.concat([x1, tf.zeros([x1.shape[0], x1.shape[1], x1.shape[2], x2.shape[3] - x1.shape[3]])], axis=-1)
        else:
            x = x1[..., :x2.shape[-1]]
        x = x + x2
        # x = x + self.bias
        return x


class _DownSampling(tf.keras.Model):
    def __init__(self, features, filter_size, res, name_scope):
        super(_DownSampling, self).__init__(name=name_scope)
        stddev = np.sqrt(2 / (filter_size**2 * features))
        self.conv1 = layers.Conv2D(features, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv1')
        self.conv2 = layers.Conv2D(features, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv2')
        if is_bn:
            self.bn1 = layers.BatchNormalization(name='bn1', momentum=0.99)
            self.bn2 = layers.BatchNormalization(name='bn2', momentum=0.99)
        if res:
            self.res_block = _Residual(features, 'res')
        self.res = res

    def call(self, input_tensor, keep_prob, train_phase):
        # conv1
        x = self.conv1(input_tensor)
        x = tf.nn.dropout(x, keep_prob)
        if is_bn:
            x = self.bn1(x, training=train_phase)
        x = tf.nn.relu(x)
        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, keep_prob)
        if is_bn:
            x = self.bn2(x, training=train_phase)
        if self.res:
            x = self.res_block(input_tensor, x, train_phase)
        x = tf.nn.relu(x)
        # res
        
        return x

class _UpSampling(tf.keras.Model):
    def __init__(self, features, filter_size, pool_size, concat_or_add, res, name_scope):
        super(_UpSampling, self).__init__(name=name_scope)
        stddev = np.sqrt(2 / (filter_size**2 * features))
        self.deconv = layers.Conv2DTranspose(features//2, filter_size, strides=(pool_size, pool_size), padding='SAME',
                                            kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                            name='deconv')
        if is_bn:
            self.bn_deconv = layers.BatchNormalization(name='bn_deconv')
        self.conv1 = layers.Conv2D(features//2, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv1')
        self.conv2 = layers.Conv2D(features//2, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv2')
        if is_bn:
            self.bn1 = layers.BatchNormalization(name='bn1', momentum=0.99)
            self.bn2 = layers.BatchNormalization(name='bn2', momentum=0.99)
        
        if res:
            self.res_block = _Residual(features//2, 'res')

        self.concat_or_add = concat_or_add
        self.res = res


    def call(self, input_tensor, dw_tensor, keep_prob, train_phase):
        # deconv
        x = self.deconv(input_tensor)
        if is_bn:
            x = self.bn_deconv(x, training=train_phase)
        x = tf.nn.relu(x)
        # concatenate
        if self.concat_or_add == 'concat':
            x = self._crop_and_concat(dw_tensor, x)
        elif self.concat_or_add == 'add':
            x = self._crop_and_add(dw_tensor, x)
        else:
            raise Exception('Wrong concatenate method!')
        res_in = x
        # conv1
        x = self.conv1(x)
        x = tf.nn.dropout(x, keep_prob)
        if is_bn:
            x = self.bn1(x, training=train_phase)
        x = tf.nn.relu(x)
        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, keep_prob)
        if is_bn:
            x = self.bn2(x, training=train_phase)
        if self.res:
            x = self.res_block(res_in, x, train_phase)
        x = tf.nn.relu(x)
        return x

    def _crop_and_concat(self, x1, x2):
        return tf.concat((x1, x2), 3)

    def _crop_and_add(self, x1, x2):
        return x1 + x2

class Unet2D(tf.keras.Model):
    def __init__(self, n_class, n_layer, features_root, filter_size, pool_size, concat_or_add, res):
        super(Unet2D, self).__init__(name='')
        self.dw_layers = dict()
        self.up_layers = dict()
        self.max_pools = dict()

        for layer in range(n_layer):
            features = 2**layer*features_root
            dict_key = str(n_layer-layer-1)
            dw = _DownSampling(features, filter_size, res, 'dw_%s'%dict_key)
            self.dw_layers[dict_key] = dw
            if layer < n_layer-1:
                pool = layers.MaxPool2D(pool_size=(pool_size, pool_size), padding='SAME')
                self.max_pools[dict_key] = pool

        for layer in range(n_layer-2, -1 ,-1):
            features = 2**(layer+1)*features_root
            dict_key = str(n_layer-layer-1)
            up = _UpSampling(features, filter_size, pool_size, concat_or_add, res, 'up_%s'%dict_key)
            self.up_layers[dict_key] = up
        
        stddev = np.sqrt(2 / (filter_size**2 * features_root))
        self.conv_out = layers.Conv2D(n_class, 1, padding='SAME', use_bias=False,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                    name='conv_out')


    def call(self, input_tensor, keep_prob, train_phase):
        dw_tensors = dict()
        x = input_tensor
        n_layer = len(self.dw_layers)
        for i in range(n_layer):
            dict_key = str(n_layer-i-1)
            dw_tensors[dict_key] = self.dw_layers[dict_key](x, keep_prob, train_phase)
            x = dw_tensors[dict_key]
            if i < len(self.max_pools):
                x = self.max_pools[dict_key](x)
        
        for i in range(n_layer-2, -1 ,-1):
            dict_key = str(n_layer-i-1)
            x = self.up_layers[dict_key](x, dw_tensors[dict_key], keep_prob, train_phase)

        x = self.conv_out(x)
        x = tf.nn.relu(x)
        return x

class Model:
    def __init__(self, n_class, n_layer=5, features_root=16, filter_size=3, pool_size=2, weight_type=None, keep_prob=1., concat_or_add='concat', res=True):
        self.net = Unet2D(n_class, n_layer, features_root, filter_size, pool_size, concat_or_add, res)
        self.n_class = n_class
        self.weight_type = weight_type
        self.keep_prob = keep_prob

    def evaluation(self, feed_dict):
        x = tf.constant(feed_dict['x'], tf.float32)
        y = tf.constant(feed_dict['y'], tf.float32)
        # mask = tf.constant(feed_dict['mask'], tf.float32)
        logits = self.net(x, 1., norm_test)
        pred_prob = tf.nn.softmax(logits, -1)
        loss = self.get_loss(logits, y, None)
        dice_loss = self.get_dice_loss(logits, y)
        total_loss = loss + dice_loss# + second_loss

        pred = tf.one_hot(tf.argmax(pred_prob, -1), self.n_class)
        flat_labels = tf.reshape(y, [-1, self.n_class])
        flat_pred = tf.reshape(pred, [-1, self.n_class])
        # flat_mask = tf.reshape(mask, [-1])
        # flat_mask = tf.concat([flat_mask, flat_mask], -1)

        # dice
        eps = 1e-5
        intersection = tf.reduce_sum(pred * y, axis=[0,1,2])
        sum_ = eps + tf.reduce_sum(pred + y, axis=[0,1,2])
        dice = 2 * intersection / sum_

        # iou
        iou = intersection / (sum_ - intersection)

        flat_labels_argmax = np.argmax(flat_labels, -1)
        flat_pred_argmax = np.argmax(flat_pred, -1)

        acc = sklearn.metrics.accuracy_score(flat_labels_argmax, flat_pred_argmax)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(flat_labels_argmax, flat_pred_argmax, labels=list(range(self.n_class))).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        dice = dice[1:]
        iou = iou[1:]

        result = {'prediction':np.array(pred_prob),
                  'loss':np.array(total_loss),
                  'dice':np.array(dice),
                  'iou':np.array(iou),
                  'acc':np.array(acc),
                  'sensitivity':np.array(sensitivity),
                  'specificity':np.array(specificity)}
        return result

    def evaluation_dice(self, feed_dict):
        x = tf.constant(feed_dict['x'], tf.float32)
        y = tf.constant(feed_dict['y'], tf.float32)
        # mask = tf.constant(feed_dict['mask'], tf.float32)
        logits = self.net(x, 1., norm_test)
        pred_prob = tf.nn.softmax(logits, -1)
        pred = tf.one_hot(tf.argmax(pred_prob, -1), self.n_class)
        flat_labels = tf.reshape(y, [-1, self.n_class])
        flat_pred = tf.reshape(pred, [-1, self.n_class])

        # dice
        eps = 1e-5
        intersection = tf.reduce_sum(flat_pred * flat_labels, axis=0)
        sum_ = eps + tf.reduce_sum(flat_pred + flat_labels, axis=0)
        dice = 2 * intersection / sum_

        return np.array(dice)


    def get_grads(self, feed_dict): 
        x = tf.constant(feed_dict['x'], tf.float32)
        y = tf.constant(feed_dict['y'], tf.float32)
        # mask = feed_dict['mask']
        with tf.GradientTape() as grads_tape:
            logits = self.net(x, self.keep_prob, True)
            loss = self.get_loss(logits, y, None)
            dice_loss = self.get_dice_loss(logits, y)
            total_loss = loss + dice_loss# + second_loss
        grads = grads_tape.gradient(total_loss, self.net.variables)
        return grads

    def get_logits(self, x): 
        logits = self.net(x, self.keep_prob, True)

        return logits

    def predict(self, x):
        x = tf.constant(x, tf.float32)
        return tf.nn.softmax(self.net(x, 1., norm_test), -1)

    def get_loss(self, logits, labels, masks):
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])
        loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)

        loss = tf.reduce_mean(loss_map)     
        return loss

    def get_dice_loss(self, logits, labels):
        if not is_dice_loss:
            return 0
        pred_prob = tf.nn.softmax(logits, -1)
        pred = tf.one_hot(tf.argmax(pred_prob, -1), self.n_class)
        eps = 1e-5
        intersection = tf.reduce_sum(pred * labels, axis=[0,1,2])
        sum_ = eps + tf.reduce_sum(pred + labels, axis=[0,1,2])
        dice = 2 * intersection / sum_
        loss = 1. - tf.reduce_mean(dice)
        return loss
