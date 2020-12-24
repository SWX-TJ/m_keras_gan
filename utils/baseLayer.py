import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

def spectral_normalization(w):
    return w / spectral_norm(w)

class CondtionBatchNorm(layers.Layer):
    def __init__(self,nums_feature,axis  = -1,momentum = 0.99,eps =1e-5,initializer = tf.initializers.he_normal(),name="CondtionBatchNorm"):
        super(CondtionBatchNorm,self).__init__(name=name)
        self.nums_feature = nums_feature
        self.axis = axis
        self.momentum = momentum
        self.eps = eps
        self.initializer = initializer
        self.gamma = layers.Dense(nums_feature,use_bias=False,kernel_initializer=self.initializer,name='gamma')
        self.beta = layers.Dense(nums_feature,use_bias=False,kernel_initializer=self.initializer,name='beta')
        self.bn = layers.BatchNormalization(self.axis,self.momentum,self.eps)
    def call(self,inputs,training=None):
        x,condition  = inputs
        n_dim = len(x.get_shape().as_list())
        x = self.bn(x,training)
        gama = self.gamma(condition)
        bta = self.beta(condition)
        for i in range(n_dim - 2):
            # print("gama_shape-->",tf.shape(gama))
            # print("bta_shape-->",tf.shape(bta))
            gama = tf.expand_dims(gama, axis = 1)
            bta = tf.expand_dims(bta, axis = 1)
        # print("f_gama_shape-->",tf.shape(gama))
        # print("f_bta_shape-->",tf.shape(bta))
        return x * gama + bta

class Global_Sum_Pooling(layers.Layer):
    def __init__(self,activation = None,name='global_sum_pool'):
        super(Global_Sum_Pooling,self).__init__(name=name)
        self.activation = activation
    def call(self, inputs):
        if self.activation == 'relu':
            inputs = tf.nn.relu(inputs)
            outputs = tf.reduce_sum(inputs, axis=[1, 2],keepdims=False)
        elif self.activation==None:
            outputs = tf.reduce_sum(inputs, axis=[1, 2],keepdims=False)
        return   outputs   

class Inner_Product(layers.Layer):
    def __init__(self,name='inner_product'):
        super(Inner_Product,self).__init__(name=name)
    def call(self,inputs):
        embed_y,h = inputs
        return tf.reduce_sum(h *embed_y, axis=-1, keepdims=True)






