import tensorflow as tf

def GeneratorLoss(fake_logits,y_true,y_pred,pose_true,pose_pred,beta1 = 0.1,beta2 = 0.1):
    gan_loss = -tf.reduce_mean(fake_logits)
    context_loss = tf.reduce_mean(tf.square(y_pred-y_true))
    pose_loss = tf.reduce_mean(tf.square(pose_pred-pose_true))
    total_loss = gan_loss+beta1*context_loss+beta2*pose_loss
    return total_loss

def DiscriminatorLoss(fake_logits,real_logits,pose_true,pose_pred):
    gan_r_loss = -tf.reduce_mean(real_logits)
    gan_f_loss = tf.reduce_mean(fake_logits)
    pose_loss  = tf.reduce_mean(tf.square(pose_pred-pose_true))
    total_loss = gan_r_loss+gan_f_loss+pose_loss
    return total_loss