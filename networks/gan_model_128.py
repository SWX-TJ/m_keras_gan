import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from utils.baseLayer import *
from utils.Loss import *
from datasets.dataloader import *
class Generator(Model):
    def __init__(self,nums_cls = 10,gf_dim=64,latent_z_dim = 7,name='Generator'):
        super(Generator,self).__init__(name=name)
        self.nums_cls = nums_cls
        self.gf_dim = gf_dim
        self.embed = layers.Embedding(self.nums_cls,latent_z_dim,name='embed_y')
        self.dense = layers.Dense(gf_dim*8*8*8,name='dense')
        self.cbn_1 = CondtionBatchNorm(gf_dim*8,name='cbn1')
        self.deconv_1 = layers.Conv2DTranspose(gf_dim*4,4,(2,2),'same',kernel_constraint=spectral_normalization,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev = 0.02))
        self.cbn_2 =CondtionBatchNorm(gf_dim*4,name='cbn2')
        self.deconv_2 = layers.Conv2DTranspose(gf_dim*2,4,(2,2),'same',kernel_constraint=spectral_normalization,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev = 0.02))
        self.cbn_3 = CondtionBatchNorm(gf_dim*2,name='cbn3')
        self.deconv_3 = layers.Conv2DTranspose(gf_dim,4,(2,2),'same',kernel_constraint=spectral_normalization,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev = 0.02))
        self.cbn_4 = CondtionBatchNorm(gf_dim,name='cbn4')
        self.deconv_4 = layers.Conv2DTranspose(3,4,(2,2),'same',kernel_constraint=spectral_normalization,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev = 0.02))
    def call(self,inputs,training=None):
        x,c = inputs
        embed_c = self.embed(c)
        x = self.dense(x)
        x = tf.reshape(x,(-1,8,8,self.gf_dim*8))
        x = self.cbn_1([x,embed_c],training)
        x = tf.nn.relu(x)
        x = self.deconv_1(x)
        x = self.cbn_2([x,embed_c],training)
        x = tf.nn.relu(x)
        x = self.deconv_2(x)
        x = self.cbn_3([x,embed_c],training)
        x = tf.nn.relu(x)
        x = self.deconv_3(x)
        x = self.cbn_4([x,embed_c],training)
        x = tf.nn.relu(x)
        x = self.deconv_4(x)
        outputs =tf.nn.tanh(x)
        return outputs

class Discriminator(Model):
    def __init__(self,nums_cls  =10,df_dim=64,latent_z_dim = 7,name='Discriminator'):
        super(Discriminator,self).__init__(name = name)
        self.df_dim = df_dim
        self.nums_cls = nums_cls
        self.embed = layers.Embedding(nums_cls,df_dim*8,name='embed')
        self.conv_1  = layers.Conv2D(df_dim,3,(2,2),'same',kernel_constraint=spectral_normalization,name='conv1')
        self.sbn1 = layers.BatchNormalization(gamma_constraint=spectral_normalization,name='bn1')
        self.conv_2 = layers.Conv2D(df_dim*2,3,(2,2),'same',kernel_constraint=spectral_normalization,name='conv2')
        self.sbn2 = layers.BatchNormalization(gamma_constraint=spectral_normalization,name='bn2')
        self.conv_3  = layers.Conv2D(df_dim*4,3,(2,2),'same',kernel_constraint=spectral_normalization,name='conv3')
        self.sbn3 = layers.BatchNormalization(gamma_constraint=spectral_normalization,name='bn3')
        self.conv_4 = layers.Conv2D(df_dim*8,3,(2,2),'same',kernel_constraint=spectral_normalization,name='conv4')
        self.sbn4 = layers.BatchNormalization(gamma_constraint=spectral_normalization,name='bn4')
        self.sdense = layers.Dense(1,kernel_constraint=spectral_normalization,name='dense1')
        self.pdense = layers.Dense(7,kernel_constraint=spectral_normalization,name='pdense')
        self.gsp = Global_Sum_Pooling(activation='relu',name='gsp')
        self.inner_prouduct= Inner_Product(name='inp')

    def call(self,inputs,training=None):
        x,c = inputs
        embed_c = spectral_normalization(self.embed(c))
        x = self.conv_1(x)
        x = self.sbn1(x,training)
        x = tf.nn.leaky_relu(x)
        x = self.conv_2(x)
        x = self.sbn2(x,training)
        x = tf.nn.leaky_relu(x)
        x = self.conv_3(x)
        x = self.sbn3(x,training)
        x  = tf.nn.leaky_relu(x)
        x = self.conv_4(x)
        x = self.sbn4(x,training)
        x = tf.nn.leaky_relu(x)
        h = self.gsp(x)
        output_pose = self.pdense(h)
        output_real_fake = self.sdense(h)
        h = self.inner_prouduct([embed_c,h])
        output_real_fake = output_real_fake+h 
        return output_real_fake,output_pose
       
        
class PoseGAN:
    def __init__(self,
                batch_size = 32,
                g_learning_rate = 1e-4,
                d_learning_rate = 1e-4,
                g_loss_beta1 = 10,
                g_loss_beta2 = 0.5,
                tflog_dirs = None,
                traindataset_loader=None,
                num_cls = 2,
                name='PoseGAN'):
        #1.datasetloader
        self.traindataset_loader = traindataset_loader.ReadTrainDatasets(False,False).shuffle(batch_size*10).batch(batch_size,drop_remainder=True).repeat()
        #2.create optimizer
        self.g_opt = tf.keras.optimizers.Adam(g_learning_rate,0,0.9)
        self.d_opt = tf.keras.optimizers.Adam(d_learning_rate,0,0.9)
        #3.create networks
        self.g_net = Generator(num_cls)
        self.d_net = Discriminator(num_cls)
        #4.create tflogdir
        self.train_logdir = tflog_dirs
        if not os.path.isdir(self.train_logdir):
                    os.makedirs(self.train_logdir)
        #5.accept super param
        self.g_loss_beta1 = g_loss_beta1
        self.g_loss_beta2 = g_loss_beta2
    def train_on_batch_g(self,real_img,real_pose,label):
        self.g_net.trainable = True
        self.d_net.trainable = False
        with tf.GradientTape() as tape:
            fake_image = self.g_net([real_pose,label],True)
            fake_logit,pred_pose = self.d_net([fake_image,label],False)
            g_loss = GeneratorLoss(fake_logit,real_img,fake_image,real_pose,pred_pose,self.g_loss_beta1,self.g_loss_beta2)
        gradients = tape.gradient(g_loss,self.g_net.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients,self.g_net.trainable_variables))
        g_out = {
                'images_fake':fake_image,
                'loss':g_loss
        }
        return g_out
    def train_ob_batch_d(self,real_img,real_pose,label):
        self.g_net.trainable = False
        self.d_net.trainable = True
        with tf.GradientTape() as tape:
            fake_image = self.g_net([real_pose,label],False)
            fake_logit,fake_pred_pose = self.d_net([fake_image,label],True)
            real_logit,real_pred_pose = self.d_net([real_img,label],True)
            d_loss = DiscriminatorLoss(fake_logit,real_logit,real_pose,real_pred_pose,fake_pred_pose)
        gradients = tape.gradient(d_loss,self.d_net.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients,self.d_net.trainable_variables))
        d_out = {
            "total_loss":d_loss
        }
        return d_out
    def fit(self,train_after_step_index = 0,total_train_steps=20000):
        #1.create summary
        summary_writer = tf.summary.create_file_writer(self.train_logdir)
        train_step = train_after_step_index%1000
        train_step_idx = train_after_step_index//1000
        print("train_step-->",train_step,train_step_idx)
        for batch_traindata in self.traindataset_loader:
            batch_traindata =  Data_Auguments(batch_traindata,False,False)
            if train_step_idx*1000+train_step ==total_train_steps:
                print("train phase has finished!")
            if train_step ==1000:
                train_step = 0
                train_step_idx +=1
            d_out = self.train_ob_batch_d(batch_traindata[0],batch_traindata[1],batch_traindata[2])
            if train_step%2==0:
                g_out = self.train_on_batch_g(batch_traindata[0],batch_traindata[1],batch_traindata[2])
                print("train_step-->%d, [D loss: %f,G loss:%f]"%(train_step_idx*1000+train_step,d_out['total_loss'],g_out['loss']))
            if train_step %10==0:
                self.g_net.save_weights(self.train_logdir+'/generator.ckpt')
                self.d_net.save_weights(self.train_logdir+'/discriminator.ckpt')
                with summary_writer.as_default():
                        tf.summary.scalar('d/loss', d_out['total_loss'], step=train_step_idx*1000+train_step)
                        tf.summary.scalar('g/loss', g_out['loss'], step=train_step_idx*1000+train_step)
                        tf.summary.image('generated_images', g_out['images_fake'],
                                 step=train_step_idx*1000+train_step, max_outputs=4)
                        tf.summary.image('real_images', batch_traindata[0],
                                 step=train_step_idx*1000+train_step,max_outputs=4)
                        summary_writer.flush()
            train_step+=1 
    def predict(self):
        pass





if __name__=="__main__":
   #0.set GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    #1.create datasetloader
    traindataloader = TfRecordDatasetsLoader(["/home/swx/m_mode_datasets/temp_and_dino_datasets"],["/home/swx/m_mode_datasets/temp_and_dino_datasets/traindatasets_tfrecords"])
    #traindatagenerator = traindataloader.ReadTrainDatasets().shuffle(300).batch(10)
    #2.create networks
    m_Pose_GAN = PoseGAN(tflog_dirs="tflogdirs/train_posegan",traindataset_loader=traindataloader)
    #3.train 
    m_Pose_GAN.fit()




        










        



        



