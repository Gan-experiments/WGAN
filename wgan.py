import tensorflow as tf
import numpy as np
import random
import datetime
from utils import*
import os 
from tfrecords_reader import TFRecordsReader

batch_size=64
class batchnorm():
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,decay=self.momentum,updates_collections=None,epsilon=self.epsilon,scale=True,is_training=train,scope=self.name)
def lrelu(x, leak=0.2):
    return tf.maximum(x,x*leak)

def conv(x,num_filters,kernel=5,stride=[1,2,2,1],name="conv",padding='SAME'):
    with tf.variable_scope(name):
        w=tf.get_variable('w',shape=[kernel,kernel,x.get_shape().as_list()[3], num_filters],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b=tf.get_variable('b',shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        con=tf.nn.conv2d(x, w, strides=stride, padding=padding)
        return tf.reshape(tf.nn.bias_add(con, b),con.shape)

def fcn(x,num_neurons,name="fcn"):#(without batchnorm )
    with tf.variable_scope(name):

        w=tf.get_variable('w',shape=[x.get_shape().as_list()[1],num_neurons],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b=tf.get_variable('b',shape=[num_neurons],
            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x,w)+b

def deconv(x,output_shape,kernel=5,stride=[1,2,2,1],name="deconv"):
    with tf.variable_scope(name):
        num_filters=output_shape[-1]
        w=tf.get_variable('w',shape=[kernel,kernel, num_filters,x.get_shape().as_list()[3]],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b=tf.get_variable('b',shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        decon=tf.nn.conv2d_transpose(x, w, strides=stride,output_shape=output_shape)
        return tf.reshape(tf.nn.bias_add(decon, b),decon.shape)

def generator(z,y):
    with tf.variable_scope("generator"):
       # y_onehot =    tf.one_hot(y,10)
        
        z_       =    tf.concat([z,y],1)

        h0       =    fcn(z_,num_neurons=512*4*4,name="g_fcn0")

        gbn0     =    batchnorm(name="g_bn0")
        
        h0       =    tf.nn.relu(gbn0(tf.reshape(h0,[batch_size,4,4,512]),train=True))

        gbn1     =    batchnorm(name="g_bn1")

        h1       =    tf.nn.relu(gbn1(deconv(h0,[batch_size,7,7,256],name="g_h1"),train=True))

        gbn2     =    batchnorm(name="g_bn2")

        h2       =    tf.nn.relu(gbn2(deconv(h1,[batch_size,14,14,128],name="g_h2"),train=True))

        h4       =    deconv(h2,[batch_size,28,28,1],name="g_h4")
        
        return tf.nn.tanh(h4)

def sampler(z,y):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()
#        y_onehot =    tf.one_hot(y,10)
        
        z_       =    tf.concat([z,y],1)

        h0       =    fcn(z_,num_neurons=512*4*4,name="g_fcn0")

        gbn0     =    batchnorm(name="g_bn0")
        
        h0       =    tf.nn.relu(gbn0(tf.reshape(h0,[batch_size,4,4,512]),train=False))

        gbn1     =    batchnorm(name="g_bn1")

        h1       =    tf.nn.relu(gbn1(deconv(h0,[batch_size,7,7,256],name="g_h1"),train=False))

        gbn2     =    batchnorm(name="g_bn2")

        h2       =    tf.nn.relu(gbn2(deconv(h1,[batch_size,14,14,128],name="g_h2"),train=False))

 #       gbn3     =    batchnorm(name="g_bn3")

  #      h3       =    tf.nn.relu(gbn3(deconv(h2,[batch_size,16*2,16*2,64],name="g_h3"),train=False))

 #       gbn4     =    batchnorm(name="g_bn4") 

        h4       =    deconv(h2,[batch_size,28,28,1],name="g_h4")
        
        return tf.nn.tanh(h4)

def discriminator(imgs,reuse=False):

    with tf.variable_scope("discriminator") as scope:

        if reuse:

            scope.reuse_variables()

        h0       =    lrelu(conv(imgs,64,name="d_h0"))

        dbn1     =    batchnorm(name="d_bn1")
        
        h1       =    lrelu(dbn1(conv(h0,128,name="d_h1")))

        dbn2     =    batchnorm(name="d_bn2")

        h2       =    lrelu(dbn2(conv(h1,128*2,name="d_h2")))

        dbn3     =    batchnorm(name="d_bn3")

        h3       =    lrelu(dbn3(conv(h1,128*2*2,name="d_h3")))

        h4       =    tf.reshape(h3,[batch_size,-1])

        source   =    fcn(h4,1,name="d_source")

        return source

def load_mnist():
    data_dir = os.path.join("./data-1", "mnist")
   # data_dir="/home/satwik/Desktop/swaayatt_satwik/gan_test_Code/data /mnist"
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
	y_vec[i,y[i]] = 1.0
    
    return (X/127.5)-1,y_vec

images  =   tf.placeholder(tf.float32, [batch_size, 28,28,1], name='images')

labels      =   tf.placeholder(tf.float32, [batch_size, 10], name='labels')

z      =   tf.placeholder(tf.float32, [batch_size, 100], name='z')

X,Y    =   load_mnist()

with tf.device('/gpu:0'):

   # labels_one_hot  =   tf.one_hot(labels,10)
    
    gen_imgs        =   generator(z=z,y=labels)

    real_source     =   discriminator(images,reuse=False)

    fake_source		=	discriminator(gen_imgs,reuse=True)

    samples 			= 	sampler(z=z,y=labels)

    d_loss			=	tf.reduce_mean(real_source) - tf.reduce_mean(fake_source)

    g_loss 			=	tf.reduce_mean(fake_source)

    t_vars			=	tf.trainable_variables()

    d_vars			=	[var for var in t_vars if 'd_' in var.name]

    g_vars			=	[var for var in t_vars if 'g_' in var.name]

    d_opt			=	tf.train.RMSPropOptimizer(learning_rate=0.00005,decay=0.9).minimize(d_loss,var_list=d_vars)

    g_opt  			= 	tf.train.RMSPropOptimizer(learning_rate=0.00005,decay=0.9).minimize(g_loss,var_list=g_vars)

    d_clip			=	[var.assign(tf.clip_by_value(var,-0.02,0.02)) for var in d_vars]

    init   = tf.global_variables_initializer()

    config = tf.ConfigProto()

    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    with tf.Session(config=config) as sess:

    	sess.run(init)

	#tf.train.start_queue_runners(sess=sess)

    	for epoch in range(10000):

		batch_x = X[(epoch%batch_size)*batch_size :(epoch%batch_size+1)*batch_size ]

		batch_y = Y[(epoch%batch_size)*batch_size :(epoch%batch_size+1)*batch_size ]

    		for n in range(5):

    			z_=np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

    			sess.run(d_opt,feed_dict={z:z_,images:batch_x.reshape([batch_size,28,28,1]),labels:batch_y})

    			sess.run(d_clip,feed_dict={z:z_,images:batch_x.reshape([batch_size,28,28,1]),labels:batch_y})

    		z_=np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

    		sess.run(g_opt,feed_dict={z:z_,labels:batch_y})

    		D_loss = sess.run(d_loss,feed_dict={z:z_,images:batch_x.reshape([batch_size,28,28,1]),labels:batch_y })

    		G_loss = sess.run(g_loss,feed_dict={z:z_,labels:batch_y })

    		print "d loss after epoch ",epoch," is ",D_loss

    		print "g loss after epoch ",epoch," is ",G_loss

    		if epoch % 10 ==0:
                
                	sample = sess.run(samples,feed_dict={z:z_,labels:batch_y})

               	 	save_images(sample, image_manifold_size(sample.shape[0]),
                          './{}/{:02d}.png'.format("w_out", epoch))







