# implement Wasserstein GAN with gradient penalty
# see Gulrajani et al. 2017
# [DONE] test source generation
# [DONE] save data to matfile
# [DONE] replace constellation with 16-QAM
# [DONE] updated to use tf v1.10
# [N/A] use kernel_constraint option in layer.dense()
# [WIP] use gradient penalty for critic

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow_probability import distributions as tfd
#from keras.constraints import Constraint

import comm_utils as dhc

# helper functions
from sys import exit
from functools import partial
from datetime import datetime

now = datetime.now().strftime("%Y%m%d-%H%M%S")
root_logdir = "wgan_gp_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# param struct
class params:
    mog_stddev = 0.01
    x_dim=2
    z_dim=2
    h_dim_gen=256
    h_dim_crit=256
    n_layers_gen=3
    n_layers_crit=2
#    kernel_init = tf.orthogonal_initializer(gain=1.4)
#    kernel_init = tf.initializers.random_normal(stddev=0.01)
    kernel_init = tf.contrib.layers.variance_scaling_initializer() # He et al. 2015
#    clamp_val = 0.01
#    clamp_hi =  clamp_val
#    clamp_lo = -clamp_val
    lambda_p = 0.1
    crit_learning_rate=1e-4
#    crit_learning_rate=5e-5
    gen_learning_rate=1e-4
#    gen_learning_rate=5e-5
#    beta1=0.8
    beta1=0.5
    beta2=0.9
#    epsilon=1e-8
    n_critic = 5
#    n_steps=50000
    n_steps=25000
#    n_steps=10000
#    n_steps=10
#    batch_size=512
#    batch_size=128
    batch_size=256
#    vis_every = 10000
    vis_every = 5000
#    log_every = 1000
    log_every = 500

p = params

# define constraint callable
#class clamp_weights(Constraint):
#    def __init__(self, min_value=-1, max_value=1):
#        self.max_value = max_value
#        self.min_value = min_value
#    def __call__(self, w):
#        return tf.clip_by_value(w, self.min_value, self.max_value)
#    def get_config(self):
#        return {'max_value': self.max_value, 'min_value': self.min_value}

# define generators and critic
################################################################################
#def gen_encoder(x, output_dim, n_hidden, n_layer):
#    dense_hidden_layer = partial(tf.layers.dense, activation=tf.nn.tanh, 
#                                 kernel_initializer=p.kernel_init)
#    with tf.variable_scope("gen_encoder"):
#        if n_layer >= 1: h = dense_hidden_layer(x, n_hidden, name="hidden1")
#        if n_layer >= 2: h = dense_hidden_layer(h, n_hidden, name="hidden2")
#        if n_layer >= 3: h = dense_hidden_layer(h, n_hidden, name="hidden3")
#        if n_layer >= 4: h = dense_hidden_layer(h, n_hidden, name="hidden4")
#        z = tf.layers.dense(h, output_dim, name="outputs")
#    return z

def gen_decoder(z, output_dim, n_hidden, n_layer):
    dense_hidden_layer = partial(tf.layers.dense, activation=tf.nn.tanh, 
                                 kernel_initializer=p.kernel_init)
    with tf.variable_scope("gen_decoder"):
        if n_layer >= 1: h = dense_hidden_layer(z, n_hidden, name="hidden1")
        if n_layer >= 2: h = dense_hidden_layer(h, n_hidden, name="hidden2")
        if n_layer >= 3: h = dense_hidden_layer(h, n_hidden, name="hidden3")
        if n_layer >= 4: h = dense_hidden_layer(h, n_hidden, name="hidden4")
        x = tf.layers.dense(h, output_dim, name="outputs")
    return x

def critic(x_z, n_hidden, n_layer, reuse=False):
    dense_hidden_layer = partial(tf.layers.dense, activation=tf.nn.tanh, 
                                 kernel_initializer=p.kernel_init)
#    dense_output_layer = partial(tf.layers.dense, 
#                            kernel_constraint=clamp_weights(p.clamp_lo, p.clamp_hi), 
#                            bias_constraint=clamp_weights(p.clamp_lo, p.clamp_hi))
    dense_output_layer = partial(tf.layers.dense)
    with tf.variable_scope("critic", reuse=reuse):
        if n_layer >= 1: h = dense_hidden_layer(x_z, n_hidden, name="hidden1")
        if n_layer >= 2: h = dense_hidden_layer(h,   n_hidden, name="hidden2")
        if n_layer >= 3: h = dense_hidden_layer(h,   n_hidden, name="hidden3")
        if n_layer >= 4: h = dense_hidden_layer(h,   n_hidden, name="hidden4")
        logits = dense_output_layer(h, 1, name="outputs")
    return logits

# construct model, loss and training steps
################################################################################
tf.reset_default_graph()

# symbols generation
with tf.variable_scope("data"):
#    consts = dhc.get_const_symbols()
    consts = dhc.get_const_symbols(mode='PSK', m=8)
    means = np.c_[consts.real, consts.imag]
    n_comps = len(consts)
    cat = tfd.Categorical(tf.zeros(n_comps)) # equal probabilities
    comps = [tfd.MultivariateNormalDiag(list(means[i]), [p.mog_stddev, p.mog_stddev]) for i in range(n_comps)]
    mixture = tfd.Mixture(cat, comps)
    data  = mixture.sample(p.batch_size)

with tf.variable_scope("noise"):
    noise = tfd.Normal(tf.zeros(p.z_dim),tf.ones(p.z_dim)).sample(p.batch_size)

#x_samps = tf.placeholder(tf.float32, shape=(None, p.x_dim), name="data")
#z_samps = tf.placeholder(tf.float32, shape=(None, p.z_dim), name="noise")
x_samps = data
z_samps = noise

#z_enc_samps = gen_encoder(x_samps, p.z_dim, p.h_dim_gen, p.n_layers_gen)
x_dec_samps = gen_decoder(z_samps, p.x_dim, p.h_dim_gen, p.n_layers_gen)

# implement f(x) and f(g(z))
real_score = critic(x_samps, p.h_dim_crit, p.n_layers_crit)
fake_score = critic(x_dec_samps, p.h_dim_crit, p.n_layers_crit, reuse=True)

# loss functions (earth-mover distance)
with tf.name_scope("loss_crit"):
    loss_crit = tf.reduce_mean(fake_score - real_score)

with tf.name_scope("loss_gen"):
    loss_gen  = - tf.reduce_mean(fake_score)

# define gradient penalty
with tf.name_scope("grad_penalty"):
    alpha = tf.random_uniform(shape=[p.batch_size, 1], minval=0., maxval=1.)
    x_interp = alpha*x_samps + (1-alpha)*x_dec_samps
    crit_interp = critic(x_interp, p.h_dim_crit, p.n_layers_crit, reuse=True)
    gradients = tf.gradients(crit_interp, [x_interp])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
    loss_crit += p.lambda_p*gradient_penalty

# define variable lists
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen_decoder")
crit_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "critic")

# training 
with tf.name_scope("train_gen"):
#    optimizer_gen = tf.train.AdamOptimizer(learning_rate=p.gen_learning_rate, beta1=p.beta1, epsilon=p.epsilon)
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=p.gen_learning_rate, beta1=p.beta1, beta2=p.beta2)
#    optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=p.gen_learning_rate)
    train_gen = optimizer_gen.minimize(loss_gen, var_list=gen_vars)


with tf.name_scope("train_crit"):
#    optimizer_crit = tf.train.AdamOptimizer(learning_rate=p.crit_learning_rate, beta1=p.beta1, epsilon=p.epsilon)
    optimizer_crit = tf.train.AdamOptimizer(learning_rate=p.crit_learning_rate, beta1=p.beta1, beta2=p.beta2)
#    optimizer_crit = tf.train.RMSPropOptimizer(learning_rate=p.crit_learning_rate)
    train_crit = optimizer_crit.minimize(loss_crit, var_list=crit_vars)
#   perform weight clamping
#    with tf.name_scope("clamp_weights"):
#        clamp_ops = [ tf.assign(var, tf.clip_by_value(var, p.clamp_lo, p.clamp_hi, name="clamp")) for var in crit_vars ]

writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
writer.flush()
print('writing graph to directory:', logdir)

#exit()

# execute model
################################################################################
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#enc_samples_list = []
dec_samples_list = []
lim = 1.5
n_batches_vis = 20
debug_en = True
save_data = False
save_state = False

with tf.Session() as sess:
    sess.run(init)

    for i in range(p.n_steps+1):

        for ci in range(p.n_critic):
            #sess.run([train_crit, clamp_ops])
            sess.run(train_crit)

        loss_c, loss_g, _ = sess.run([loss_crit, loss_gen, train_gen])

        if i % p.log_every == 0:
            print('step', i, "EM dist =", -loss_c, "loss_gen =", loss_g)
            if i % p.vis_every == 0:
                #enc_samples = np.vstack([sess.run(z_enc_samps) for _ in range(n_batches_vis)])
                dec_samples = np.vstack([sess.run(x_dec_samps) for _ in range(n_batches_vis)])
                #enc_samples_list.append(enc_samples)
                dec_samples_list.append(dec_samples)

    # generate source data
    src_size = p.batch_size * n_batches_vis
    x_data = mixture.sample(src_size)
    x_src = sess.run(x_data)

    if save_state:
        sname = "./wgan_model_final.ckpt"
        save_path = saver.save(sess, sname)
        print('saved model to file:',sname)

    if save_data:
        # save data to matfile
        import scipy.io as sio
        data_dict = {}
        #data_dict['enc_samples_list'] = enc_samples_list
        data_dict['dec_samples_list'] = dec_samples_list
        data_dict['x_src'] = x_src
        fname = "wgan.mat"
        sio.savemat(fname, data_dict)
        print('saved data to file:',fname)
        # data_dict = sio.loadmat(fname)

    if debug_en:
        import seaborn as sns
        bg_color_g = sns.color_palette('Greens', n_colors=256)[0]
        bg_color_r = sns.color_palette('Reds', n_colors=256)[0]

        dec_samples_ = dec_samples_list
        #enc_samples_ = enc_samples_list
        cols = len(dec_samples_)
        n_cols = cols + 1
        n_rows = 1
        plt.figure(figsize=(2*n_cols, 2*n_rows))
        for i, samps in enumerate(dec_samples_):
            if i == 0:
                ax = plt.subplot(n_rows,n_cols,1)
                ax.set_ylabel('WGAN decoder')
            else:
                plt.subplot(n_rows,n_cols,i+1)
            ax = sns.kdeplot(samps[:, 0], samps[:, 1], bw=0.1,
                   shade=True, cmap='Greens', n_levels=20)
            ax.axis('equal')
            ax.set_xlim(np.array((-1,1))*lim)
            ax.set_ylim(np.array((-1,1))*lim)
            ax.set_facecolor(bg_color_g)
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*p.vis_every))

        plt.subplot(n_rows,n_cols,n_cols)
        ax = sns.kdeplot(x_src[:, 0], x_src[:, 1], bw=0.1,
               shade=True, cmap='Greens', n_levels=20)
        ax.set_facecolor(bg_color_g)
        plt.xticks([]); plt.yticks([])
        plt.title('target');
        plt.axis('equal')
        plt.xlim(np.array((-1,1))*lim)
        plt.ylim(np.array((-1,1))*lim)

        plt.show(block=False)
        plt.waitforbuttonpress(0) # this will wait for indefinite time
        plt.close('all')


