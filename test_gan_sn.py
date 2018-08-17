# implement GAN with spectral normalization (SN-GAN)
# see Mayato et al. 2018
# [DONE] test source generation
# [DONE] save data to matfile
# [DONE] replace constellation with 16-QAM
# [DONE] use partial() function

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
root_logdir = "gan_sn_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# param struct
class params:
    bname = "gan"
#    mode = "normal"
    mode = "spec_norm"
    lip_norm = 3
    name = bname + '_' + mode
#    mode = "spec_norm"
    mog_stddev = 0.01
    x_dim=2
    z_dim=2
    h_dim_gen=256
    h_dim_disc=256
    n_layers_gen=3
    n_layers_disc=3
#    kernel_init = tf.orthogonal_initializer(gain=1.4)
#    kernel_init = tf.initializers.random_normal(stddev=0.2)
    kernel_init = tf.contrib.layers.variance_scaling_initializer() # He et al. 2015
    disc_learning_rate=1e-4
    gen_learning_rate=1e-4
    beta1=0.5
    beta2=0.9
    n_disc = 5
#    epsilon=1e-8
#    n_steps=50000
    n_steps=25000
#    n_steps=10000
#    n_steps=10
#    batch_size=512
    batch_size=256
    vis_every = 10000
    vis_every = 5000
    log_every = 1000
    log_every = 500
    trace_every = 5
    n_trace = int(n_steps / trace_every) + 1

p = params

# define generators and disciminator
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

def discriminator(x, n_hidden, n_layer, reuse=False):
    n_inputs = int(x.get_shape()[1])
    n_outputs = 1
    dense_hidden_layer = partial(tf.layers.dense, activation=tf.nn.tanh, 
                                 kernel_initializer=p.kernel_init)
    with tf.variable_scope("discriminator", reuse=reuse):
        if n_layer >= 1: h = dense_hidden_layer(x, n_hidden, name="hidden1")
        if n_layer >= 2: h = dense_hidden_layer(h, n_hidden, name="hidden2")
        if n_layer >= 3: h = dense_hidden_layer(h, n_hidden, name="hidden3")
        if n_layer >= 4: h = dense_hidden_layer(h, n_hidden, name="hidden4")
        logits = tf.layers.dense(h, n_outputs, name="outputs")

    with tf.variable_scope("spec_norm", reuse=reuse):
        if n_layer >= 1: u = tf.Variable(tf.random_normal(shape=(n_hidden,1)), name="left1")
        if n_layer >= 2: u = tf.Variable(tf.random_normal(shape=(n_hidden,1)), name="left2")
        if n_layer >= 3: u = tf.Variable(tf.random_normal(shape=(n_hidden,1)), name="left3")
        if n_layer >= 4: u = tf.Variable(tf.random_normal(shape=(n_hidden,1)), name="left4")
        u = tf.Variable(tf.random_normal(shape=(n_outputs,1)), name="lefto")

        # [TODO] storage not needed for the right singular vectors
        if n_layer >= 1: v = tf.Variable(tf.zeros(shape=(n_inputs,1)), name="right1")
        if n_layer >= 2: v = tf.Variable(tf.zeros(shape=(n_hidden,1)), name="right2")
        if n_layer >= 3: v = tf.Variable(tf.zeros(shape=(n_hidden,1)), name="right3")
        if n_layer >= 4: v = tf.Variable(tf.zeros(shape=(n_hidden,1)), name="right4")
        v = tf.Variable(tf.random_normal(shape=(n_hidden,1)), name="righto")

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

real_score = discriminator(x_samps, p.h_dim_disc, p.n_layers_disc)
fake_score = discriminator(x_dec_samps, p.h_dim_disc, p.n_layers_disc, reuse=True)

# loss functions (Jenson-Shannon)
with tf.name_scope("loss_disc"):
    # NOTE: cross entropy is defined as [for x \in {0,1}, x ~ p(x)]
    #       H(p,q) = - mean_x (x * log(q(x)) + (1-x) * log(1-q(x)))
    #       maximize GAN objective == minimize cross entropy (as defined below)
    loss_disc = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, 
                                                labels=tf.ones_like(real_score)) +
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, 
                                                labels=tf.zeros_like(fake_score)))

with tf.name_scope("loss_gen"):
    loss_gen  = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, 
                                                labels=tf.ones_like(fake_score)))

# saddle objective
#with tf.name_scope("saddle_loss"):
#    # NOTE: cross entropy is defined as [for x \in {0,1}, x ~ p(x)]
#    #       H(p,q) = - mean_x (x * log(q(x)) + (1-x) * log(1-q(x)))
#    #       maximize GAN objective == minimize cross entropy (as defined below)
#    loss = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=encoder_logits, 
#                                                labels=tf.ones_like(encoder_logits)) +
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_logits, 
#                                                labels=tf.zeros_like(decoder_logits)))

# define variable lists
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen_decoder")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

# python 3.6 needs regex patterns in raw strings (r"xxx")
gen_kernels = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, r"gen_decoder/\w+/kernel")
disc_kernels = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, r"discriminator/\w+/kernel")

left_vecs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, r"spec_norm/left\w")
right_vecs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, r"spec_norm/right\w")

# training 
with tf.name_scope("train_gen"):
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=p.gen_learning_rate, beta1=p.beta1, beta2=p.beta2)
    train_gen = optimizer_gen.minimize(loss_gen, var_list=gen_vars)

with tf.name_scope("train_disc"):
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=p.disc_learning_rate, beta1=p.beta1, beta2=p.beta2)
    train_disc = optimizer_disc.minimize(loss_disc, var_list=disc_vars)

with tf.name_scope("sn_updates"):
    # NOTE: W * X + b is implemented as 
    #       tf.matmul(X,W) + b so W is transposed
    n_kernels = len(disc_kernels)
    for i in range(n_kernels):
        proj_u = tf.matmul(disc_kernels[i], left_vecs[i])
        r_vec = tf.assign(right_vecs[i], proj_u / tf.norm(proj_u), name="right_upd"+str(i))
        #tf.add_to_collection("right_upd_ops", r_vec)
        proj_v = tf.matmul(tf.transpose(disc_kernels[i]), r_vec)
        l_vec = tf.assign(left_vecs[i], proj_v / tf.norm(proj_v), name="left_upd"+str(i))
        #tf.add_to_collection("left_upd_ops", l_vec)

        def quad_form(M, u, v): # implement u' * M * v
            return tf.matmul(v, tf.matmul(M,u), transpose_a=True)

        sigma_kernel = quad_form(disc_kernels[i], l_vec, r_vec)
        tf.add_to_collection("sigma_kernel_ops", sigma_kernel)
        scale_factor = p.lip_norm / sigma_kernel
        M_upd = tf.assign(disc_kernels[i], disc_kernels[i] * scale_factor, name="kernel_upd"+str(i))
        tf.add_to_collection("kernel_upd_ops", M_upd)

        def compute_sigma(M, u):
            proj_u = tf.matmul(M, u)
            v_til = proj_u / tf.norm(proj_u)
            proj_v = tf.matmul(tf.transpose(M), v_til)
            u_til = proj_v / tf.norm(proj_v)
            return quad_form(M, u_til, v_til)

        sigma_kernel_upd = quad_form(M_upd, l_vec, r_vec)
        tf.add_to_collection("sigma_kernel_upd_ops", sigma_kernel_upd)

#right_upd_ops = tf.get_collection("right_upd_ops")
#left_upd_ops = tf.get_collection("left_upd_ops")
kernel_upd_ops = tf.get_collection("kernel_upd_ops")
sigma_kernel_ops = tf.get_collection("sigma_kernel_ops")
sigma_kernel_upd_ops = tf.get_collection("sigma_kernel_upd_ops")
n_sigmas = len(sigma_kernel_ops)

#sn_update_ops = [right_upd_ops, left_upd_ops, kernel_upd_ops]
sn_update_ops = kernel_upd_ops

writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
writer.flush()
print('writing graph to directory:', logdir)

#exit()

# execute model
################################################################################
init = tf.global_variables_initializer()
saver = tf.train.Saver()

import itertools
loss_vec = np.zeros(p.n_trace)
loss_it = itertools.count()

enc_samples_list = []
dec_samples_list = []
lim = 1.5
n_batches_vis = 20
#debug_en = True
debug_level = 1
save_data = True
save_state = True

print("sim mode =", p.mode)

with tf.Session() as sess:
    sess.run(init)

    for i in range(p.n_steps+1):

        for ci in range(p.n_disc):
            if p.mode == "normal":
                sess.run([train_disc])
            elif p.mode == "spec_norm":
                sess.run([sn_update_ops, train_disc])
            else:
                raise Exception("unknown mode")

        loss_d, loss_g, _ = sess.run([loss_disc, loss_gen, train_gen])

        if i % p.trace_every == 0:
            loss_vec[next(loss_it)] = loss_d

        if i % p.log_every == 0:
            print('step %5d loss_disc = %.6f loss_gen = %.6f' % (i, loss_d, loss_g))
            if i % p.vis_every == 0:
                #enc_samples = np.vstack([sess.run(z_enc_samps) for _ in range(n_batches_vis)])
                dec_samples = np.vstack([sess.run(x_dec_samps) for _ in range(n_batches_vis)])
                #enc_samples_list.append(enc_samples)
                dec_samples_list.append(dec_samples)

        if debug_level >= 2 and i % 50 == 0:
            sigma_kernel_vals = sess.run(sigma_kernel_ops)
            # debug spectral norm (matrix l2 norm)
            val_str = '{:.6f} ' * n_sigmas
            format_str = 'before = ( ' + val_str + ')'
            sigma_kernel_floats = [np.squeeze(x) for x in sigma_kernel_vals ]
            print(format_str.format(*sigma_kernel_floats))

            sigma_kernel_upd_vals = sess.run(sigma_kernel_upd_ops)
            val_str = '{:.6f} ' * n_sigmas
            format_str = 'after  = ( ' + val_str + ')'
            sigma_kernel_upd_floats = [np.squeeze(x) for x in sigma_kernel_upd_vals ]
            print(format_str.format(*sigma_kernel_upd_floats))

    # generate source data
    src_size = p.batch_size * n_batches_vis
    x_data = mixture.sample(src_size)
    x_src = sess.run(x_data)

    if save_state:
#        sname = "./wgan_model_final.ckpt"
        sname = './' + p.name + '.ckpt'
        save_path = saver.save(sess, sname)
        print('saved model to file:',sname)

    if save_data:
        # save data to matfile
        import scipy.io as sio
        data_dict = {}
        data_dict['params'] = params
        #data_dict['enc_samples_list'] = enc_samples_list
        data_dict['dec_samples_list'] = dec_samples_list
        data_dict['x_src'] = x_src
        data_dict['loss_vec'] = loss_vec
        fname = p.name + '.mat'
        sio.savemat(fname, data_dict)
        print('saved data to file:',fname)
        # data_dict = sio.loadmat(fname)

    if debug_level >= 1:
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
                ax.set_ylabel('GAN decoder')
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


