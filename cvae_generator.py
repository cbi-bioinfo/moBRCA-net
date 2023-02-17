from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from pandas import DataFrame as df


# SET ENV
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44


# FUNCTIONS
def fc_bn(_x, _output, _phase, _scope):
	with tf.variable_scope(_scope):
		h1 = tf.contrib.layers.fully_connected(_x, _output, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.variance_scaling_initializer())
		h2 = tf.contrib.layers.batch_norm(h1, updates_collections=None, fused=True, decay=0.9, center=True, scale=True, is_training=_phase, scope='bn')
		return h2


# READ RAW DATA
filename_x = sys.argv[1]
filename_y = sys.argv[2]
filename_simulation_label = sys.argv[3]
group_num = sys.argv[4]

x_data_temp = pd.read_csv(filename_x, delimiter=",", dtype=np.float32)
cpg_list = x_data_temp.columns
x_data = x_data_temp.values
y_data = pd.read_csv(filename_y, delimiter=",", dtype=np.float32).values
generated_test_Y = pd.read_csv(filename_simulation_label, delimiter=",", dtype = np.float32).values

# RELEASE MEMORY
del(x_data_temp)

n_features = len(x_data[0])
n_classes = len(y_data[0])

# PLACEHOLDER
tf_raw_X = tf.placeholder(tf.float32, [None, n_features])
tf_raw_Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name='phase')
handle = tf.placeholder(tf.string, shape=[])
noise_r = tf.placeholder(tf.float32, shape=[])

distributions = tf.distributions

# PARAMETERS
batch_size = 500 
ep_repeat = 10 #REPLATED TO PRINT FREQUENCY
prefetch_size = batch_size * 2

learn_rate_vae = 1e-3
keep_rate_vae = 1
repeat_num = 1
train_vae_eps = 10000


# DATASET & ITERATOR
dataset_train = tf.data.Dataset.from_tensor_slices((tf_raw_X, tf_raw_Y))
dataset_train = dataset_train.shuffle(buffer_size=batch_size * 2)
dataset_train = dataset_train.repeat(ep_repeat).batch(batch_size).prefetch(prefetch_size)
iterator_train = dataset_train.make_initializable_iterator()

iter = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)

tf_X, tf_Y = iter.get_next()

# MODEL STRUCTURE
n_in = n_features
n_en_h1 = 500
n_en_h2 = 250
n_code_h = 125
n_de_h1 = 250
n_de_h2 = 500
n_out = n_features


print("# feature:", n_features, "# train sample:", len(x_data))

# MODEL FUNCTIONS

# gaussian MLP encoder
def encoder(_X, _Y,_keep_prob, _phase):
	dim_y = int(_Y.get_shape()[1])
	#concatenate condition and input
	input = tf.concat(axis = 1, values = [_X, _Y])
	en1 = tf.nn.dropout(tf.nn.elu(fc_bn(input, n_en_h1 + dim_y, _phase, "en1")), _keep_prob)
	en2 = tf.nn.dropout(tf.nn.tanh(fc_bn(en1, n_en_h2, _phase, "en2")), _keep_prob)
	gaussian_params = fc_bn(en2, n_code_h * 2, _phase, "gaussian_params")
	mu = gaussian_params[:, :n_code_h]
	#The standard deviation must be positive. Parameterize with a softplus and add a small epsilon for numerical stability
	sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, n_code_h:]) 
	return mu, sigma


def decoder(_code, _Y ,_keep_prob, _phase, _reuse = False):
	with tf.variable_scope("bernoulli_decoder", reuse = _reuse) :
		#concatenate condition and latent vectors
		input = tf.concat(axis = 1, values = [_code, _Y])
		de2 = tf.nn.dropout(tf.nn.tanh(fc_bn(input, n_de_h1, _phase, "de2")), _keep_prob)
		de3 = tf.nn.dropout(tf.nn.elu(fc_bn(de2, n_de_h2, _phase, "de3")), _keep_prob)
		de_out = fc_bn(de3, n_out, _phase, "decode")
		decode = tf.nn.sigmoid(de_out)
		return decode,de_out



# MODEL
mu, sigma = encoder(tf_X, tf_Y, keep_prob, phase)
z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype = tf.float32)
decode,de_out = decoder(z, tf_Y, keep_prob, phase)

# GENERATOR
n_samples = len(generated_test_Y)
z_input = distributions.Normal(loc = np.zeros(n_code_h, dtype = np.float32), scale = np.ones(n_code_h, dtype = np.float32))
z_input_sample = z_input.sample(n_samples)


def generator(_z, _y) :
	x_, _ = decoder(_z, _y, 1.0, False, _reuse = True)
	return x_


x_generated = generator(z_input_sample, tf_Y)


print ("MODEL READY") 


# DEFINE LOSS AND OPTIMIZER
marginal_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=de_out, labels = tf_X), 1)
KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) -1, 1)

marginal_likelihood = tf.reduce_mean(marginal_likelihood)
KL_divergence = tf.reduce_mean(KL_divergence)
vae_cost = -1 * marginal_likelihood + KL_divergence
train_op_vae = tf.train.AdamOptimizer(learning_rate=learn_rate_vae).minimize(vae_cost)
_accuracy = tf.Variable(0)

print ("FUNCTIONS READY") 

# START SESSION
sess = tf.Session(config=config)
handle_train = sess.run(iterator_train.string_handle())
saver = tf.train.Saver()

print ("START OPTIMIZATION & TESTING")
for model_num in xrange(repeat_num):
	sess.run(tf.global_variables_initializer())	

	# SET OPS & FEED_DICT
	vae_ops = [vae_cost, train_op_vae, _accuracy, KL_divergence, z]
	vae_feed_dict = {handle: handle_train, keep_prob : keep_rate_vae, phase: True}

	for temp_ep, meta_step, temp_ops, temp_feed_dict in zip([train_vae_eps], ["_vae"], [vae_ops], [vae_feed_dict]):	
		for ep in xrange(temp_ep/ep_repeat):
			sess.run(iterator_train.initializer, feed_dict={tf_raw_X: x_data, tf_raw_Y: y_data})

			# REPEAT NUMBER OF EPS WITHOUT BREAK. SET BY ep_repeat
			while True: 
				try:
					cur_cost_val, _, cur_accuracy, loss_divergence, cur_z  = sess.run(temp_ops, feed_dict = temp_feed_dict)
						
				except tf.errors.OutOfRangeError:
					break

			# EXECUTED PER ep_repeat
			print("Model#:%02d," % model_num, "Ep:%04d," % (ep*ep_repeat), "Cost" + meta_step + ":%.9f" % cur_cost_val, end='\n')



	final_generated_x = sess.run(x_generated, feed_dict = {tf_Y : generated_test_Y})
	simulation_X = pd.DataFrame(data = final_generated_x, columns = cpg_list)
	simulation_X.to_csv("generation_split_num_" + group_num + ".csv", mode = 'w', index = False)
	print("")
