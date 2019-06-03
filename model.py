import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import preprocessor as prep

def init_model(sess, input_images, steering):
	input_image = tf.reshape(images, [-1, 460, 640, 3])

	norm_1 = tf.layers.batch_normalization(input_image)

	conv_1 = tf.layers.conv3d(
		inputs=norm_1,
		filters=3,
		kernel_size=[5,5]
		padding='same',
		activation=tf.nn.relu
	)

	conv_2 = tf.layers.conv3d(
		inputs=conv_1,
		filters=24,
		kernel_size=[5,5]
		padding='same',
		activation=tf.nn.relu
	)

	conv_3 = tf.layers.conv3d(
		inputs=conv_2,
		filters=36,
		kernel_size=[5,5]
		padding='same',
		activation=tf.nn.relu
	)

	conv_4 = tf.layers.conv3d(
		inputs=conv_3,
		filters=48,
		kernel_size=[3,3]
		padding='same',
		activation=tf.nn.relu
	)

	conv_5 = tf.layers.conv3d(
		inputs=conv_4,
		filters=64,
		kernel_size=[3,3]
		padding='same',
		activation=tf.nn.relu
	)

	dense_input = tf.reshape(inputs=conv_5, [-1, ])

	dense_1 = tf.layers.dense(dense_input, units=1164)
	dense_2 = tf.layers.dense(dense_2, units=100)
	dense_3 = tf.layers.dense(dense_3, units=50)
	dense_4 = tf.layers.dense(dense_4, units=10)
	dense_5 = tf.layers.dense(dense_5, units=1)

	cost  = tf.reduce_sum(tf.pow(predicted_steering - steering, 2)) / 2*len(input_images)
	optimizer = tf.train.AdamOptimizer().minimize()
	return tf.initialize_all_variables(), optimizer, {
		image: input_images
	}


if __name__ == '__main__':
	print('preprocessing data...')
	
	images, steering = prep.process()

	sess = tf.Session()

	operations, optimizer, tensor_dict = init_model(sess, images, steering)

	sess.run(operations)

	for _ in range(50):
		sess.run(optimizer, feed_dict=tensor_dict)