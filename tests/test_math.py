import pytest
import sys, os
from types import SimpleNamespace

import tensorflow as tf

import numpy as np

sys.path.append(os.path.abspath('../src'))
from network import Network
import constants

@pytest.fixture
def network():
	mock_args_dict = {'languages': ['en'],
	                  'tasks': ['dep_distance', 'dep_depth'],
	                  'out_dir': '',
	                  'layer_index': 6,
	                  'ml_probe': 1024,
	                  'seed': 42,
	                  'batch_size': 20,
	                  'epochs': 40,
	                  'learning_rate': 0.001,
	                  'ortho': 1.,
	                  'l2': None,
	                  'clip_norm': 30.,
	                  'bert_path': "bert-{}{}-{}".format(constants.SIZE_LARGE, constants.LANGUAGE_ENGLISH, constants.CASING_CASED)}
	mock_args = SimpleNamespace(**mock_args_dict)

	return Network(mock_args)


def test_orthogonal_contraint(network):
	identity_small = tf.Variable(tf.initializers.Identity(gain=1.0)((5, 5)))

	tf.debugging.assert_near(network.probe.ortho_reguralization(identity_small), 0.)

	identity_big = tf.Variable(tf.initializers.Identity(gain=1.0)((500, 500)))

	tf.debugging.assert_near(network.probe.ortho_reguralization(identity_big), 0.)

	ones_matrix = tf.ones([5,5])

	tf.debugging.assert_none_equal(network.probe.ortho_reguralization(identity_big), 0.)
	tf.debugging.assert_positive(network.probe.ortho_reguralization(identity_big))

def test_loss_distance(network):

	target = tf.fill([3,3,3], 5.)
	mask_ones = tf.fill([3,3,3], 1.)
	mask_zeros = tf.fill([3,3,3], 0.)
	token_lengths = tf.constant([3,3,3])

	pred_A = tf.fill([3,3,3], 5.)
	pred_B = tf.fill([3,3,3], 4.)

	tf.debugging.assert_near(network.distance_probe._loss(target, pred_A, mask_ones, token_lengths), 0.)
	tf.debugging.assert_near(network.distance_probe._loss(target, pred_B, mask_ones, token_lengths), 3.)

	tf.debugging.assert_near(network.distance_probe._loss(target, pred_A, mask_zeros, token_lengths), 0.)
	tf.debugging.assert_near(network.distance_probe._loss(target, pred_B, mask_zeros, token_lengths), 0.)


def test_loss_depth(network):

	target = tf.fill([3,3], 5.)
	mask_ones = tf.fill([3,3], 1.)
	mask_zeros = tf.fill([3,3], 0.)
	token_lengths = tf.constant([3,3,3])

	pred_A = tf.fill([3,3], 5.)
	pred_B = tf.fill([3,3], 4.)

	tf.debugging.assert_near(network.depth_probe._loss(target, pred_A, mask_ones, token_lengths), 0.)
	tf.debugging.assert_near(network.depth_probe._loss(target, pred_B, mask_ones, token_lengths), 3.)

	tf.debugging.assert_near(network.depth_probe._loss(target, pred_A, mask_zeros, token_lengths), 0.)
	tf.debugging.assert_near(network.depth_probe._loss(target, pred_B, mask_zeros, token_lengths), 0.)
