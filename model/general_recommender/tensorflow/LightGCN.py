"""
Paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Author: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang
Reference: https://github.com/hexiangnan/LightGCN
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["LightGCN"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import inner_product, l2_loss
from util.tensorflow import pairwise_loss, pointwise_loss
from util.tensorflow import get_initializer, get_session
from util.common import Reduction
from data import PairwiseSampler, PointwiseSampler
import numpy as np
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix
from util.tensorflow import sp_mat_to_sp_tensor


class LightGCN(AbstractRecommender):
    def __init__(self, config):
        super(LightGCN, self).__init__(config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']
        self.adj_type = config["adj_type"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self._build_model()
        self.sess = get_session(config["gpu_mem"])

    @timer
    def create_adj_mat(self, adj_type):
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        up_left_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = up_left_adj + up_left_adj.T

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalize_adj_matrix(adj_mat + sp.eye(adj_mat.shape[0]), norm_method="left")
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="left")
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalize_adj_matrix(adj_mat, norm_method="left")
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

    def _create_variable(self):

        self.user_ph = tf.placeholder(tf.int32, shape=(None,))
        self.pos_item_ph = tf.placeholder(tf.int32, shape=(None,))
        self.neg_item_ph = tf.placeholder(tf.int32, shape=(None,))
        self.label_ph = tf.placeholder(tf.float32, shape=(None,))

        self.weights = dict()
        init = get_initializer(self.param_init)
        self.weights['user_embedding'] = tf.Variable(init([self.num_users, self.emb_size]), name='user_embedding')
        self.weights['item_embedding'] = tf.Variable(init([self.num_items, self.emb_size]), name='item_embedding')

    def _build_model(self):
        self._create_variable()
        norm_adj = self.create_adj_mat(self.adj_type)
        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed(norm_adj)

        # predict user-item preference based on the final representations
        user_emb = tf.nn.embedding_lookup(self.ua_embeddings, self.user_ph)
        pos_item_emb = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_item_ph)
        neg_item_emb = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_item_ph)

        yi_hat = inner_product(user_emb, pos_item_emb)
        yj_hat = inner_product(user_emb, neg_item_emb)

        _user_emb = tf.nn.embedding_lookup(self.weights['user_embedding'], self.user_ph)
        _pos_item_emb = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_item_ph)
        _neg_item_emb = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_item_ph)

        reg_loss = l2_loss(_user_emb, _pos_item_emb)
        if self.is_pairwise:
            model_loss = pairwise_loss(self.loss_func, yi_hat - yj_hat, reduction=Reduction.MEAN)
            reg_loss += l2_loss(_neg_item_emb)
        else:
            model_loss = pointwise_loss(self.loss_func, yi_hat, self.label_ph, reduction=Reduction.MEAN)

        final_loss = model_loss + self.reg * reg_loss/self.batch_size

        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(final_loss)

        # for evaluation
        zero_init = get_initializer("zeros")
        # for accelerating evaluation
        self.item_embeddings_final = tf.Variable(zero_init([self.num_items, self.emb_size]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(zero_init([self.num_users, self.emb_size]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.ua_embeddings),
                           tf.assign(self.item_embeddings_final, self.ia_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.user_ph)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)

    def _create_lightgcn_embed(self, norm_adj):
        adj_mat = sp_mat_to_sp_tensor(norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings

    def train_model(self):
        if self.is_pairwise:
            return self._train_pairwise()
        else:
            return self._train_pointwise()

    def _train_pairwise(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        results = []
        for epoch in range(self.epochs):
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.user_ph: bat_users,
                        self.pos_item_ph: bat_pos_items,
                        self.neg_item_ph: bat_neg_items}
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            results.append([result])
            buf = '\t'.join([("%.8f" % x).ljust(12) for x in result])
            self.logger.info("epoch %d:\t%s" % (epoch, buf))
        return results

    def _train_pointwise(self):
        data_iter = PointwiseSampler(self.dataset.train_data, num_neg=1,
                                     batch_size=self.batch_size,
                                     shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        results = []
        for epoch in range(self.epochs):
            for bat_users, bat_items, bat_labels in data_iter:
                feed = {self.user_ph: bat_users,
                        self.pos_item_ph: bat_items,
                        self.label_ph: bat_labels}
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            results.append([result])
            buf = '\t'.join([("%.8f" % x).ljust(12) for x in result])
            self.logger.info("epoch %d:\t%s" % (epoch, buf))
        return results

    def evaluate_model(self):
        self.sess.run(self.assign_opt)  # for accelerating evaluation
        return self.evaluator.my_evaluate(self)

    def predict(self, users, neg_items=None):
        feed_dict = {self.user_ph: users}
        ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
        return ratings
