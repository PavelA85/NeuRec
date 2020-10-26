"""
Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback
Author: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["MF"]

import tensorflow as tf
from model.base import AbstractRecommender
from util.tensorflow import inner_product, l2_loss
from util.tensorflow import pairwise_loss, pointwise_loss
from util.tensorflow import get_initializer, get_session
from util.common import Reduction
from data import PairwiseSampler, PointwiseSampler


# noinspection DuplicatedCode
class MF(AbstractRecommender):
    def __init__(self, config):
        super(MF, self).__init__(config)
        self.emb_size = config["embedding_size"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self._build_model()
        self.sess = get_session(config["gpu_mem"])

    def _create_variable(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.pos_item_ph = tf.placeholder(tf.int32, [None], name="pos_item")
        self.neg_item_ph = tf.placeholder(tf.int32, [None], name="neg_item")
        self.label_ph = tf.placeholder(tf.float32, [None], name="label")

        # embedding layers
        init = get_initializer(self.param_init)
        zero_init = get_initializer("zeros")
        self.user_embeddings = tf.Variable(init([self.num_users, self.emb_size]),
                                           name="user_embedding")
        self.item_embeddings = tf.Variable(init([self.num_items, self.emb_size]),
                                           name="item_embedding")
        self.item_biases = tf.Variable(zero_init([self.num_items]), name="item_bias")

    def _build_model(self):
        self._create_variable()
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)
        pos_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.pos_item_ph)
        neg_item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.neg_item_ph)
        pos_bias = tf.gather(self.item_biases, self.pos_item_ph)
        neg_bias = tf.gather(self.item_biases, self.neg_item_ph)

        yi_hat = inner_product(user_emb, pos_item_emb) + pos_bias
        yj_hat = inner_product(user_emb, neg_item_emb) + neg_bias

        # reg loss
        reg_loss = l2_loss(user_emb, pos_item_emb, pos_bias)
        if self.is_pairwise:
            model_loss = pairwise_loss(self.loss_func, yi_hat - yj_hat, reduction=Reduction.SUM)
            reg_loss += l2_loss(neg_item_emb, neg_bias)
        else:
            model_loss = pointwise_loss(self.loss_func, yi_hat, self.label_ph, reduction=Reduction.SUM)

        final_loss = model_loss + self.reg * reg_loss

        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss, name="train_opt")

        # for evaluation
        u_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)
        self.batch_ratings = tf.matmul(u_emb, self.item_embeddings, transpose_b=True) + self.item_biases

    def train_model(self):
        if self.is_pairwise:
            return self._train_pairwise()
        else:
            return self._train_pointwise()

    def _train_pairwise(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        results = []
        self.logger.info(self.evaluator.metrics_info())
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
        results = []
        self.logger.info(self.evaluator.metrics_info())
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
        return self.evaluator.my_evaluate(self)

    def predict(self, users):
        all_ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_ph: users})
        return all_ratings
