#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import time
import chainer


class Controller(object):
    def __init__(self):
        self.iter_cnt = 0
        self.sum_accuracy = 0
        self.sum_loss = 0
        self.st = time.time()
        chainer.config.train = True
        chainer.config.enable_backprop = True

    def next_epoch(self):
        self.valid_loss = self.sum_loss / self.valid_count
        self.valid_acc = self.sum_accuracy / self.valid_count
        self.sum_accuracy = 0
        self.sum_loss = 0
        self.elapsed = time.time() - self.st
        self.st = time.time()
        chainer.config.train = True
        chainer.config.enable_backprop = True

    def validate(self):
        self.train_loss = self.sum_loss / self.train_count
        self.train_acc = self.sum_accuracy / self.train_count
        self.sum_accuracy = 0
        self.sum_loss = 0
        chainer.config.train = False
        chainer.config.enable_backprop = False

    def update(self, batch_size, loss, acc=None):
        sum_loss += loss * batch_size
        if acc is not None:
            sum_accuracy += acc * batch_size
        self.train_cnt += len(t.data)
        self.iter_cnt += 1

    def update_valid(self, batch_size, loss, acc=None):
        sum_loss += loss * batch_size
        if acc is not None:
            sum_accuracy += acc * batch_size
        self.valid_cnt += len(t.data)
