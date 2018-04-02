#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np


def sample_uniform(labels, batch_size, oversample=False):
    n = len(labels)
    labels = np.array(labels)
    label_idxs = np.arange(len(labels))
    if oversample:
        unique, counts = np.unique(labels, return_counts=True)
        nb_max = np.max(counts)
        add_idxs = np.concatenate([np.random.choice(label_idxs[labels == l], nb_max - cnt)
                                  for l, cnt in zip(unique, counts)])
        label_idxs = np.concatenate((label_idxs, add_idxs))
        labels = np.concatenate((labels, labels[add_idxs]))

    nb_batches = (len(labels) - 1) // batch_size + 1
    unique, counts = np.unique(labels, return_counts=True)
    lbls_sorted = np.argsort(counts)[::-1]
    orders = [[] for _ in range(nb_batches)]
    for l, cnt in zip(unique[lbls_sorted], counts[lbls_sorted]):
        lengths = [len(x) for idx, x in enumerate(orders) if len(x) < batch_size]
        bidx_sorted = np.argsort(lengths)
        lbl_subset = label_idxs[labels == l]
        if cnt - cnt // 2 > len(bidx_sorted[cnt // 2:]):
            order_sub = np.random.permutation(range(cnt))
            for lidx in lbl_subset[order_sub]:
                valid_batches = [idx for idx, x in enumerate(orders) if len(x) < batch_size]
                orders[valid_batches[np.random.randint(len(valid_batches))]].append(lidx)
        else:
            selected_random = np.random.choice(bidx_sorted[cnt // 2:], cnt - cnt // 2, replace=False)
            selected = np.concatenate((bidx_sorted[:cnt // 2], selected_random))
            for b, lidx in zip(selected, lbl_subset):
                orders[b].append(lidx)

    return np.concatenate([np.random.permutation(x) for x in orders])[:n]
