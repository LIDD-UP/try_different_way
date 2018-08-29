# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: draw_diagram.py
@time: 2018/8/29
"""

import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np # linear algebra
from matplotlib import pyplot as plt # for plotting graphs
from functools import cmp_to_key
from sklearn import metrics
import math
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)

# draw diagram
def min2(l, default=0.0):
    if len(l) == 0:
        return default
    else:
        return min(l)


def max2(l, default=0.0):
    if len(l) == 0:
        return default
    else:
        return max(l)


def avg2(l, default=0.0):
    if len(l) == 0:
        return default
    else:
        return float(sum(l)) / float(len(l))


def std2(l, default=0.0):
    if len(l) == 0:
        return default
    else:
        return np.std(l)


def histogram_for_non_numerical_series(s):
    d = {}
    for v in s:
        d[v] = d.get(v, 0) + 1
    bin_s_label = list(d.keys())
    bin_s_label.sort()
    bin_s = list(range(0, len(bin_s_label)))
    hist_s = [d[v] for v in bin_s_label]
    bin_s.append(len(bin_s))
    bin_s_label.insert(0, '_')
    return (hist_s, bin_s, bin_s_label)


def plot_hist_with_target3(plt, df, feature, target, histogram_bins=10):
    # reference:
    #    https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
    #    https://matplotlib.org/gallery/api/two_scales.html
    #    https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
    #    https://matplotlib.org/2.0.0/examples/color/named_colors.html
    #    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html
    title = feature
    plt.title(title)
    s = df[feature]
    t = df[target]
    t_max = max(t)
    # get histogram of the feature
    bin_s_label = None
    # fillna with 0.0 or '_N/A_'
    na_cnt = sum(s.isna())
    if na_cnt > 0:
        if True in [type(_) == str for _ in s]:
            print('found %d na in string field %s' % (na_cnt, feature))
            s = s.fillna('_N/A_')
        else:
            print('found %d na in numerical field %s' % (na_cnt, feature))
            s = s.fillna(-1.0)
    try:
        hist_s, bin_s = np.histogram(s, bins=histogram_bins)
    except Exception as e:
        # print('ERROR: failed to draw histogram for %s: %s: %s' % (name, type(e).__name__, str(e)))
        hist_s, bin_s, bin_s_label = histogram_for_non_numerical_series(s)
        # return
    # histogram of target by distribution of feature
    hist_t_by_s_cnt = [0] * (len(bin_s) - 1)
    hist_t_by_s = []
    for i in range(0, (len(bin_s) - 1)):
        hist_t_by_s.append([])
    # get target histogram for numerical feature
    if bin_s_label is None:
        for (sv, tv) in zip(s, t):
            pos = 0
            for i in range(0, len(bin_s) - 1):
                if sv >= bin_s[i]:
                    pos = i
            hist_t_by_s_cnt[pos] += 1
            hist_t_by_s[pos].append(tv)
    else:
        for (sv, tv) in zip(s, t):
            pos = bin_s_label.index(sv) - 1
            hist_t_by_s_cnt[pos] += 1
            hist_t_by_s[pos].append(tv)
        # count avg, to re-sort bin_s and bin_s_label by avg
        hist_t_by_s_avg = [float(avg2(n)) for n in hist_t_by_s]
        # hist_t_by_s_std = [float(std2(n)) for n in hist_t_by_s]
        # hist_t_by_s_adj = list(np.array(hist_t_by_s_avg) + np.array(hist_t_by_s_std))
        hist_t_by_s_adj = hist_t_by_s_avg
        # print('before sort:\n%s\n%s\n%s' % (bin_s, bin_s_label, hist_t_by_s_adj))
        bin_hist_label = list(zip(bin_s[1:], hist_t_by_s_adj, bin_s_label[1:]))
        bin_hist_label.sort(key=cmp_to_key(lambda x, y: x[1] - y[1]))
        (bin_s, hist_t_by_s_adj, bin_s_label) = zip(*bin_hist_label)
        bin_s = list(bin_s)
        hist_t_by_s_adj = list(hist_t_by_s_adj)
        bin_s_label = list(bin_s_label)
        bin_s.insert(0, 0)
        bin_s_label.insert(0, '_')
        # re-arrange hist_s and hist_t_by_s
        hist_s_new = []
        hist_t_by_s_new = []
        for i in bin_s[1:]:
            hist_s_new.append(hist_s[i - 1])
            hist_t_by_s_new.append(hist_t_by_s[i - 1])
        hist_s = hist_s_new
        hist_t_by_s = hist_t_by_s_new
        # print('after sort:\n%s\n%s\n%s' % (bin_s, bin_s_label, hist_t_by_s_adj))
        # reset bin_s's ordering
        bin_s.sort()
    hist_s = list(hist_s)
    if len(hist_s) < len(bin_s):
        hist_s.insert(0, 0.0)
    hist_s_max = max(hist_s)
    plt.fill_between(bin_s, hist_s, step='mid', alpha=0.5, label=feature)
    if bin_s_label is not None:
        plt.xticks(bin_s, bin_s_label)
    plt.xticks(rotation=90)
    # just to show legend for ax2
    # plt.errorbar([], [], yerr = [], fmt = 'ok', lw = 3, ecolor = 'sienna', mfc = 'sienna', label = target)
    plt.legend(loc='upper right')
    hist_t_by_s = list(hist_t_by_s)
    if len(hist_t_by_s) < len(bin_s):
        hist_t_by_s.insert(0, [0.0])
    hist_t_by_s_min = [float(min2(n)) for n in hist_t_by_s]
    hist_t_by_s_max = [float(max2(n)) for n in hist_t_by_s]
    hist_t_by_s_avg = [float(avg2(n)) for n in hist_t_by_s]
    hist_t_by_s_std = [float(std2(n)) for n in hist_t_by_s]
    hist_t_by_s_err = [np.array(hist_t_by_s_avg) - np.array(hist_t_by_s_min),
                       np.array(hist_t_by_s_max) - np.array(hist_t_by_s_avg)]
    plt.xlabel(feature)
    plt.ylabel('Count')
    ax2 = plt.twinx()
    ax2.grid(False)
    ax2.errorbar(bin_s, hist_t_by_s_avg, yerr=hist_t_by_s_err, fmt='.k', lw=1, ecolor='sienna')
    ax2.errorbar(bin_s, hist_t_by_s_avg, yerr=hist_t_by_s_std, fmt='ok', lw=3, ecolor='sienna', mfc='sienna',
                 label=target)
    ax2.set_ylabel(target)
    plt.legend(loc='upper left')
    plt.tight_layout()


# check thess vareables
numerical_fields = [
    'longitude','latitude','bedrooms','price','washrooms','bedroomsPlus','lotDepth','lotFront',
    'kitchens','kitchensPlus','parkingSpaces','room1Length','room1Width','room2Length',
    'room3Length', 'room3Width', 'room4Length', 'room4Width', 'room5Length', 'room5Width',
    'room6Length','room6Width',   'room7Length',    'room7Width',  'room8Length',
    'room8Width',  'room9Length',   'room9Width',         'rooms',
    'taxes',  'garageSpaces',  'totalParkingSpaces'

]

categorical_fields = [

]


fields = numerical_fields + categorical_fields

# plt.figure(figsize = (20, 90))
# i = 1
# for name in fields:
#     plt.subplot(21, 4, i)
#     plot_hist_with_target3(plt, df, name, 'daysOnMarket', histogram_bins = 'rice')
#     i += 1
# plt.tight_layout()
