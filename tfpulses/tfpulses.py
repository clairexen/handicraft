#!/usr/bin/env python3

import numpy as np
from numba import jit
import tensorflow as tf

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys, time, json, os.path

hardwaves = []

@jit
def makewave(make_hard_prop = 0.0):
    n = 30
    samples = np.zeros(n)
    trigger = np.zeros(n)

    count_pulses = 0
    count_maxima = 0

    for pcnt in range(4):
        # pulse amplitude, sigma, and position
        a = np.random.uniform(3, 10)
        s = np.random.uniform(2, 3)
        p = np.random.uniform(3*s, n-3*s)

        collision = False
        for i in range(int(p)-1, int(p)+3):
            if trigger[i] > 0.1:
                collision = True
        if collision:
            continue

        trigger[int(p)+1] = int(32 * (p - int(p))) / 32.0
        trigger[int(p)] = 1.0 - trigger[int(p)+1]

        for i in range(n):
            samples[i] += a * np.exp(-((i-p)**2) / (s**2))

        count_pulses += 1

    for i in range(1, n-1):
        if (samples[i-1] < samples[i]) and (samples[i+1] < samples[i]):
            count_maxima += 1

    if count_maxima < count_pulses:
        if len(hardwaves) < 10000:
            hardwaves.append([samples, trigger])
        else:
            idx = np.random.randint(len(hardwaves))
            hardwaves[idx] = [samples, trigger]

    elif len(hardwaves) > 0:
        if np.random.uniform() <= make_hard_prop:
            idx = np.random.randint(len(hardwaves))
            samples, trigger = hardwaves[idx]

    samples = np.max([np.array(samples) + np.random.uniform(-0.1, 0.1, n), np.zeros(n)], 0)

    return samples, trigger


class machina:
    def __init__(self, nsamples, nlookback, nnwidth, keep_prop):
        self.nsamples = nsamples
        self.nlookback = nlookback
        self.nnwidth = nnwidth
        self.keep_prop = keep_prop

        self.param_w1 = []
        self.param_b1 = []

        for i in range(len(self.nnwidth)-1):
            dim1 = self.nnwidth[i-1] if i > 0 else self.nnwidth[-1] + self.nsamples + 2*self.nlookback
            dim2 = self.nnwidth[i]
            self.param_w1.append(tf.Variable(tf.truncated_normal([dim1, dim2], stddev=0.5), name=("w1_%d" % i)))
            self.param_b1.append(tf.Variable(tf.truncated_normal([dim2], stddev=0.5), name=("b1_%d" % i)))

        self.param_w2 = tf.Variable(tf.zeros([self.nnwidth[-2], 2]), name="w2")
        self.param_b2 = tf.Variable(tf.zeros([2]), name="b2")

        self.sigfront = None
        self.lookback = None
        self.triggers = []

    def mkslice(self, samples):
        if self.sigfront is None:
            self.zero_column = tf.slice(0 * samples, [0, 0], [-1, 1])
            self.ones_column = self.zero_column + 1
            self.sigfront = tf.concat(1, self.nnwidth[-1] * [self.zero_column])
        else:
            self.sigfront = tf.slice(self.sigfront, [0, 0], [-1, self.nnwidth[-1]])

        if self.lookback is None:
            init_lookback_row = tf.constant([[float(i % 2) for i in range(2*self.nlookback)],])
            self.lookback = tf.matmul(self.ones_column, init_lookback_row)

        # normalize samples
        max_samples = tf.reduce_max(tf.concat(1, [samples, self.ones_column]), 1, True)
        samples /= tf.tile(max_samples, [1, self.nsamples])

        self.sigfront = tf.concat(1, [samples, self.lookback, self.sigfront])

        # fully connected layers
        for i in range(len(self.nnwidth)-1):
            self.sigfront = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(self.sigfront, self.param_w1[i]) + self.param_b1[i]), self.keep_prop)

        output = tf.nn.softmax(tf.matmul(self.sigfront, self.param_w2) + self.param_b2)
        self.lookback = tf.concat(1, [tf.slice(self.lookback, [0, 2], [-1, -1]), output])

        self.triggers.append(tf.slice(output, [0, 0], [-1, 1]))

    def mktrig(self):
        return tf.concat(1, self.triggers)


class trainer:
    def __init__(self):
        self.window = [2, 4]
        self.nlookback = 3
        self.nnwidth = [100, 100, 100, 10]
        self.training_batchsize = 1000
        self.learning_rate = 0.001
        self.learning_keep = 0.5

    def setup(self):
        self.samples = tf.placeholder("float", shape=[None, 30])
        self.triggers = tf.placeholder("float", shape=[None, 30 - self.window[0] - self.window[1]])
        self.keep_prob = tf.placeholder("float")

        self.M = machina(self.window[0] + self.window[1] + 1, self.nlookback, self.nnwidth, self.keep_prob)

        for i in range(0, 30 - self.window[0] - self.window[1]):
            sample_slice = tf.slice(self.samples, [0, i], [-1, self.window[0] + self.window[1] + 1])
            self.M.mkslice(sample_slice)

        self.triggers_out = self.M.mktrig()

        self.cnt = 0
        self.training_time = 0
        self.cross_entropy_history = []
        self.cross_entropy_history_lp = []

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()

        self.cross_entropy = -(tf.reduce_sum(self.triggers * tf.log(self.triggers_out)) +
                tf.reduce_sum((1.0-self.triggers) * tf.log(1.0-self.triggers_out)))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.sess.run(tf.initialize_all_variables())

    def save(self, filename_prefix):
        with open(filename_prefix + ".json", "w") as f:
            print(json.dumps({
                "cnt": self.cnt,
                "training_time": self.training_time,
                "cross_entropy_history": self.cross_entropy_history,
                "cross_entropy_history_lp": self.cross_entropy_history_lp,
            }), file=f)

        self.saver.save(self.sess, filename_prefix + ".ckpt")

    def load(self, filename_prefix):
        if not os.path.isfile(filename_prefix + ".json"): return False
        if not os.path.isfile(filename_prefix + ".ckpt"): return False

        with open(filename_prefix + ".json", "r") as f:
            data = json.loads(f.read())
            self.cnt = data["cnt"]
            self.training_time = data["training_time"]
            self.cross_entropy_history = data["cross_entropy_history"]
            self.cross_entropy_history_lp = data["cross_entropy_history_lp"]

        self.saver.restore(self.sess, filename_prefix + ".ckpt")
        return True

    def load_or_train(self, filename_prefix, train_iterations):
        if not self.load(filename_prefix):
            self.run_training(train_iterations)
            self.save(filename_prefix)
        else:
            self.plot_history()

    def plot_cross_entropy_history(self):
        title = "Final (low-pass filtered) cross entropy after %d training steps (%d:%02d:%02d): %d" % \
              (len(self.cross_entropy_history), int(self.training_time) // 3600, (int(self.training_time) // 60) % 60,
              int(self.training_time) % 60, self.cross_entropy_history_lp[-1])

        if len(self.cross_entropy_history) < 250:
            plt.plot(self.cross_entropy_history, "k.")
            plt.plot(self.cross_entropy_history_lp, "r")
            plt.title(title)
            plt.grid()

        else:
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
            # plt.suptitle(title)

            plt.subplot(gs[0])
            if len(self.cross_entropy_history) < 500:
                plt.plot(self.cross_entropy_history, "k.")
                plt.plot(self.cross_entropy_history_lp, "r")
            else:
                x = np.arange(50, len(self.cross_entropy_history))
                plt.plot(x, self.cross_entropy_history[50:], "k.")
                plt.plot(x, self.cross_entropy_history_lp[50:], "r")
                plt.xlim(0, len(self.cross_entropy_history))
            plt.title("Avg. rate: %d iter/hour" % (len(self.cross_entropy_history) * 3600 / self.training_time))
            plt.grid()

            plt.subplot(gs[1])
            k = 100 * (len(self.cross_entropy_history) // 200)
            x = np.arange(k, len(self.cross_entropy_history))
            plt.plot(x, self.cross_entropy_history[k:], "k.")
            plt.plot(x, self.cross_entropy_history_lp[k:], "r")
            plt.xlim(k, len(self.cross_entropy_history))
            plt.title(title)
            plt.grid()

    def get_events(self, offset, seq):
        positions = []
        for i in range(1, len(seq)-1):
            if seq[i-1] < seq[i] >= seq[i+1] and seq[i] > 0.1:
                wl, wc, wr = seq[i-1:i+2]
                positions.append(offset + (wl*(i-1) + wc*i + wr*(i+1)) / (wl + wc + wr))
        return positions

    def plot_events(self, positions, style):
        if len(positions) > 0:
            _,_,baseline = plt.stem(positions, len(positions)*[-1 if style == "bd" else -1.5], "k", style);
            plt.setp(baseline, 'linewidth', 0)

    def plot_waveform(self, s, t, tout):
        plt.plot(s, "k")
        plt.plot(5*np.array(t), "b.");
        plt.plot(range(self.window[0], 30 - self.window[1]), 5*tout, "g")
        self.plot_events(self.get_events(0, t), "bd")
        self.plot_events(self.get_events(self.window[0], tout), "gd")

    def plot_single_example_waveform(self):
        s, t = makewave(1.0)
        tout = self.sess.run(self.triggers_out, feed_dict={self.samples: [s], self.triggers: [t[self.window[0]:30 - self.window[1]]], self.keep_prob: 1.0})
        self.plot_waveform(s, t, tout[0])
        plt.title('Results for a random "hard" waveform')

    def run_training(self, train_iterations):
        start_time = time.time()

        for cnt in range(train_iterations):
            if cnt % 25 == 0:
                clear_output()
                stop_time = time.time()
                self.training_time += stop_time - start_time
                start_time = stop_time
                if len(self.cross_entropy_history) > 10:
                    plt.figure(figsize=(14.7, 4))
                    self.plot_cross_entropy_history()
                    plt.show()
                    plt.figure(figsize=(15, 3))
                    self.plot_single_example_waveform()
                    plt.show()
                if cnt > 0:
                    print("[cont.]  avg. training rate: %d iterations per hour" % \
                          (len(self.cross_entropy_history) * 3600 / self.training_time))

            print("%5d %3d: Making.." % (self.cnt, cnt), end="")
            sys.stdout.flush()

            samples_data = []
            triggers_data = []
            for i in range(self.training_batchsize):
                s, t = makewave(0.9)
                samples_data.append(s)
                triggers_data.append(t[self.window[0]:30 - self.window[1]])

            this_cross_entropy = self.sess.run(self.cross_entropy, feed_dict={self.samples: samples_data, self.triggers: triggers_data, self.keep_prob: 1.0})

            if len(self.cross_entropy_history) == 0:
                last_cross_entropy = this_cross_entropy
            else:
                last_cross_entropy = self.cross_entropy_history[-1]

            self.cross_entropy_history.append(float(this_cross_entropy))

            if len(self.cross_entropy_history_lp) < 10:
                self.cross_entropy_history_lp.append(float(this_cross_entropy))
            else:
                q = max(0.05, 10 / len(self.cross_entropy_history_lp))
                self.cross_entropy_history_lp.append(float(q * this_cross_entropy + (1-q) * self.cross_entropy_history_lp[-1]))

            print(" %.3f [%+.2e]" % (this_cross_entropy, this_cross_entropy - last_cross_entropy), end="")

            print(" Training..", end="")
            sys.stdout.flush()
            self.sess.run(self.train_step, feed_dict={self.samples: samples_data, self.triggers: triggers_data, self.keep_prob: self.learning_keep})

            print("")
            self.cnt += 1

        clear_output()
        self.plot_history()

    def plot_history(self):
        plt.figure(figsize=(14.7, 4))
        self.plot_cross_entropy_history()
        plt.show()

    def run_testing(self):
        print("Running 100 random hard waveforms..")
        samples_data = []
        triggers_data = []

        for i in range(100):
            s, t = makewave(1.0)
            samples_data.append(s)
            triggers_data.append(t)

        tout = self.sess.run(self.triggers_out, feed_dict={self.samples: samples_data, \
                self.triggers: [col[self.window[0]:30 - self.window[1]] for col in triggers_data], self.keep_prob: 1.0})

        quality_10 = 0
        quality_50 = 0
        quality_90 = 0
        quality_xx = 0
        quality_list = []

        for i in range(100):
            ref_events = self.get_events(0, triggers_data[i])
            out_events = self.get_events(self.window[0], tout[i])
            if len(ref_events) != len(out_events):
                worst_distance = abs(len(ref_events) - len(out_events))
            else:
                quality_xx += 1
                worst_distance = 0.0
                for k in range(len(ref_events)):
                    worst_distance = max(worst_distance, abs(ref_events[k] - out_events[k]))
            if worst_distance < 0.1: quality_10 += 1
            if worst_distance < 0.5: quality_50 += 1
            if worst_distance < 0.9: quality_90 += 1
            quality_list.append((worst_distance, i))

        print("Percentage of correct solutions:    %3d%%" % quality_xx)
        print("Percentage of solutions within 0.9: %3d%%" % quality_90)
        print("Percentage of solutions within 0.5: %3d%%" % quality_50)
        print("Percentage of solutions within 0.1: %3d%%" % quality_10)

        quality_list.sort()

        for i, label in [(99, "Worst"), (50, "Median"), (0, "Best")]:
            plt.figure(figsize=(15,3))
            plt.title("%s of 100 hard waveforms (max distance %.2f):" % (label, quality_list[i][0]))
            self.plot_waveform(samples_data[quality_list[i][1]], triggers_data[quality_list[i][1]], tout[quality_list[i][1]])

