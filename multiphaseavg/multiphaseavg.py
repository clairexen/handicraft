# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# -------------------------------------------------------
# Written by Clifford Wolf <clifford@clifford.at> in 2016
# -------------------------------------------------------

import numpy as np

def radix2butterfly(dft_even, dft_odd):
    """Combine two N/2-DFTs into one N-DFT using a Radix-2 Cooley–Tukey Butterfly.

    Combine the FFT of even (0, 2, 4, ...) and odd (1, 3, 5, ...) samples of a signal,
    as returned by np.fft.fft(), into the FFT for the entire signal.
    """

    N = len(dft_even)
    assert N == len(dft_odd)
    dft = np.zeros(2 * N, np.complex)
    omegas = np.exp(-2j * np.pi * np.arange(N) / (2*N))
    dft[0:N] = dft_even + dft_odd*omegas
    dft[N:2*N] = dft_even - dft_odd*omegas
    return dft

def radix2fft(samples):
    """Perform a single 1D Radix-2 Cooley–Tukey FFT

    This performs a single 1D FFT on the given set of samples. The returned array
    is equivalent to the array np.fft.fft() would return.
    """
    if len(samples) == 1:
        return samples
    dft_even = radix2fft(samples[0::2])
    dft_odd = radix2fft(samples[1::2])
    return radix2butterfly(dft_even, dft_odd)

def multiphaseavg(samples, phases, oversampling):
    """Average multiple undersampled waveforms at arbitrary phases into one oversampled waveform.

    This function combines undersampled, noisy waveforms taken at arbitrary phases, into one
    oversampled waveform.

    Parameters
    ----------
    samples: 2d array
        1st dimension: number of waveforms
        2nd dimension: number of samples per waveform
    phases: 1d array
        phase for each waveform. 1 = offset by one sample
    oversampling: int, power of two
        oversampling factor
    """

    assert bin(oversampling).count("1") == 1
    assert bin(samples.shape[1]).count("1") == 1
    assert samples.shape[0] == len(phases)

    # many forward DFTs
    samples_dft = np.zeros(samples.shape, np.complex)
    for i in range(samples.shape[0]):
        samples_dft[i, :] = radix2fft(samples[i, :])

    freq_dft = np.hstack((np.arange(0, samples.shape[1]//2), np.arange(-samples.shape[1]//2, 0))) / samples.shape[1]

    # correct phases and assign buckets
    buckets = [list() for i in range(oversampling)]
    for i in range(samples_dft.shape[0]):
        b = int(round(phases[i] * oversampling)) % oversampling
        p = phases[i] - b / oversampling
        samples_dft[i, :] *= np.exp(-2j * np.pi * p * freq_dft)
        buckets[b].append(i)

    # create per-bucket avg. FFTs
    avg_dft = np.zeros((oversampling, samples_dft.shape[1]), np.complex)
    for b in range(oversampling):
        avg_dft[b, :] = np.mean(samples_dft[buckets[b], :], 0)

    # combine individual FFTs using butterflies
    while avg_dft.shape[0] > 1:
        new_avg_dft = np.zeros((avg_dft.shape[0] // 2, avg_dft.shape[1] * 2), np.complex)
        for i in range(new_avg_dft.shape[0]):
            new_avg_dft[i, :] = radix2butterfly(avg_dft[i, :], avg_dft[i + new_avg_dft.shape[0], :])
        avg_dft = new_avg_dft

    # obtain oversampled time-domain representation using an inverse FFT
    outsamples = np.real(radix2fft(np.conj(avg_dft[0, :])))
    outsamples /= len(outsamples)
    return outsamples

def testbench(N, plt=None, datfile=None):
    """A simple test bench for multiphaseavg()"""

    def resample(wave, phase):
        W = np.fft.fft(wave)
        W = W * np.exp(2j * np.pi * phase * np.arange(len(wave)) / (len(wave)))
        W = np.real(np.fft.ifft(W))
        return W

    waveform = np.random.uniform(size=64)
    samples = np.zeros([N, 16])
    phases = np.zeros([N])

    for i in range(N):
        phase_fine = np.random.uniform()
        phase_coarse = np.random.randint(4)
        phases[i] = (phase_coarse + phase_fine) / 4
        samples[i, :] = resample(waveform, phase_fine)[phase_coarse::4]

    avgwaveform = multiphaseavg(samples, phases, 32)

    if plt is not None:
        plt.plot(waveform, 'ok')
        for i in range(N):
            plt.plot(4 * (np.arange(16) + phases[i]), samples[i,:], ',r')
        plt.plot(np.arange(512) / 8, avgwaveform, 'g')

    if datfile is not None:
        for i in range(N):
            datfile.write("I %f %s\n" % (phases[i], " ".join(["%f" % v for v in samples[i, :]])))
        datfile.write("O %s\n" % " ".join(["%f" % v for v in avgwaveform]))
