# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# -------------------------------------------------------
# Written by Clifford Wolf <clifford@clifford.at> in 2015
# -------------------------------------------------------

import numpy as np

def butterfly_combine_dft(dft_even, dft_odd):
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

def fast_multiphase_avg(samples, phases, oversampling, bwlimit):
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
    bwlimit: real
        bandwith limit of the signal, given as factor to
        the nyquist frequency of the undersampled waveforms
    """
    assert bin(oversampling).count("1") == 1
    assert samples.shape[0] == len(phases)
    
    # many forward DFTs
    samples_dft = np.fft.fft(samples)
    freq_dft = np.fft.fftfreq(samples.shape[1])
    
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
            new_avg_dft[i, :] = butterfly_combine_dft(avg_dft[i, :], avg_dft[i + new_avg_dft.shape[0], :])
        avg_dft = new_avg_dft
    
    # low-pass filter results
    avg_dft[0, :] *= np.abs(np.fft.fftfreq(avg_dft.shape[1]))*2*oversampling < bwlimit

    # obtain oversampled time-domain representation using an inverse FFT
    results_x = np.arange(avg_dft.shape[1]) / oversampling
    results_y = np.real(np.fft.ifft(avg_dft[0, :]))
    
    return results_x, results_y
