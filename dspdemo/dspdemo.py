#
#  Copyright (C) 2013  Clifford Wolf <clifford@clifford.at>
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#  
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

from __future__ import division
from __future__ import print_function

# set this to the IP address of your RIGOL DS2000 scope
# (for other scopes you might need to change the GetWafeform function below)
DS2000_VXI11_IP = '192.168.1.20'
VXI11_VERBOSE = False

# PyVXI11 can be downloaded from https://pypi.python.org/pypi/PyVXI11
# This has been tested with version 1.11.3.5 of PyVXI11.
import vxi11Device

from numpy.fft import rfft, irfft
from numpy import sinc

from os import remove
from os.path import exists
from time import sleep

import argparse
import matplotlib.pyplot as pyplot

###########################################################################

def GetWafeform():
    """
    Download a waveform from the RIGOL DS2000 scope
    """

    print('Downloading waveform data..')

    # connecting to the scope
    osc = vxi11Device.Vxi11Device(DS2000_VXI11_IP, 'inst0')

    # helper functions for optional verbose output
    def print_verbose(msg, end='\n'):
        if VXI11_VERBOSE:
            print(msg, end=end)
    def osc_write_verbose(msg):
        print_verbose(msg)
        osc.write(msg)

    # we expect CH1 to be active and CH2 to be inactive
    if int(osc.ask(':CHAN1:DISP?')) != 1:
        print('ERROR: Scope channel 1 is not active! Active channel 1 and deactive channel 2.');
        exit(1)
    if int(osc.ask(':CHAN2:DISP?')) != 0:
        print('ERROR: Scope channel 2 is active! Active channel 1 and deactive channel 2.');
        exit(1)

    # set memory depth
    osc_write_verbose(':ACQUIRE:MDEPTH 140000')

    # trigger and wait for single shot
    osc_write_verbose(':SINGLE')
    while True:
        status = osc.ask(':TRIGGER:STATUS?').rstrip()
        print_verbose(' -> status: {}'.format(status))
        if status == 'STOP':
            break
        sleep(0.05)

    # setup
    osc_write_verbose(':WAVEFORM:MODE RAW')
    osc_write_verbose(':WAVEFORM:SOURCE CHAN1')
    osc_write_verbose(':WAVEFORM:POINTS 140000')
    osc_write_verbose(':WAVEFORM:RESET')

    # read data from scope
    osc_write_verbose(':WAVEFORM:BEGIN')
    data = []
    while True:
        status = osc.ask(':WAVEFORM:STATUS?').rstrip()
        print_verbose('Downloading "{}" at {}:'.format(status, len(data)), end='')
        block = osc.ask(':WAVEFORM:DATA?')
        assert block[0] == '#'
        block = bytearray(block[int(block[1])+2:-2])
        for value in block:
            data.append(int(value))
        print_verbose(' {}'.format(len(block)))
        if status[0] == 'I':
            break
    osc_write_verbose(':WAVEFORM:END')
    print('Got {} bytes from device.'.format(len(data)))

    # go back to run state and return the recorded wave form
    osc_write_verbose(':RUN')
    return data

###########################################################################

class PatternGen:
    """
    A simple class for generating pattern for RS232-like data streams.
    """

    # internal array of all data bits
    data_bits = []

    def add_bits(self, values):
        """
        Add individual bits to the pattern
        """

        self.data_bits.extend(values)

    def add_byte(self, value):
        """
        Add one byte to the pattern
        (rs232 encoding: one start bit, 8 data bist, one stop bit)
        """

        # start bit
        self.data_bits.append(False)

        # data bits (LSB first)
        for i in range(8):
            self.data_bits.append((value & (1 << i)) != 0)

        # stop bit
        self.data_bits.append(True)

    def finalize(self, num_samples):
        """
        Generate and return sample array (num_samples samples in 0.0 .. 1.0)
        """

        samples = []
        for i in range(num_samples):
            if self.data_bits[len(self.data_bits)*i//num_samples]:
                samples.append(1.0)
            else:
                samples.append(0.0)
        return samples;

###########################################################################

def RdfWrite(filename, samples):
    """
    A simple function for writing RDF files for a RIGOL DG1000
    """

    # be verbose
    print("Generating {}...".format(filename))

    # do it
    f = open(filename, "wb")
    val_min = min(samples)
    val_max = max(samples)
    for val in samples:
        intval = int(16383 * (val - val_min) / (val_max - val_min))
        f.write(bytearray([ intval & 0xff, intval >> 8 ]))
    f.close();

###########################################################################

def AsciiRead(filename):
    """
    Read sample data from ASCII file and normalize to -1.0 .. +1.0 with DC=0
    """

    # read samples from file
    samples = []
    val_sum = 0
    f = open(filename, "rt")
    for line in f:
        val = float(line)
        val_sum = val_sum + val
        samples.append(val)
    f.close()

    # perform normalization
    val_avg = val_sum / len(samples)
    val_scale = 1.0 / max([ abs(max(samples) - val_avg), abs(min(samples) - val_avg) ])
    for i in range(len(samples)):
        samples[i] = (samples[i]-val_avg) * val_scale

    return samples

###########################################################################

def SincSamples(num_samples, num_period_samples):
    """
    Generate samples of a sinc waveform
    """

    samples = []
    for i in range(num_samples):
        samples.append(sinc((i-num_samples/2) / num_period_samples))
    return samples

###########################################################################

def main_patterngen():
    """
    Main function for -gen mode:
    Generate RDF files for the RIGOL DG1000
    """

    pg = PatternGen()

    # idle for 10 cycles
    pg.add_bits([True] * 10)

    # data
    pg.add_byte(ord("D"))
    pg.add_byte(ord("S"))
    pg.add_byte(ord("P"))

    # idle for 10 more cycles
    pg.add_bits([True] * 10)

    # generate pattern samples array
    pattern_samples = pg.finalize(4096)

    # write RDF raw pattern file
    RdfWrite("pattern_raw.rdf", pattern_samples)

    # generate sinc samples array and create RDF file
    sinc_samples = SincSamples(4096, 100)
    RdfWrite("pattern_sinc.rdf", sinc_samples)

    # plot sample data
    pyplot.subplot(211)
    pyplot.plot(pattern_samples)
    pyplot.subplot(212)
    pyplot.plot(sinc_samples)
    pyplot.show()

    # only execute 2nd half if we have a sinc_response.txt file
    if not exists("sinc_response.txt"):
        exit(0)

    # read sinc response
    sinc_response = AsciiRead("sinc_response.txt")

    # transform everything we need to the spectrum
    spect_pattern  = rfft(pattern_samples)
    spect_sinc_in  = rfft(sinc_samples)
    spect_sinc_out = rfft(sinc_response)

    # apply the inverse of spect_sinc_in -> spect_sinc_out to spect_pattern
    for i in range(len(spect_pattern)):
        factor = spect_sinc_in[i] / spect_sinc_out[i]
        spect_pattern[i] = factor * spect_pattern[i]

    # transform results back to time domain
    pattern_eq = irfft(spect_pattern)

    # plot sample data
    pyplot.subplot(211)
    pyplot.plot(pattern_samples)
    pyplot.subplot(212)
    pyplot.plot(pattern_eq)
    pyplot.show()

    # write equalized RDF pattern file
    RdfWrite("pattern_eq.rdf", pattern_eq)

###########################################################################

def main_getwaveform():
    """
    Main function for -get mode:
    Download waveform from scope and extract one period of the signal
    """

    data = GetWafeform()

    # convert waveform to spectrum, perform convolution (autocorrelation)
    # and lowpass-filtering, and convert back to the time domain
    spectr = rfft(data)
    for i in range(len(spectr)):
        if 0 < i and i < 100:
            spectr[i] = spectr[i] * spectr[i].conjugate()
        else:
            spectr[i] = 0
    conv = irfft(spectr)

    # find the period of the waveform
    period = 0
    for i in range(len(conv) // 2, 0, -1):
        if conv[i-1] < conv[i] and conv[i] > conv[i+1] and conv[period]*0.5 < conv[i]:
            period = i

    # extract one period from the center of the recorded waveform
    if period > 0:
        start = (len(data) - period) // 2
        wdata = data[start:start+period]
    else:
        wdata = data

    # interpolate/extrapolate to 4096 samples
    wdata = rfft(wdata)
    if len(wdata) < 2049:
        wdata.extend([0] * (2049-len(wdata)))
    if len(wdata) > 2049:
        wdata = wdata[0:2049];
    wdata = irfft(wdata)

    # normalize waveform
    min_sample = min(wdata)
    max_sample = max(wdata)
    for i in range(len(wdata)):
        wdata[i] = (wdata[i]-min_sample) / (max_sample-min_sample)

    # plot what we have got
    pyplot.subplot(311)
    pyplot.plot(data)
    pyplot.subplot(312)
    pyplot.plot(conv)
    pyplot.subplot(313)
    pyplot.plot(wdata)
    pyplot.show()

    # print waveform samples to file
    f = open("sinc_response.txt", "wb")
    for value in wdata:
        f.write("{}\n".format(value))
    f.close()

###########################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Little DSP demo using a RIGOL DG1000 and a RIGOL DS2000")
    parser.add_argument("--gen", action='store_true', help="Generate RDF files for the signal generator")
    parser.add_argument("--get", action='store_true', help="Download waveform and extract one period")
    args = parser.parse_args()

    if args.gen == args.get:
        parser.error("Either --gen or --get must be used.");

    if args.gen:
        main_patterngen()

    if args.get:
        main_getwaveform()

