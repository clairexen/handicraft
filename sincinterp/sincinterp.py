
# Usage:
# ipython --pylab
# execfile('sincinterp.py')

from scipy import *

def sincinterp_time_domain(t, samples):
    y = 0
    my_inf = 10
    for i in range(int(t)-my_inf, int(t)+my_inf):
        idx = max(0, min(len(samples)-1, i))
        y += samples[idx] * sinc(t-i)
    return y

def sincinterp_freq_domain(divider, offset, samples):
    yy = zeros(len(samples)*divider)
    for i in range(len(samples)):
        yy[i*divider + offset] = samples[i]
    yy = fft(yy)
    for i in range(divider/4+1, len(yy)-divider/4):
        yy[i] = 0;
    yy[divider/4] *= 0.5;
    yy[len(yy)-divider/4] *= 0.5;
    yy = real(ifft(yy))
    maxidx = samples.index(max(samples))
    return multiply(yy,  samples[maxidx] / yy[maxidx*divider+offset])

XX = range(128)
YY = exp(-power(divide(subtract(XX, max(XX)/2.), 16.0), 2.0))

def pulse10():
    samples = [
         0.00000,      0.00020,      0.00055,      0.00318,      0.02458,
         0.04002,      0.14753,      0.67434,      1.26619,      2.11464,
         3.12810,      4.75223,      6.97377,      9.86608,     13.76470,
        18.59100,     24.50840,     31.56800,     40.04780,     49.94170,
        61.40080,     74.05150,     88.47600,    104.35400,    121.38200,
       139.99000,    159.60000,    180.42700,    202.22200,    224.56100,
       247.56600,    270.72600,    294.01000,    317.03300,    339.61200,
       361.70100,    382.96500,    402.67700,    421.25600,    438.40000,
       453.78100,    467.49400,    479.46000,    489.53100,    497.36500,
       502.38900,    505.91100,    507.28600,    506.48900,    503.72600,
       498.89500,    492.30700,    484.14800,    474.34400,    462.75400,
       450.35400,    436.34300,    421.74800,    406.65500,    390.94700,
       374.30100,    356.83100,    339.61100,    322.42300,    305.30200,
       288.21000,    271.58300,    254.90100,    238.95400,    223.49100,
       208.30900,    193.97600,    180.14000,    166.93800,    154.50100,
       142.63300,    131.46800,    120.93000,    111.13700,    101.99300,
        93.37630,     85.35020,     78.20640,     71.36580,     65.18420,
        59.49890,     54.27170,     49.55360,     45.16160,     41.14150,
        37.65300,     34.33050,     31.25190,     28.41130,     26.11440,
        23.89890,     21.91830,     19.96480,     18.38210,     16.85760,
        15.51410,     14.24750,     13.18580,     12.25280,     11.17870,
        10.44880,      9.52696,      8.84962,      8.32787,      7.73260,
         7.16982,      6.78424,      6.21274,      5.82974,      5.50081,
         5.32134,      5.05028,      4.82693,      4.56845,      4.46100,
         4.29423,      4.21737,      4.01212,      3.89536,      3.94619,
         3.96168,      3.99930,      4.01738
    ]
    return samples

def pulse50():
    samples = [
       1.1100e-01, 1.1923e+00, 3.1045e+00, 7.0015e+00, 1.3701e+01, 2.4549e+01,
       4.0711e+01, 6.3853e+01, 9.5120e+01, 1.3528e+02, 1.8510e+02, 2.4541e+02,
       3.1694e+02, 3.9874e+02, 4.9103e+02, 5.9327e+02, 7.0498e+02, 8.2491e+02,
       9.5224e+02, 1.0853e+03, 1.2228e+03, 1.3640e+03, 1.5075e+03, 1.6521e+03,
       1.7966e+03, 1.9393e+03, 2.0793e+03, 2.2162e+03, 2.3496e+03, 2.4777e+03,
       2.6004e+03, 2.7171e+03, 2.8274e+03, 2.9313e+03, 3.0284e+03, 3.1188e+03,
       3.2019e+03, 3.2783e+03, 3.3477e+03, 3.4102e+03, 3.4659e+03, 3.5147e+03,
       3.5567e+03, 3.5918e+03, 3.6204e+03, 3.6418e+03, 3.6565e+03, 3.6641e+03,
       3.6651e+03, 3.6595e+03, 3.6474e+03, 3.6289e+03, 3.6044e+03, 3.5742e+03,
       3.5389e+03, 3.4983e+03, 3.4532e+03, 3.4043e+03, 3.3520e+03, 3.2963e+03,
       3.2380e+03, 3.1776e+03, 3.1153e+03, 3.0516e+03, 2.9875e+03, 2.9228e+03,
       2.8578e+03, 2.7930e+03, 2.7288e+03, 2.6653e+03, 2.6030e+03, 2.5422e+03,
       2.4828e+03, 2.4252e+03, 2.3696e+03, 2.3163e+03, 2.2654e+03, 2.2168e+03,
       2.1706e+03, 2.1271e+03, 2.0857e+03, 2.0470e+03, 2.0106e+03, 1.9765e+03,
       1.9446e+03, 1.9146e+03, 1.8866e+03, 1.8602e+03, 1.8356e+03, 1.8125e+03,
       1.7909e+03, 1.7705e+03, 1.7514e+03, 1.7335e+03, 1.7167e+03, 1.7013e+03,
       1.6867e+03, 1.6733e+03, 1.6608e+03, 1.6494e+03, 1.6388e+03, 1.6292e+03,
       1.6203e+03, 1.6121e+03, 1.6045e+03, 1.5975e+03, 1.5910e+03, 1.5849e+03,
       1.5792e+03, 1.5738e+03, 1.5686e+03, 1.5637e+03, 1.5589e+03, 1.5544e+03,
       1.5501e+03, 1.5459e+03, 1.5420e+03, 1.5382e+03, 1.5347e+03, 1.5313e+03,
       1.5282e+03, 1.5253e+03, 1.5226e+03, 1.5200e+03, 1.5175e+03, 1.5152e+03,
       1.5129e+03, 1.5107e+03
    ]
    return samples

def add_noise(level):
    for i in range(len(YY)):
            YY[i] += rand()*level - level/2.0

def plot_time_domain_interp():
    figure()
    plot(XX, YY, '-')
    for phi in range(16):
        Y = []
        for i in range(len(XX) / 16):
            Y.append(YY[i*16 + phi])

        YI = []
        for t in XX:
            YI.append(sincinterp_time_domain((t-phi) / 16.0, Y))

        plot(XX, YI, ':')
    title('Time-domain sin(x)/x interpolation')

def plot_freq_domain_interp():
    figure()
    plot(XX, YY, '-')
    for phi in range(16):
        Y = []
        for i in range(len(XX) / 16):
            Y.append(YY[i*16 + phi])

        YI = sincinterp_freq_domain(16, phi, Y)
        plot(XX, YI, ':')
    title('Freq-domain sin(x)/x interpolation')

def plot_spectrum():
    figure()
    for phi in range(16):
        yy = zeros(len(YY))
        for i in range(phi, len(YY), 16):
            yy[i] = YY[i]
        plot(abs(fft(yy)))
    title('Spectrum after sampling')

def plot_best_phase():
    figure()
    plot(XX, YY, '-b')

    phases=[]
    for phi in range(16):
        yy = zeros(len(YY))
        for i in range(phi, len(YY), 16):
            yy[i] = YY[i]
        phases.append(min(abs(fft(yy))))
    phi = phases.index(min(phases))

    X = []
    Y = []
    for i in range(len(XX) / 16):
        X.append(XX[i*16 + phi])
        Y.append(YY[i*16 + phi])
    plot(X, Y, 'og')

    YI = sincinterp_freq_domain(16, phi, Y)
    plot(XX, YI, '-r')

    YI = []
    for t in XX:
        YI.append(sincinterp_time_domain((t-phi) / 16.0, Y))
    plot(XX, YI, ':r')

    yy = zeros(len(YY))
    for i in range(phi, len(YY), 16):
        yy[i] = YY[i]
    yy = fft(yy)
    plot(multiply(abs(yy), 0.5 * max(YY) / max(abs(yy))), '-y')

    for i in range(5, len(yy)-4):
        yy[i] = 0;
    yy[4] *= 0.5;
    yy[len(yy)-4] *= 0.5;
    plot(multiply(abs(yy), 0.5 * max(YY) / max(abs(yy))), '-m')

    title('Interpolation and spectrum at best phase (phi={})'.format(phi))

YY = pulse50()
# YY = pulse10()
# add_noise(3)

plot_time_domain_interp()
plot_freq_domain_interp()
plot_spectrum()
plot_best_phase()

