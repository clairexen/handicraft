%
% Generate DAC samples to follow a given waveform after a first-order RC filter
%
% --------------------------------------------------------
%
% Copyright (C) 2013  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

global R C F B
R = 220;  % Resistance (Ohm)
C = 1e-8; % Capacitance (F)
F = 1e+6; % Sample-Rate (Hz)
B = 8;    % DAC Precicion (Bits)

function y = filter_response_single(t)
	global R C F
	tau = C*R*F;
	if t < 0
		y = 0;
	elseif t < 1
		y = 1 - exp(-t/tau);
	else
		y = exp((1-t)/tau) - exp(-t/tau);
	end
end

function Y = filter_response(T)
	Y = arrayfun(@filter_response_single, T);
end

function y = sinc_single(t)
	if abs(t) > 1e-3
		y = sin(t*pi/10)/(t*pi/10);
	else
		y = 1;
	end
end

function Y = sinc(T)
	Y = arrayfun(@sinc_single, T);
end

T = [-100:0.1:100];
Y = sinc(T);

A = zeros(length(T), length(-100:99));
for i = 1:length(-100:99)
	t = i-101;
	A(:,i) = filter_response(T-t);
end
X = (A'*A) \ (A'*Y');
X = round(X * 2^(B-1)) / 2^(B-1);

plot(T, Y, ';reference waveform;', T, A*X, ';synthesized waveform;', reshape([1 1]'*[-99:99], 1, 398), reshape([ X(1:199)'; X(2:200)' ], 1, 398), ';DAC output;');

