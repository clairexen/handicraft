% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

% x' = -10*x + (101/10)*sin(t)
odefun = @(t, x) -10*x + (101/10)*sin(t);
t0 = 0;
tf = 10;
x0 = 1;

% solve using ode45 and ode15s
[ t_ode45,  x_ode45  ] = ode45(odefun, [t0 tf], x0);
[ t_ode15s, x_ode15s ] = ode15s(odefun, [t0 tf], x0);

% exact solution:
% x = sin(t) - 0.1*cos(t) + k*exp(-10*t)
k = (x0 - sin(t0) + 0.1*cos(t0))/exp(-10*t0);
x = @(t) sin(t) - 0.1*cos(t) + k*exp(-10*t);
t_exact = t0:(tf-t0)/1000:tf;
x_exact = x(t_exact);

% plot solutions
subplot(3,1,1);
plot(t_ode45, x_ode45(:,1), ...
     t_ode15s, x_ode15s(:,1), ...
     t_exact, x_exact);
legend('ode45', 'ode15s', 'exact');

% plot errors
subplot(3,1,2);
e_ode45 = abs(x_ode45(:,1) - x(t_ode45));
e_ode15s = abs(x_ode15s(:,1) - x(t_ode15s));
semilogy(t_ode45, e_ode45, ...
         t_ode15s, e_ode15s);
legend('error ode45', 'error ode15s');

% plot step size
subplot(3,1,3);
tbar_ode45 = t_ode45(1:size(t_ode45)-1) + diff(t_ode45);
tbar_ode15s = t_ode15s(1:size(t_ode15s)-1) + diff(t_ode15s);
semilogy(tbar_ode45, diff(t_ode45), ...
         tbar_ode15s, diff(t_ode15s));
legend('step ode45', 'step ode15s');
