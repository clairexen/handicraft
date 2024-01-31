% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

function blatt03a(A,b,c)
    %% Plotting Config
    % Empfehlung fuer die Wahl des plotting algorithmus:
    %  - 'trace' fuer Kurven aus einer Linie (Ellipse, Parabel)
    %  - 'points' fuer Kurven aus zwei Linien (Hyperbel, Geradenpaar)
    %  - 'ezplot' fuer die Loesung mit Matlab black magic
    plotconfig = { 'trace', -15, +15, 0.05, 1};
    % plotconfig = { 'points', -15, +15, 0.1 };
    % plotconfig = { 'ezplot', -15, +15 };

    %% Initialize Plotting Backend
    clf; hold on;

    %% Original Quadrik (rot)
    % x'*A*x + b'*x + c = 0
    h1 = showquad(A, b, c, plotconfig, 'r');
    
    %% Rotation (gruen)
    % Sei y := V'*x und A = V'*D*V:
    % x'*V'*D*V*x + b'*V*V'*x + c = 0 =>
    % y'*D*y + (V'*b)'*y + c = 0
    [V,D] = eig(A);
    gamma = V'*b;
    h2 = showquad(D, gamma, c, plotconfig, 'g');

    %% Verschieben (blau)
    % Sei z := y + t:
    % (z-t)'*D*(z-t) + gamma'*(z-t) + c = 0 =>
    % z'*D*z + (gamma - 2*D*t)'*z + (t'*D*t - gamma'*t + c) = 0
    t = zeros(2,1);
    for i=1:2
        if abs(D(i,i)) > sqrt(eps)
            t(i) = gamma(i) / (2*D(i,i));
        end
    end
    mu = gamma - 2*D*t;
    beta = t'*D*t - gamma'*t + c;
    h3 = showquad(D, mu, beta, plotconfig, 'b');

    %% Anpassen des Plot Windows
    legend([h1, h2, h3], { 'Original', 'Gedreht', 'Verschoben'});
    axis([1 0; 0 1; 1 0; 0 1]*[plotconfig{2}; plotconfig{3}]);
end

function h = showquad(A, b, c, config, style)
    fprintf(1, '%.2f*x^2 + %.2f*y^2 + %.2f*xy + %.2f*x + %.2f*y + %.2f = 0\n', ...
            A(1,1), A(2,2), A(1,2)+A(2,1), b(1), b(2), c);
    if strcmp(config{1}, 'points')
        h = showquad_points(A, b, c, config{2}:config{4}:config{3}, style);
    elseif strcmp(config{1}, 'trace')
        h = showquad_trace(A, b, c, config{2}, config{4}, config{3}, style, config{5});
    elseif strcmp(config{1}, 'ezplot')
        h = ezplot(@(X,Y) ezplot_fun(X, Y, A, b, c), [config{2}, config{3}]);
        set(h, 'Color', style);
    end
end

function Z = ezplot_fun(X, Y, A, b, c)
    % X, Y und Z sind jeweils Spaltenvektoren
    x = [X, Y]';
    Z = (x' .* (A*x)')*[1;1] + x'*b + c;
end

function h = showquad_points(inA, inb, inc, range, style)
    % Symbolische Loesung (mit Maxima):
    %
    % solve(a*x*x+b*y*y+c*x*y+d*x+e*y+f = 0,x)
    % x1 = -(sqrt((c^2-4*a*b)*y^2+(2*c*d-4*a*e)*y-4*a*f+d^2)+c*y+d)/(2*a)
    % x2 = +(sqrt((c^2-4*a*b)*y^2+(2*c*d-4*a*e)*y-4*a*f+d^2)-c*y-d)/(2*a)
    %
    % solve(a*x*x+b*y*y+c*x*y+d*x+e*y+f = 0,y)
    % y1 = -(sqrt((c^2-4*a*b)*x^2+(2*c*e-4*b*d)*x-4*b*f+e^2)+c*x+e)/(2*b)
    % y2 = +(sqrt((c^2-4*a*b)*x^2+(2*c*e-4*b*d)*x-4*b*f+e^2)-c*x-e)/(2*b)

    %% Zerlegen der Eingangsdaten in einzelne Skalare
    a = inA(1,1);
    b = inA(2,2);
    c = inA(1,2) + inA(2,1);
    d = inb(1);
    e = inb(2);
    f = inc;
    
    %% Berechnen der Punkte der Kurve
    % um eine moeglichst gleichmaessige Abrasterung der Kurve
    % zu erzielen wird das Gleichungssystem sowohl nach der x- als
    % auch der y-Koordinate geloest.
    data = zeros(2,0);
    for i=range
        % Berechnen der x-Werte aus dem y-Wert
        % (ueberspringen bei horizontaler Gerade)
        if a ~= 0
            y = i;
            x1 = -(sqrt((c^2-4*a*b)*y^2+(2*c*d-4*a*e)*y-4*a*f+d^2)+c*y+d)/(2*a);
            x2 = +(sqrt((c^2-4*a*b)*y^2+(2*c*d-4*a*e)*y-4*a*f+d^2)-c*y-d)/(2*a);
            if imag(x1) == 0
                data = [ data [x1; y] ];
            end
            if imag(x2) == 0
                data = [ data [x2; y] ];
            end
        end
        % Berechnen der y-Werte aus dem x-Wert
        % (ueberspringen bei vertikaler Gerade)
        if b ~= 0
            x = i;
            y1 = -(sqrt((c^2-4*a*b)*x^2+(2*c*e-4*b*d)*x-4*b*f+e^2)+c*x+e)/(2*b);
            y2 = +(sqrt((c^2-4*a*b)*x^2+(2*c*e-4*b*d)*x-4*b*f+e^2)-c*x-e)/(2*b);
            if imag(y1) == 0
                data = [ data, [x; y1] ];
            end
            if imag(y2) == 0
                data = [ data, [x; y2] ];
            end
        end
    end
    
    %% Zeichnen der Kurve
    if length(data) > 0
        h = plot(data(1,:), data(2,:), ['.' style]);
    end
end

function h = showquad_trace(A, b, c, minpos, stepsize, maxpos, style, showgrad)
    %% Suchen eines geeigneten Startpunktes
    x = numsolve_quad(A, b, c, randn(2,1));
    loop_is_closed = 0;
    data = x;
    
    %% Numerische Integration normal zum Gradientenfeld (Richtung 1)
    while minpos-stepsize < x(1) && x(1) < maxpos+stepsize && ...
            minpos-stepsize < x(2) && x(2) < maxpos+stepsize && ~loop_is_closed
        % vstep = Vektor mit Laenge step entlang der Niveaulinie
        grad_f = (A+A') * x + b;
        vstep = [0 stepsize; -stepsize 0] * grad_f / norm(grad_f);
        % Ermitteln und korrigieren der Zielkoordinate
        x = numsolve_quad(A, b, c, x + vstep);
        data = [ data, x ];
        % Abschluss wenn wir zurueck am startpunkt sind
        if length(data) > 3 && norm(data(:,1) - x) < stepsize
            data = [ data, data(:,1) ];
            loop_is_closed = 1;
        end
    end
    
    %% Numerische Integration normal zum Gradientenfeld (Richtung 2)
    x = data(:,1);
    while minpos < x(1) && x(1) < maxpos && minpos < x(2) && x(2) < maxpos && ~loop_is_closed
        % Unterschiede zu Richtung 1:
        %  - umgedrehtes Vorzeichen bei der Anwendung von "vstep"
        %  - einfuegen der neuen Koordinaten am Anfang von "data"
        %  - die ueberpruefung auf geschlossene Schleife ist nicht notwendig
        grad_f = (A+A') * x + b;
        vstep = [0 stepsize; -stepsize 0] * grad_f / norm(grad_f);
        x = numsolve_quad(A, b, c, x - vstep);
        data = [ x, data ];
    end
    
    %% Plotten des Gradientenfeldes
    if showgrad
        center = (max(data') + min(data'))' * 0.5;
        radius = max(max(data') - min(data')) * 0.5;
        plot_gradient_field(A, b, radius, center);
    end
       
    %% Zeichnen der Kurve
    h = plot(data(1,:), data(2,:), style);
end

function plot_gradient_field(A, b, r, x)
    N = 11;
    Mx = zeros(N);
    My = zeros(N);
    Mu = zeros(N);
    Mv = zeros(N);
    for i=1:N
        for j=1:N
            N2 = (N-1)/2;
            p = x + [r*(i-N2-1) / N2; r*(j-N2-1) / N2];
            g = (A+A') * p + b;
            Mx(i,j) = p(1);
            My(i,j) = p(2);
            Mu(i,j) = g(1);
            Mv(i,j) = g(2);
        end
    end
    quiver(Mx, My, Mu, Mv, 'Color', [.8, .8, .8]);
end

function x = numsolve_quad(A, b, c, x)
    % x_neu = x - t
    % mit t = f(x) * (grad f(x)).^-1
    % und grad f(x) = (A+A') * x + b
    % .. iteration bis f(x) nahe 0
    itercount = 0;
    f = x'*A*x + b'*x + c;
    while abs(f) > sqrt(eps)
        grad_f = (A+A') * x + b;
        if norm(grad_f) > sqrt(eps)
            t = f * grad_f / norm(grad_f)^2;
            x_temp = x - t;
            f_temp = x_temp'*A*x_temp + b'*x_temp + c;
            if abs(f_temp) > abs(f)
                % Exponentielle Daempfung
                x_best = x;
                f_best = f;
                for i=-5:0.5:1
                    x_temp = x - t * exp(i);
                    f_temp = x_temp'*A*x_temp + b'*x_temp + c;
                    if abs(f_temp) < abs(f_best)
                        x_best = x_temp;
                        f_best = f_temp;
                    end
                end
                x_temp = x_best;
            end
            x = x_temp;
        else
            % Jacobi singularitaet!
            % .. etwas rauschen kann da helfen
            warning('Jacobi Singularitaet in numsolve_quad()!');
            x = x + randn(2,1)*0.01;
        end
        f = x'*A*x + b'*x + c;
        % Abbruchkriterium fuer reell nicht loesbare Gleichungen
        % (oder lokale Minima - afaics gibt es das bei Quadriken aber nicht)
        itercount = itercount + 1;
        if itercount > 100
            error('numsolve_quad kann diese Quadrik nicht loesen!');
        end
    end
end

%{
Testcase (copy&paste to command window):
blatt03a([7, 3*sqrt(3); 3*sqrt(3), 13], [-12*(sqrt(3)+4); -12*(4*sqrt(3)-1)], 164);
blatt03a(10*rand(2), 10*rand(2,1), 100*rand());
%}
