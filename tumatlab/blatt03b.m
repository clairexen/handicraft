% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

function blatt03b(x0, y0)
    clf; hold on

    [X, Y] = meshgrid(-1:0.1:+1, -1:0.1:+1);
    Z = (X.*Y) ./ (X.^2 + Y.^2);
    mesh(X, Y, Z);

    % use trange = 90:90:180 for tx and ty only
    trange = 15:15:180;
    
    [z0, dx, dy] = tangent(x0, y0);
    for alpha=trange
        x = sin(alpha*pi/180);
        y = cos(alpha*pi/180);
        plot3([x0-x, x0+x], [y0-y, y0+y], ...
            [z0-x*dx-y*dy, z0+x*dx+y*dy], 'k');
    end
end

function [z, dx, dy] = tangent(x, y)
    z = (x*y) / (x^2 + y^2);
    dx = ((x^2 + y^2)*y - 2*x^2*(x*y)) / (x^2 + y^2)^2;
    dy = ((x^2 + y^2)*x - 2*y^2*(x*y)) / (x^2 + y^2)^2;
end
