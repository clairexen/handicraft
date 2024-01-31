% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

function x0 = blatt04a(x0, n)
    % f(x,y) = x^3 + x^2*y^2 - 14*x - 5*y
    fx = @(x, y) 3*x^2 + 2*x*y^2 - 14;
    fy = @(x, y) 2*x^2*y - 5;
    fxx = @(x, y) 6*x + 2*y^2;
    fxy = @(x, y) 4*x*y;
    fyx = @(x, y) 4*x*y;
    fyy = @(x, y) 2*x^2;
    
    % gesucht: (x,y) sodass (fx, fy) = (0, 0)
    for i=1:n
        x = x0(1);
        y = x0(2);
        
        r = [ fx(x, y); fy(x, y) ];
        A = [ fxx(x, y), fxy(x, y); ...
              fyx(x, y), fyy(x, y); ];
        h = A \ r;
        
        x0 = x0 - h;
    end
end

%{
Testcase (copy&paste to command window):
blatt04a([0.8; 2.1], 5);
%}
