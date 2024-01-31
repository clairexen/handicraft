% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

%% Orthogonalisieren
B = blatt02(A);

%% Ermitteln von [v]_B
vb = B \ v;

%% Testcase: copy&paste to command window
%{
A = rand(3)
v = rand(3,1)
onb
[ B vb ]
[ v B*vb ]
%}
