# http://stackoverflow.com/questions/22883102/solving-system-of-equations
# glpsol --math q22883102.mod

var x1;
var x2;
var x3;
var x4;
var R;

s.t. rel1: 10*x1 + 10*x2 >= 23;
s.t. rel2: 5*x3 + 15*x4 >= 42;
s.t. rel3: x1 + x3 <= R;
s.t. rel4: x1 + x4 <= R;
s.t. rel5: x2 + x4 <= R;

minimize obj: R;

solve;

printf "Result: %f %f %f %f %f\n", x1, x2, x3, x4, R;

end;
