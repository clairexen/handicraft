# http://stackoverflow.com/questions/22487133/how-to-find-the-optimal-sum/22488944#22488944
# glpsol --math optsum.mod

set A, dimen 3;
set B, dimen 2;

set I := setof{(i,j,v) in A} i;
set J := setof{(i,j,v) in A} j;

var x{I, J}, binary;
var s{J}, binary;

s.t. lines {i in I}:
  sum{j in J} x[i, j] = 1;

s.t. rows {j in J, i in I}:
  x[i, j] <= s[j];

minimize obj:
  (sum{(i,j,v) in A} x[i, j]*v) + (sum{(j,v) in B} s[j]*v);

solve;

printf "Result:\n";
printf {(i,j,v) in A : x[i, j] == 1} " %3d %3d %5d\n", i, j, v;
printf {(j,v) in B : s[j] == 1} " --- %3d %5d\n", j, v;
printf " --- --- %5d\n", (sum{(i,j,v) in A} x[i, j]*v) + (sum{(j,v) in B} s[j]*v);
printf "\n";

data;

set A :=
# Line 1
  1 1 2
  1 2 1
  1 3 1
# Line 2
  2 1 1
  2 2 4
  2 3 1
# Line 3
  3 1 3
  3 2 1
  3 3 3;

set B :=
# Line 4
  1 6
  2 5
  3 4;

end;
