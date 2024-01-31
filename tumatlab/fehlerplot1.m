% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

function fehlerplot1;
	semilogy([1:15], arrayfun(@blatt1_err, [1:15]));
end

function e = blatt1_err(n)
	[ A x d ] = blatt1(n);
	e = sum(abs(x - ones(n,1))) / n;
end

