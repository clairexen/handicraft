% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

function [A, x, d] = blatt1(n)
	% Berechnen von A: Variante 1: Der ganz direkte Weg
	if 0
		A = hilb(n);
	end

	% Berechnen von A: Variante 2: Von hand
	if 1
		A_colcount = [1:n]' * ones(1,n);
		A = ones(n) ./ (A_colcount + A_colcount' - ones(n));
	end

	% Loesung Ax = b
	b = sum(A, 2);
	x = A \ b;

	% Berechnen der Determinante: Variante 1: Der ganz direkte Weg
	if 0
		d = det(A);
	end

	% Berechnen der Determinante: Variante 2: Durch Rekursion (Laplace)
	%
	% Dieser Weg entspricht der Berechnung der Determinante mit
	% der "grossen Loesungformel".
	if 0
		% laplace nach erster zeile. numerische stabilitaet ist
		% nicht so super und es ist sehr langsam. Ich nehme an,
		% dass das der eigentlich gefragte Loesungsweg ist.
		d = blatt1_mydet(A);
	end

	% Berechnen der Determinante: Variante 3: Durch LU-Decomposition
	%
	% Hinweis: Die Hilbert-Matrix ist positiv definit. D.h. die Determinante
	% muss immer positiv sein. Daher waere es auch moeglich das Vorzeichen
	% der Permutation zu ignorieren und statt dessen einfach den Betrag des
	% Produkts der Elemente entlang der Hauptdiagonalen von U zurueckzuliefern.
	if 1
		[L, U, P] = lu(A);
		% Vorzeichen der permutation mit laplace ermitteln
		% (dieser schritt ist numerisch stabil und geht wegen der spezial-
		% behandlung von duenn besetzten matrizen in blatt1_mydet schnell)
		d = blatt1_mydet(P);
		% produkt der diagonalen elemente von U
		d = d * prod((U .* eye(n)) * ones(n,1));
	end

	% Berechnen der Determinante: Variante 4: Mit Wikipedia
	%
	% Auf http://de.wikipedia.org/wiki/Hilbert-Matrix steht die Formel
	% fuer die Determinante des Inversen der Hilbert-Matrix.
	if 0
		d = 1 / prod((2*[1:n-1]+ones(1,n-1)) .* ...
			arrayfun(@(i) nchoosek(2*i, i), [1:n-1]).^(2*ones(1,n-1)));
	end

	% Berechnen der Determinante: Variante 5: Mit Eigenwerten
	%
	% In der Angabe stand nur man darf kein det() verwenden..
	if 0
		d = prod(eig(A));
	end
end

% Funktion fuer rekursive Determinantenberechnung
% (Laplace-Entwicklung nach der ersten Zeile)
function d = blatt1_mydet(A)
	if size(A,1) == 1
		d = A(1,1);
	else
		d = 0;
		% Rekursion nur wenn der Koeffizient nicht null ist.
		% Das kommt bei der Hilbert Matrix nicht vor, beschleunigt
		% aber das berechnen des Vorzeichens der Permutation
		% ganz btraechtlich!   => O(n) statt O(n!)
		for i = find(A(1,:) ~= 0)
			cofactor = (-1)^(i+1) * A(1,i) * ...
				blatt1_mydet(A(2:size(A,2), find([1:size(A,1)] ~= i)));
			d = d + cofactor;
		end
	end
end

