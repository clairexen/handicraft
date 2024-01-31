% Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.

%% Spaltenweise Orthogonalisieren und Normalisieren
% der Vektor s beinhaltet jew. die Koeffizienten s_ij
% die im Gram-Schmidt-Verfahren auftreten. Die
% hinteren Elemente von s beinhalten nullen.
function A = blatt02(A)
    n = size(A,2);
    for i=1:n
        s = arrayfun(@(j) calc_s_ij(A, i, j), 1:n);
        w = A * s';
        A(:,i) = w / norm(w);
    end
end

%% Berechnung: Koeffizient s_ij fuer Gram-Schmidt
% Es steht in A in der Spalte j bereits der neue
% Vektor (w_j) und in der Spalte i der gerade zu
% orthogonalisierende Vektor (v_i). Im Fall j<i
% liefert die Funktion den Ausdruck gem. Skriptum
% seite 63 zurueck. Im Fall j=i wird 1 und im Fall
% j>i wird 0 zurueckgegeben.
function s_ij = calc_s_ij(A, i, j)
    if j > i
        s_ij = 0;
    elseif j == i
        s_ij = 1;
    else
	% In unserem Fall ist A(:,j) bereits normalisiert
	% d.h. die division ist eine division durch 1 und
	% koennte zur optimierung der performance und
	% numerischen stabilitaet auch weggelassen werden.
        s_ij = -dot(A(:,i), A(:,j)) / dot(A(:,j), A(:,j));
    end
end

%% Test case: copy&paste to command window
%{
A = rand(10, 5)
B = blatt02(A)
[ rank(A) rank(B) rank([A B]) ]
arrayfun(@(i,j) dot(B(:,i), B(:,j)), ...
    repmat([1:5], 5, 1), repmat([1:5]', 1, 5))
%}

%% Typo in der Angabe:
% wir sollen eine Funktion blatt03 implementieren..
function A = blatt03(A)
    A = blatt02(A);
end
