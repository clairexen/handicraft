/*
 *  EE-Trainer -- a JavaScript tool for training simple network analysis
 *
 *  Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation; either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

// eq-system data structure:
// [ [ [ var, alias1, sign1, alias2, sign2, ..], .. ],
//   [ var1, const1, var2, const2, ..],
//   [ [ comment, const, var1, koeff1, var2, koeff2, ..], .. ],
//   [ var1, solution1, var2, solution2, .. ]

function generateEquationSystem(circuit, optimize)
{
	var eqsys = [ [], [], [], [] ];
	var e2k1 = circuit[0][2];
	var e2k2 = circuit[0][3];
	var comp = circuit[1];
	var edge = circuit[2];

	// knot equations
	var knoteqs = { };
	var largest_knoteq = 0;
	for (var i = 0; i < comp.length; i++)
	{
		var e = edge[i], k1 = e2k1[e], k2 = e2k2[e];

		if (!(k1 in knoteqs)) {
			knoteqs[k1] = eqsys[2].length;
			eqsys[2].push([ "Knot "+k1, 0 ]);
		}
		if (!(k2 in knoteqs)) {
			knoteqs[k2] = eqsys[2].length;
			eqsys[2].push([ "Knot "+k2, 0 ]);
		}

		eqsys[2][knoteqs[k1]].push("I"+e);
		eqsys[2][knoteqs[k1]].push(-1);
		eqsys[2][knoteqs[k2]].push("I"+e);
		eqsys[2][knoteqs[k2]].push(+1);

		if (eqsys[2][knoteqs[k1]].length > eqsys[2][largest_knoteq].length)
			largest_knoteq = knoteqs[k1];

		if (eqsys[2][knoteqs[k2]].length > eqsys[2][largest_knoteq].length)
			largest_knoteq = knoteqs[k2];
	}

	// throw away largest knot equation
	eqsys[2].splice(largest_knoteq, 1);

	// add simple equation for reference ground
	eqsys[2].push([ "Ground", 0, "U"+circuit[3], 1 ]);

	// edge equations
	for (var i = 0; i < comp.length; i++)
	{
		var c = comp[i], e = edge[i];
		var k1 = e2k1[e], k2 = e2k2[e];
		if (c == "S") {
			eqsys[2].push([ "Short @I"+e, 0, "U"+k1, 1, "U"+k2, -1 ]);
		}
		else if (c.match(/^R([0-9]*)(k?)/)) {
			var val = parseInt(RegExp.$1) * (RegExp.$2 == "k" ? 1000 : 1);
			eqsys[2].push([ "Resistor @I"+e, 0, "U"+k1, 1, "U"+k2, -1, "I"+e, -val ]);
		}
		else if (c.match(/^V([+-])(.*)/)) {
			var val = parseInt(RegExp.$2) * (RegExp.$1 == "+" ? +1 : -1);
			eqsys[2].push([ "VoltageSource @I"+e, val, "U"+k2, 1, "U"+k1, -1 ]);
		}
		else if (c.match(/^A([+-])([0-9]*)(m?)/)) {
			var val = parseInt(RegExp.$2) * (RegExp.$1 == "+" ? +1 : -1);
			val *= (RegExp.$3 == "m" ? 0.001 : 1);
			eqsys[2].push([ "CurrentSource @I"+e, val, "I"+e, 1 ]);
		}
		else {
			throw "Unkown component type: " + c;
		}
	}

	// actually we are done here.
	// the rest of this function just simplifies the equation system.
	if (!optimize)
		return eqsys;

	// extract aliases from equation system
	var aliasidx = {}, aliassign = {};
	for (var i = 0; i < eqsys[2].length; i++)
	{
		if (eqsys[2][i].length != 6 || eqsys[2][i][1] != 0 ||
				Math.abs(eqsys[2][i][3]) != Math.abs(eqsys[2][i][5]))
			continue;
		var sign = eqsys[2][i][3] * eqsys[2][i][5] > 0 ? -1 : +1;
		var var1 = eqsys[2][i][2], var2 = eqsys[2][i][4];
		if ((var1 in aliasidx) && (var2 in aliasidx)) {
			if (aliasidx[var1] != aliasidx[var2]) {
				var list1 = aliasidx[var1], list2 = aliasidx[var2];
				sign = aliassign[var1] == aliassign[var2] ? +sign : -sign;
				var mergelist = eqsys[0][list2];
				mergelist.splice(1, 0, 1);
				for (var j = 0; j < mergelist.length; j += 2) {
					eqsys[0][list1].push(mergelist[j]);
					eqsys[0][list1].push(sign * mergelist[j+1]);
					aliasidx[mergelist[j]] = list1;
					aliassign[mergelist[j]] *= sign;
				}
				eqsys[0].splice(list2, 1);
				for (j in aliasidx) {
					if (aliasidx[j] > list2)
						aliasidx[j]--;
				}
			}
		} else if (var1 in aliasidx) {
			sign *= aliassign[var1];
			eqsys[0][aliasidx[var1]].push(var2);
			eqsys[0][aliasidx[var1]].push(sign);
			aliasidx[var2] = aliasidx[var1];
			aliassign[var2] = sign;
		} else if (var2 in aliasidx) {
			sign *= aliassign[var2];
			eqsys[0][aliasidx[var2]].push(var1);
			eqsys[0][aliasidx[var2]].push(sign);
			aliasidx[var1] = aliasidx[var2];
			aliassign[var1] = sign;
		} else {
			aliasidx[var1] = eqsys[0].length;
			aliasidx[var2] = eqsys[0].length;
			aliassign[var1] = +1;
			aliassign[var2] = sign;
			eqsys[0].push([ var1, var2, sign ]);
		}
		eqsys[2].splice(i--, 1);
	}

	// substitute aliases in equation system
	for (var i = 0; i < eqsys[2].length; i++)
	for (var j = 2; j < eqsys[2][i].length; j += 2) {
		var v = eqsys[2][i][j];
		if (v in aliasidx) {
			eqsys[2][i][j] = eqsys[0][aliasidx[v]][0];
			eqsys[2][i][j+1] *= aliassign[v];
		}
	}

	// find and fold duplicates in equation system
	for (var i = 0; i < eqsys[2].length; i++)
	for (var j = 2; j < eqsys[2][i].length; j += 2)
	{
		var v = eqsys[2][i][j];
		for (var k = j + 2; k < eqsys[2][i].length; k += 2) {
			if (v != eqsys[2][i][k])
				continue;
			eqsys[2][i][j+1] += eqsys[2][i][k+1];
			eqsys[2][i].splice(k, 2);
			k -= 2;
		}
		if (eqsys[2][i][j+1] == 0) {
			eqsys[2][i].splice(j, 2);
			j -= 2;
		}
	}

	// find and fold constants in equation system
	var done_constfold = false;
	while (!done_constfold)
	{
		done_constfold = true;
		for (var i = 0; i < eqsys[2].length; i++)
		{
			if (eqsys[2][i].length != 4)
				continue;

			var v = eqsys[2][i][2];
			var c = eqsys[2][i][1] / eqsys[2][i][3];
			eqsys[1].push(v);
			eqsys[1].push(c);
			eqsys[2].splice(i--, 1);
			done_constfold = false;

			for (var j = 0; j < eqsys[2].length; j++)
			for (var k = 2; k < eqsys[2][j].length; k += 2) {
				if (eqsys[2][j][k] != v)
					continue;
				eqsys[2][j][1] -= c * eqsys[2][j][k+1];
				eqsys[2][j].splice(k, 2);
				k -= 2;
			}
		}
	}

	return eqsys;
}

function solveEquationSystem(eqsys)
{
	var eqs = eqsys[2];
	var sol = [];
	var n = eqs.length;
	eqsys[3] = [];

	// sometimes the optimizer in generateEquationSystem
	// can solve already the system by const folding
	if (n == 0)
		return true;

	// we are going to solve Ax = y
	var A = NumJS.MAT(n, n);
	var y = NumJS.MAT(n, 1);
	var x;

	// map variable names to positions in x-vector
	var next_vi_idx = 0;
	var vi = {};

	// create A and y
	for (var i = 0; i < n; i++) {
		y.set(i, 0, eqs[i][1]);
		for (var j = 2; j < eqs[i].length; j += 2) {
			var v = eqs[i][j], k = eqs[i][j+1];
			if (!(v in vi)) {
				if (next_vi_idx >= n)
					return false;
				vi[v] = next_vi_idx++;
			}
			A.set(i, vi[v], NumJS.ADD(k, A.get(i, vi[v])));
		}
	}

	// just abort if we left zero columns in the matrix
	if (next_vi_idx < n)
		return false;

	// solve it!
	x = NumJS.SOLVE(A, y);
	if (x == null)
		return false;

	// store results
	for (v in vi) {
		eqsys[3].push(v);
		eqsys[3].push(x.get(vi[v], 0));
	}

	return true;
}

function expandSolution(eqsys)
{
	var sol = { };

	// copy constants
	for (var i = 0; i < eqsys[1].length; i += 2)
		sol[eqsys[1][i]] = eqsys[1][i+1];

	// copy actual solutions
	for (var i = 0; i < eqsys[3].length; i += 2)
		sol[eqsys[3][i]] = eqsys[3][i+1];

	// expand aliases
	for (var i = 0; i < eqsys[0].length; i++)
	for (var j = 1; j < eqsys[0][i].length; j += 2)
		sol[eqsys[0][i][j]] = eqsys[0][i][j+1] * sol[eqsys[0][i][0]];

	// check if we actually have a valid solution
	// (sometimes a branch like "0 = 0*In" gets removed from the
	// equation system leaving a set of aliases with no solutions)
	for (var i in sol) {
		if (typeof(sol[i]) != "number")
			return null;
		if (isNaN(sol[i]))
			return null;
	}

	return sol;
}

function eqsysToString(eqsys)
{
	var eqtxt = "";
	for (var j = 0; j < eqsys[0].length; j++) {
		eqtxt += eqsys[0][j][0];
		for (var k = 1; k < eqsys[0][j].length; k += 2)
			eqtxt += " = " + (eqsys[0][j][k+1] > 0 ? "" : "-") + eqsys[0][j][k];
		eqtxt += "\n";
	}
	if (eqsys[0].length > 0)
		eqtxt += "\n";
	for (var j = 0; j < eqsys[1].length; j += 2)
		eqtxt += eqsys[1][j] + " = " + eqsys[1][j+1] + "\n";
	if (eqsys[1].length > 0)
		eqtxt += "\n";
	for (var j = 0; j < eqsys[2].length; j++) {
		eqtxt += eqsys[2][j][1] + " =";
		for (var k = 2; k < eqsys[2][j].length; k += 2) {
			eqtxt += eqsys[2][j][k+1] < 0 ? " -" : " +";
			if (Math.abs(eqsys[2][j][k+1]) != 1)
				eqtxt += Math.abs(eqsys[2][j][k+1]) + "*";
			eqtxt += eqsys[2][j][k];
		}
		if (eqsys[2][j].length == 2)
			eqtxt += " 0";
		eqtxt += "   // " + eqsys[2][j][0] + "\n";
	}
	if (eqsys[2].length > 0)
		eqtxt += "\n";
	for (var j = 0; j < eqsys[3].length; j += 2) {
		eqtxt += eqsys[3][j] + " = " + eqsys[3][j+1] + "\n";
	}
	if (eqsys[3].length > 0)
		eqtxt += "\n";
	return eqtxt;
}

