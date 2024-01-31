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

function findGoodProblem(circuit, logger)
{
	function myClone(d) {
		if (d instanceof Array) {
			var c = [];
			for (var i = 0; i < d.length; i++)
				c.push(myClone(d[i]));
			return c;
		}
		return d;
	}

	function myIsEmpty(h) {
		for (var i in h)
			return false;
		return true;
	}

	// check if system can be solved
	var eqsys = generateEquationSystem(circuit, true);
	if (!solveEquationSystem(eqsys)) {
		if (logger != null)
			logger("Failed to solve circuit in the first place.");
		return false;
	}
	var sol = expandSolution(eqsys);
	if (sol === null) {
		if (logger != null)
			logger("Failed to solve circuit uniquely in the first place.");
		return false;
	}

	// safty check: re-evaluate without optimizer
	if (0) {
		var eqsys2 = generateEquationSystem(circuit, false);
		if (!solveEquationSystem(eqsys2))
			throw "Found optimizer error!";
		var sol2 = expandSolution(eqsys2);
		if (sol2 === null)
			throw "Found optimizer error!";
		for (var i in sol)
			if (Math.abs(sol[i]-sol2[i]) > 1e-6)
				throw "Found optimizer error!";
	}

	// we are only interested in non-zero values
	var dellist = [];
	for (var j in sol) {
		if (Math.abs(sol[j]) < 1e-3)
			dellist.push(j);
	}
	for (var j in dellist) {
		delete sol[dellist[j]];
	}
	if (myIsEmpty(sol)) {
		if (logger != null)
			logger("Nothing of interest to ask for.");
		return false;
	}

	// extract non-short components and generate alternative values
	// we are only interested in problems that depend on all components
	var complist_idx = [], complist_alt = [];
	for (var i = 0; i < circuit[1].length; i++) {
		var a = circuit[1][i];
		if (a != "S") {
			a = a.replace(/[0-9]+/, function(num) { return num + "5"; });
			complist_idx.push(i);
			complist_alt.push(a);
		}
	}

	// circuits with less than 3 real components are boring
	if (complist_idx.length < 3) {
		if (logger != null)
			logger("Boring component count.");
		return false;
	}

	// check variations of the circuit
	// count number of components that a sol entry is independent
	// of. (this way we don't end up asking for stuff like the
	// current thru a current source)
	var sol_indep = { };
	for (var j in sol)
		sol_indep[j] = 0;
	for (var i = 0; i < complist_idx.length; i++) {
		var altcirc = myClone(circuit);
		altcirc[1][complist_idx[i]] = complist_alt[i];
		var alteqsys = generateEquationSystem(altcirc, true);
		if (!solveEquationSystem(alteqsys)) {
			if (logger != null)
				logger("Failed to solve circuit for " + complist_alt[i] + " at I" + circuit[2][complist_idx[i]]);
			return false;
		}
		var altsol = expandSolution(alteqsys);
		if (altsol === null) {
			if (logger != null)
				logger("Failed to solve circuit uniquely for " + complist_alt[i] + " at I" + circuit[2][complist_idx[i]]);
			return false;
		}
		var dellist = [];
		for (var j in sol) {
			if (Math.abs(sol[j]-altsol[j]) < 1e-3) {
				if (logger != null)
					logger("Probably independent: " + j + " and " +
							circuit[1][complist_idx[i]] +
							" at I" + circuit[2][complist_idx[i]] +
							" #" + sol_indep[j]);
				if (++sol_indep[j] > 1)
					dellist.push(j);
			}
		}
		for (var j in dellist) {
			delete sol[dellist[j]];
			delete sol_indep[dellist[j]];
		}
		if (myIsEmpty(sol)) {
			if (logger != null)
				logger("Found independent parameters for all components.");
			return false;
		}
	}

	// do we have a current that depends on all component values?
	// if so (and the circuit is large enough) we prefer problems
	// that depend on all but one component values.
	var fully_connected = 0;
	var use_indep1_sol = 0;
	for (var i in sol) {
		if (sol_indep[i] == 0 && i.match(/^I/))
			fully_connected = 1;
		if (sol_indep[i] == 1)
			use_indep1_sol = 1;
	}
	if (complist_idx.length < 5)
		use_indep1_sol = 0;
	if (!fully_connected)
		use_indep1_sol = 0;
	if (logger != null) {
		logger(fully_connected ? "Circuit is fully connected using closed loops." :
				"Circuit may not be fully connected using closed loops.");
		if (use_indep1_sol)
			logger("Using problem that's independent of one of the components.");
	}

	// still with us? so lets randomly select a question
	var questions = [];
	for (var i in sol) {
		if (sol_indep[i] == use_indep1_sol)
			questions.push(i);
	}
	if (questions.length == 0) {
		if (logger != null)
			logger("No remaining questions to ask.");
		return false;
	}
	if (logger != null)
		logger("Remaining questions: " + questions.toString());
	var q = questions[pseudorand(questions.length)];

	// write all information about the question to the circuit description
	if (q.match(/^I([0-9]+)/)) {
		circuit[4] = -1;
		circuit[5] = parseInt(RegExp.$1);
		circuit[6] = sol[q];
		if (logger != null)
			logger("What is the current " + q+ "?");
		return true;
	}
	if (q.match(/^U([0-9]+)/)) {
		circuit[4] = parseInt(RegExp.$1);
		circuit[5] = -1;
		circuit[6] = sol[q];
		if (logger != null)
			logger("What is the voltage " + q + " relative to U" + circuit[3] + "?");
		return true;
	}
	return false;
}

