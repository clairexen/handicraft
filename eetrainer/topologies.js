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

// topology data structure:
// [ [knot1x, knot2x, ..], [knot1y, knot2y, ..], [edge1_knot1, edge2_knot1, ..], [edge1_knot2, edge2_knot2, ..] ];

function genTopologies()
{
	function K(x, y) {
		current_top[0].push(x);
		current_top[1].push(y);
		return current_top[0].length - 1;
	}

	function E() {
		for (var i = 1; i < arguments.length; i++) {
			current_top[2].push(arguments[i-1]);
			current_top[3].push(arguments[i]);
		}
	}

	function top0()
	{
		// knots: 2x3 -- indexes like on the numeric keypad
		var k4 = K(0, 1), k5 = K(1, 1), k6 = K(2, 1);
		var k1 = K(0, 0), k2 = K(1, 0), k3 = K(2, 0);

		// horizontal edges
		E(k4, k5, k6);
		E(k1, k2, k3);

		// vertical edges
		E(k4, k1);
		E(k5, k2);
		E(k6, k3);
	}

	function top1()
	{
		// knots: 3x3 -- indexes like on the numeric keypad
		var k7 = K(0, 2), k8 = K(1, 2), k9 = K(2, 2);
		var k4 = K(0, 1), k5 = K(1, 1), k6 = K(2, 1);
		var k1 = K(0, 0), k2 = K(1, 0), k3 = K(2, 0);

		// horizontal edges
		E(k7, k8, k9);
		E(k4, k5, k6);
		E(k1, k2, k3);

		// vertical edges
		E(k7, k4, k1);
		E(k8, k5, k2);
		E(k9, k6, k3);

		// diagonal edges (star)
		E(k7, k5, k3);
		E(k1, k5, k9);
	}

	function top2()
	{
		// knots: 3x3 -- indexes like on the numeric keypad
		var k7 = K(0, 2), k8 = K(1, 2), k9 = K(2, 2);
		var k4 = K(0, 1), k5 = K(1, 1), k6 = K(2, 1);
		var k1 = K(0, 0), k2 = K(1, 0), k3 = K(2, 0);

		// horizontal edges
		E(k7, k8, k9);
		E(k4, k5, k6);
		E(k1, k2, k3);

		// vertical edges
		E(k7, k4, k1);
		E(k8, k5, k2);
		E(k9, k6, k3);

		// diagonal edges (rhomb)
		E(k4, k8, k6);
		E(k4, k2, k6);
	}

	function top3()
	{
		// a 4x4 raster
		var k = { };
		for (var x = 0; x < 4; x++)
		for (var y = 0; y < 4; y++)
			k[x + "" + y] = K(x,y);
		for (var i = 0; i < 4; i++) {
			E(k[i+"3"], k[i+"2"], k[i+"1"], k[i+"0"]);
			E(k["0"+i], k["1"+i], k["2"+i], k["3"+i]);
		}
	}

	var tops = [];
	var current_top;
	var topfuncs = [ top0, top1, top2, top3 ];

	for (var i = 0; i < topfuncs.length; i++) {
		current_top = [ [], [], [], [] ];
		topfuncs[i]();
		tops.push(current_top);
	}

	return tops;
}

function paintTopology(canvas, topdata)
{
	var knots = topdata[0].length;
	var edges = topdata[2].length;
	var kx = topdata[0], ky = topdata[1];
	var e1 = topdata[2], e2 = topdata[3];
	var mx = kx[0], my = ky[0];

	for (var i = 1; i < knots; i++) {
		if (mx < kx[i])
			mx = kx[i];
		if (my < ky[i])
			my = ky[i];
	}

	canvas.setAttribute("width", 100+mx*100);
	canvas.setAttribute("height", 100+my*100);

	var ctx = canvas.getContext('2d');
	ctx.strokeStyle = "rgb(10,10,10)";
	ctx.fillStyle = "rgb(255,255,255)";
	ctx.fillRect(0, 0, 100+mx*100, 100+my*100);

	ctx.fillStyle = "rgb(10,10,10)";

	for (var i = 0; i < edges; i++)
	{
		var x1 = 50 + kx[e1[i]]*100, y1 = 50 + ky[e1[i]]*100;
		var x2 = 50 + kx[e2[i]]*100, y2 = 50 + ky[e2[i]]*100;
		var dx = (x2-x1), dy = (y2-y1), dl = Math.sqrt(dx*dx + dy*dy);
		ctx.beginPath();
		ctx.moveTo(x1, y1);
		ctx.lineTo(x2, y2);
		ctx.stroke();
		ctx.beginPath();
		ctx.moveTo(x1 + 0.7*dx - 3*dy/dl, y1 + 0.7*dy + 3*dx/dl);
		ctx.lineTo(x1 + 0.7*dx + 5*dx/dl, y1 + 0.7*dy + 5*dy/dl);
		ctx.lineTo(x1 + 0.7*dx + 3*dy/dl, y1 + 0.7*dy - 3*dx/dl);
		ctx.fill();
	}

	ctx.fillStyle = "rgb(255,255,255)";

	for (var i = 0; i < knots; i++)
	{
		ctx.beginPath();
		ctx.arc(50 + kx[i]*100, 50 + ky[i]*100, 5, 0, 2*Math.PI, false);
		ctx.stroke();
		ctx.fill();
	}

	ctx.font = "10pt Arial";
	ctx.textAlign = "center";
	ctx.fillStyle = "rgb(10,10,10)";

	for (var i = 0; i < edges; i++)
	{
		var x1 = 50 + kx[e1[i]]*100, y1 = 50 + ky[e1[i]]*100;
		var x2 = 50 + kx[e2[i]]*100, y2 = 50 + ky[e2[i]]*100;
		var dx = (x2-x1), dy = (y2-y1), dl = Math.sqrt(dx*dx + dy*dy);
		var x = 0.5*(x1+x2) - 10*dy/dl;
		var y = 0.5*(y1+y2) + 12*dx/dl;
		ctx.fillText("I"+i, x, y);
	}

	for (var i = 0; i < knots; i++) {
		ctx.fillText("U"+i, 35 + kx[i]*100, 42 + ky[i]*100);
	}
}

