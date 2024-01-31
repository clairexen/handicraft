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

// circuit data structure:
// [ topology, [comp1_type, comp2_type, ..], [comp1_edge, comp2_edge, ..],
//   ground_knot, ref_knot, ref_current, solution ]
//
// with compN_type beeing a string like:
// "V+5" ...... a 5V voltage source
// "V-12" ..... a 12V voltage source with inverse polarity
// "A+10m" .... a 10mA current source
// "R100k" .... a 100 kOhm resistor
// "S" ........ a direct connection (short)

function getMagnitudeSuffix(mag)
{
	var suffix = "m";
	for (var i = -3; i < mag; i++)
		suffix = "0" + suffix;
	suffix = suffix.replace(/000m$/, "");
	suffix = suffix.replace(/000$/, "k");
	return suffix;
}

function genRandomMagnitudes()
{
	var vmag = pseudorand(2);
	var rmag = pseudorand(3)+vmag;
	var amag = vmag - rmag;
	return [ vmag, rmag, amag ];
}

function genRandomComponent(mags)
{
	var type = pseudorand(11);
	if (type < 1) {
		var val = (1+pseudorand(9));
		return "V+" + val + getMagnitudeSuffix(mags[0]);
	}
	if (type < 2) {
		var val = (1+pseudorand(9));
		return "V-" + val + getMagnitudeSuffix(mags[0]);
	}
	if (type < 3) {
		var val = (1+pseudorand(9));
		return "A+" + val + getMagnitudeSuffix(mags[2]);
	}
	if (type < 4) {
		var val = (1+pseudorand(9));
		return "A-" + val + getMagnitudeSuffix(mags[2]);
	}
	if (type < 8) {
		var val = (1+pseudorand(9));
		return "R" + val + getMagnitudeSuffix(mags[1]);
	}
	return "S";
}

function genRandomCircuit(topdata, components)
{
	var circuit = [ topdata, [], [], -1, -1, -1, 0 ];
	var edges = topdata[2].length;
	var ke1 = topdata[2], ke2 = topdata[3];
	var e = [], k = [];

	for (var i = 0; i < edges; i++) {
		e.push(i);
		k[ke1[i]] = 0;
		k[ke2[i]] = 0;
	}
	for (var i = 0; i < edges; i++) {
		var j = pseudorand(edges);
		var tmp = e[i];
		e[i] = e[j];
		e[j] = tmp;
	}
	for (var i = 0; i < edges && i < components.length; i++) {
		circuit[1].push(components[i]);
		circuit[2].push(e[i]);
		k[ke1[e[i]]]++;
		k[ke2[e[i]]]++;
	}

	// add shorts to close loose ends
	var remedges = [];
	for (var i = components.length; i < edges; i++)
		remedges.push(e[i]);
	var done = false;
	while (!done) {
		done = true;
		for (var i in k) {
			if (k[i] != 1)
				continue;
			var j = -1;
			for (var l = 0; l < remedges.length; l++) {
				var ed = remedges[l];
				if (ed < 0)
					continue;
				if (ke1[ed] == i) {
					j = l;
					if (k[ke2[ed]] != 0)
						break;
				}
				if (ke2[ed] == i) {
					j = l;
					if (k[ke1[ed]] != 0)
						break;
				}
			}
			if (j < 0)
				throw "Found loose end in topology!";
			circuit[1].push("S");
			circuit[2].push(remedges[j]);
			k[ke1[remedges[j]]]++;
			k[ke2[remedges[j]]]++;
			remedges[j] = -1;
			done = false;
		}
	}

	// randomly select ground knot
	var klist = [];
	for (var i in k)
		if (k[i] > 0)
			klist.push(i);
	circuit[3] = klist[pseudorand(klist.length)];

	return circuit;
}

function paintCircuit(canvas, circuit)
{
	var topdata = circuit[0];
	var components = circuit[1].length;
	var ct = circuit[1], ce = circuit[2];
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

	ctx.font = "10pt Arial";
	ctx.textAlign = "center";

	var kcount = {};

	for (var i = 0; i < components; i++)
	{
		kcount[e1[ce[i]]] = e1[ce[i]] in kcount ? kcount[e1[ce[i]]] + 1 : 1;
		kcount[e2[ce[i]]] = e2[ce[i]] in kcount ? kcount[e2[ce[i]]] + 1 : 1;

		var x1 = 50 + kx[e1[ce[i]]]*100;
		var y1 = 50 + ky[e1[ce[i]]]*100;
		var x2 = 50 + kx[e2[ce[i]]]*100;
		var y2 = 50 + ky[e2[ce[i]]]*100;
		var comptype = ct[i];

		ctx.save();
		ctx.translate(x1, y1);
		ctx.rotate(+Math.atan2(y2-y1, x2-x1));
		var len = Math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));

		if (comptype == "S") {
			ctx.beginPath();
			ctx.moveTo(0, 0);
			ctx.lineTo(len, 0);
			ctx.stroke();
		}
		else if (comptype.match(/^R([0-9]*)(k?)/)) {
			ctx.beginPath();
			ctx.moveTo(0, 0);
			ctx.lineTo(len/2-20, 0);
			ctx.moveTo(len/2-20, -5);
			ctx.lineTo(len/2-20, +5);
			ctx.lineTo(len/2+20, +5);
			ctx.lineTo(len/2+20, -5);
			ctx.lineTo(len/2-20, -5);
			ctx.moveTo(len/2+20, 0);
			ctx.lineTo(len, 0);
			ctx.stroke();
			ctx.fillStyle = "rgb(10,10,10)";
			ctx.fillText(RegExp.$1 + " " + RegExp.$2 + String.fromCharCode(937), len/2, -10);
		}
		else if (comptype.match(/^V([+-])(.*)/)) {
			ctx.beginPath();
			ctx.moveTo(0, 0);
			ctx.lineTo(len, 0);
			ctx.stroke();
			ctx.beginPath();
			ctx.arc(len/2, 0, 13, 0, 2*Math.PI, false);
			ctx.stroke();
			ctx.fillStyle = "rgb(10,10,10)";
			if (RegExp.$1 == "+")
				ctx.fillText("+", len/2+17, -3);
			else
				ctx.fillText("+", len/2-17, -3);
			ctx.fillText(RegExp.$2 + " V", len/2, -17);
		}
		else if (comptype.match(/^A([+-])([0-9]*)(m?)/)) {
			ctx.beginPath();
			ctx.moveTo(0, 0);
			ctx.lineTo(len/2-13, 0);
			ctx.moveTo(len/2, 13);
			ctx.lineTo(len/2, -13);
			ctx.moveTo(len/2+13, 0);
			ctx.lineTo(len, 0);
			ctx.stroke();
			ctx.beginPath();
			ctx.arc(len/2, 0, 13, 0, 2*Math.PI, false);
			ctx.stroke();
			ctx.fillStyle = "rgb(10,10,10)";
			if (RegExp.$1 == "+")
				ctx.fillText("+", len/2+17, -3);
			else
				ctx.fillText("+", len/2-17, -3);
			ctx.fillText(RegExp.$2 + " " + RegExp.$3 + "A", len/2, -17);
		}
		else {
			throw "Unkown component type: " + comptype;
		}

		ctx.restore();
	}

	ctx.fillStyle = "rgb(10,10,10)";
	for (i in kcount) {
		if (kcount[i] <= 2)
			continue;
		var x = 50 + kx[i]*100;
		var y = 50 + ky[i]*100;
		ctx.beginPath();
		ctx.arc(x, y, 2, 0, 2*Math.PI, false);
		ctx.fill();
	}

	if (circuit[3] >= 0 && circuit[4] >= 0) {
		var x = 50 + kx[circuit[3]]*100;
		var y = 50 + ky[circuit[3]]*100;
		ctx.fillStyle = "rgb(255,255,255)";
		ctx.beginPath();
		ctx.arc(x, y, 5, 0, 2*Math.PI, false);
		ctx.stroke();
		ctx.fill();
		ctx.fillStyle = "rgb(10,10,10)";
		ctx.fillText("B", x-7, y-7);
	}

	if (circuit[3] >= 0 && circuit[4] >= 0) {
		var x = 50 + kx[circuit[4]]*100;
		var y = 50 + ky[circuit[4]]*100;
		ctx.fillStyle = "rgb(255,255,255)";
		ctx.beginPath();
		ctx.arc(x, y, 5, 0, 2*Math.PI, false);
		ctx.stroke();
		ctx.fill();
		ctx.fillStyle = "rgb(10,10,10)";
		ctx.fillText("A", x-7, y-7);
	}

	if (circuit[5] >= 0) {
		var x1 = 50 + kx[e1[circuit[5]]]*100, y1 = 50 + ky[e1[circuit[5]]]*100;
		var x2 = 50 + kx[e2[circuit[5]]]*100, y2 = 50 + ky[e2[circuit[5]]]*100;
		var dx = (x2-x1), dy = (y2-y1), dl = Math.sqrt(dx*dx + dy*dy);
		ctx.beginPath();
		ctx.fillStyle = "rgb(10,10,10)";
		ctx.moveTo(x1 + 0.8*dx - 5*dy/dl, y1 + 0.8*dy + 5*dx/dl);
		ctx.lineTo(x1 + 0.8*dx + 10*dx/dl, y1 + 0.8*dy + 10*dy/dl);
		ctx.lineTo(x1 + 0.8*dx + 5*dy/dl, y1 + 0.8*dy - 5*dx/dl);
		ctx.fill();
	}
}

