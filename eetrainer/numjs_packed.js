/*
 *  NumJS -- a JavaScript library for numerical computing
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


// Git: 5e89927
// Modules: GenOps Parser Cmplx Matrix MatRREF MatLU
var NumJS = new Object();


/*
 *  NumJS -- a JavaScript library for numerical computing
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


NumJS.eps = (function(){
	var i = 0;
	while (1 + Math.pow(2, -i) != 1)
		i++;
	return Math.pow(2, -i/2);
})();

NumJS.ADD = function(a, b) {
	if (typeof(a.op_add) == "function")
		return a.op_add(a, b);
	if (typeof(b.op_add) == "function")
		return b.op_add(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return a + b;
	throw "NumJS.GenOps type error";
};

NumJS.SUB = function(a, b) {
	if (typeof(a.op_sub) == "function")
		return a.op_sub(a, b);
	if (typeof(b.op_sub) == "function")
		return b.op_sub(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return a - b;
	throw "NumJS.GenOps type error";
};

NumJS.MUL = function(a, b) {
	if (typeof(a.op_mul) == "function")
		return a.op_mul(a, b);
	if (typeof(b.op_mul) == "function")
		return b.op_mul(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return a * b;
	throw "NumJS.GenOps type error";
};

NumJS.DOT = function(a, b) {
	if (typeof(a.op_mul) == "function")
		return a.op_dot(a, b);
	if (typeof(b.op_mul) == "function")
		return b.op_dot(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return a * b;
	throw "NumJS.GenOps type error";
};

NumJS.DIV = function(a, b) {
	if (typeof(a.op_div) == "function")
		return a.op_div(a, b);
	if (typeof(b.op_div) == "function")
		return b.op_div(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return a / b;
	throw "NumJS.GenOps type error";
};

NumJS.SOLVE = function(a, b) {
	if (typeof(a.op_solve) == "function")
		return a.op_solve(a, b);
	if (typeof(b.op_solve) == "function")
		return b.op_solve(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return b / a;
	throw "NumJS.GenOps type error";
};

NumJS.INV = function(a) {
	if (typeof(a.op_inv) == "function")
		return a.op_inv(a);
	if (typeof(a) == "number")
		return 1 / a;
	throw "NumJS.GenOps type error";
};

NumJS.NEG = function(a) {
	if (typeof(a.op_neg) == "function")
		return a.op_neg(a);
	if (typeof(a) == "number")
		return -a;
	throw "NumJS.GenOps type error";
};

NumJS.ABS = function(a) {
	if (typeof(a.op_abs) == "function")
		return a.op_abs(a);
	if (typeof(a) == "number")
		return Math.abs(a);
	throw "NumJS.GenOps type error";
};

NumJS.NORM = function(a) {
	if (typeof(a.op_norm) == "function")
		return a.op_norm(a);
	if (typeof(a) == "number")
		return Math.abs(a);
	throw "NumJS.GenOps type error";
};

NumJS.ARG = function(a) {
	if (typeof(a.op_arg) == "function")
		return a.op_arg(a);
	if (typeof(a) == "number")
		return a >= 0 ? 0 : Math.PI;
	throw "NumJS.GenOps type error";
};

NumJS.CONJ = function(a) {
	if (typeof(a.op_conj) == "function")
		return a.op_conj(a);
	if (typeof(a) == "number")
		return a;
	throw "NumJS.GenOps type error";
};

NumJS.TRANSP = function(a) {
	if (typeof(a.op_transp) == "function")
		return a.op_transp(a);
	if (typeof(a) == "number")
		return a;
	throw "NumJS.GenOps type error";
};

NumJS.POW = function(a, b) {
	if (typeof(a.op_pow) == "function")
		return a.op_pow(a, b);
	if (typeof(b.op_pow) == "function")
		return b.op_pow(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return Math.pow(a, b);
	throw "NumJS.GenOps type error";
};

NumJS.EXP = function(a) {
	if (typeof(a.op_exp) == "function")
		return a.op_exp(a);
	if (typeof(a) == "number")
		return Math.exp(a);
	throw "NumJS.GenOps type error";
};

NumJS.LOG = function(a) {
	if (typeof(a.op_log) == "function")
		return a.op_log(a);
	if (typeof(a) == "number")
		return Math.log(a);
	throw "NumJS.GenOps type error";
};

NumJS.DET = function(a) {
	if (typeof(a.op_det) == "function")
		return a.op_det(a);
	if (typeof(a) == "number")
		return a;
	throw "NumJS.GenOps type error";
};

NumJS.RE = function(a) {
	if (typeof(a.op_re) == "function")
		return a.op_re(a);
	if (typeof(a) == "number")
		return a;
	throw "NumJS.GenOps type error";
};

NumJS.IM = function(a) {
	if (typeof(a.op_im) == "function")
		return a.op_im(a);
	if (typeof(a) == "number")
		return 0;
	throw "NumJS.GenOps type error";
};

NumJS.ROUND = function(a, n) {
	if (typeof(n) == "undefined")
		n = 0;
	if (typeof(a.op_round) == "function")
		return a.op_round(a, n);
	if (typeof(a) == "number") {
		var factor = Math.pow(10, n);
		return Math.round(a * factor) / factor;
	}
	throw "NumJS.GenOps type error";
};

NumJS.EQ = function(a, b) {
	if (typeof(a.op_eq) == "function")
		return a.op_eq(a, b);
	if (typeof(b.op_eq) == "function")
		return b.op_eq(a, b);
	if (typeof(a) == "number" && typeof(b) == "number")
		return a == b;
	throw "NumJS.GenOps type error";
};

NumJS.EQ_ABS = function(a, b, d) {
	if (typeof(a.op_eq) == "function")
		return a.op_eq_abs(a, b, d);
	if (typeof(b.op_eq) == "function")
		return b.op_eq_abs(a, b, d);
	if (typeof(a) == "number" && typeof(b) == "number")
		return Math.abs(a - b) <= d;
	throw "NumJS.GenOps type error";
};

NumJS.EQ_REL = function(a, b, d) {
	if (typeof(a.op_eq) == "function")
		return a.op_eq_rel(a, b, d);
	if (typeof(b.op_eq) == "function")
		return b.op_eq_rel(a, b, d);
	if (typeof(a) == "number" && typeof(b) == "number")
		return Math.abs(a - b) <= d * (Math.abs(a) + Math.abs(b)) * 0.5;
	throw "NumJS.GenOps type error";
};

/*
 *  NumJS -- a JavaScript library for numerical computing
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


NumJS.Parse = function(text, args, defs)
{
	var code = "(function(" + (typeof(args) != "undefined" ? args : "") + "){\n";
	var idx = 0;

	if (typeof(defs) != "undefined") {
		for (i in defs)
			code += "var " + i + " = " + defs[i] + ";\n";
	}

	// lexical pass: used to identify literals and extract 'non-trivial'
	// literals. I.e. literals that contain '+' and '-' such as "1.23e-5"
	// that would confuse the actual parser.
	var tokens = text.match(/(\.?[0-9][0-9.]*e[+-]?[0-9]+i?|[a-zA-Z_0-9.]+|.)/g);
	text = "";
	for (var i in tokens) {
		if (tokens[i].search(/^\.?[0-9][0-9.]*e[+-]?[0-9]+i?$/) >= 0) {
			text += "$" + idx;
			code += "var $" + (idx++) + " = ";
			if (tokens[i].search(/(.*)i$/) >= 0)
				code += "NumJS.C(0, " + RegExp.$1 + ");\n";
			else
				code += tokens[i] + ";\n";
		} else
			text += tokens[i];
	}

	// this function reduces a textual expression to
	// a single variable index and returns the corrspondig
	// basic "$n" expression, writing the actual js code
	// to the code variable on the way.
	function reduce(text)
	{
		var pos;
		while (1)
		{
			// blocks and function calls
			pos = text.search(/[a-zA-Z_0-9.]*\(/);
			if (pos >= 0)
			{
				var prefix = text.slice(0, pos);
				var postfix = text.slice(pos);
				var funcname = postfix.match(/^[a-zA-Z_0-9.]*/)[0];
				postfix = postfix.slice(funcname.length + 1);

				var i, pcount = 1;
				for (i = 0; pcount > 0; i++) {
					if (i >= postfix.length)
						throw "NumJS.Parse parser error.";
					var ch = postfix.substr(i, 1);
					if (ch == "(")
						pcount++;
					if (ch == ")")
						pcount--;
				}
				var inner = postfix.slice(0, i-1);
				postfix = postfix.slice(i);

				// handle a function call
				if (funcname != "")
				{
					var fcall = funcname + "(";
					if (inner != "") {
						var start = 0;
						for (i = 0; i < inner.length; i++) {
							var ch = inner.substr(i, 1);
							if (ch == "(")
								pcount++;
							if (ch == ")")
								pcount--;
							if (ch == "," && pcount == 0) {
								var arg = inner.slice(start, i);
								if (start != 0)
									fcall += ", ";
								fcall += reduce(arg);
								start = i + 1;
							}
						}
						var arg = inner.slice(start, i);
						if (start != 0)
							fcall += ", ";
						fcall += reduce(arg);
					}

					text = prefix + "$" + idx + postfix;
					code += "var $" + (idx++) + " = " + fcall + ");\n";
					continue;
				}

				text = prefix + reduce(inner) + postfix;
				continue;
			}

			// matrix notation
			pos = text.search(/\[/);
			if (pos >= 0)
			{
				var prefix = text.slice(0, pos);
				var postfix = text.slice(pos + 1);

				var i, pcount = 1;
				for (i = 0; pcount > 0; i++) {
					if (i >= postfix.length)
						throw "NumJS.Parse parser error.";
					var ch = postfix.substr(i, 1);
					if (ch == "[")
						pcount++;
					if (ch == "]")
						pcount--;
				}
				var inner = postfix.slice(0, i-1);
				postfix = postfix.slice(i);

				var start = 0;
				var cellData = new Object();
				var rowNum = 0, colNum = 0;
				var rowId = 0, colId = 0;
				function addCell(txt, term) {
					if (rowId >= rowNum)
						rowNum = rowId + 1;
					if (colId >= colNum)
						colNum = colId + 1;
					cellData[rowId + " " + colId] = reduce(txt);
					if (term == ",")
						colId++;
					if (term == ";")
						rowId++, colId = 0;
				}
				for (i = 0; i < inner.length; i++) {
					var ch = inner.substr(i, 1);
					if (ch == "[")
						pcount++;
					if (ch == "]")
						pcount--;
					if ((ch == "," || ch == ";") && pcount == 0) {
						addCell(inner.slice(start, i), ch);
						start = i + 1;
					}
				}
				if (start != i)
					addCell(inner.slice(start, i), ch);

				code += "var $" + idx + " = NumJS.MAT(" + rowNum + ", " + colNum + ", [";
				for (var i = 0; i < rowNum; i++)
				for (var j = 0; j < colNum; j++) {
					if (i != 0 || j != 0)
						code += ", ";
					if ((i + " " + j) in cellData)
						code +=  cellData[i + " " + j];
					else
						code += "0";
				}
				code += " ]);\n";

				text = prefix + "$" + (idx++) + postfix;
				continue;
			}

			// prefix '+' and '-'
			pos = text.search(/(^|[^$a-zA-Z_0-9.])(\+|-)([$a-zA-Z_0-9.]+)/);
			if (pos >= 0)
			{
				pos += RegExp.$1.length;
				var prefix = text.slice(0, pos);
				var op = RegExp.$2
				pos += op.length;
				var val = RegExp.$3
				pos += val.length;
				var postfix = text.slice(pos);

				val = reduce(val);

				text = prefix + "$" + idx + postfix;
				code += "var $" + (idx++) + " = ";
				if (op == "+")
					code += val + ";\n";
				if (op == "-")
					code += "NumJS.NEG(" + val + ");\n";
				continue;
			}

			// multiply and divide
			pos = text.search(/([$a-zA-Z_0-9.]+)(\*|\/|\\)([$a-zA-Z_0-9.]+)/);
			if (pos >= 0)
			{
				var prefix = text.slice(0, pos);
				var val1 = RegExp.$1;
				pos += val1.length;
				var op = RegExp.$2
				pos += op.length;
				var val2 = RegExp.$3
				pos += val2.length;
				var postfix = text.slice(pos);

				val1 = reduce(val1);
				val2 = reduce(val2);

				text = prefix + "$" + idx + postfix;
				code += "var $" + (idx++) + " = ";
				if (op == "*")
					code += "NumJS.MUL";
				if (op == ".")
					code += "NumJS.DOT";
				if (op == "/")
					code += "NumJS.DIV";
				if (op == "\\")
					code += "NumJS.SOLVE";
				code += "(" + val1 + ", " + val2 + ");\n";
				continue;
			}

			// add and subtract
			pos = text.search(/([$a-zA-Z_0-9.]+)(\+|-)([$a-zA-Z_0-9.]+)/);
			if (pos >= 0)
			{
				var prefix = text.slice(0, pos);
				var val1 = RegExp.$1;
				pos += val1.length;
				var op = RegExp.$2
				pos += op.length;
				var val2 = RegExp.$3
				pos += val2.length;
				var postfix = text.slice(pos);

				val1 = reduce(val1);
				val2 = reduce(val2);

				text = prefix + "$" + idx + postfix;
				code += "var $" + (idx++) + " = ";
				if (op == "+")
					code += "NumJS.ADD";
				if (op == "-")
					code += "NumJS.SUB";
				code += "(" + val1 + ", " + val2 + ");\n";
				continue;
			}

			// handle imaginary literals
			pos = text.search(/^([0-9][0-9.]*)i$/);
			if (pos >= 0) {
				code += "var $" + idx + " = NumJS.C(0, " + RegExp.$1 + ");\n";
				return "$" + (idx++);
			}

			// we are done
			pos = text.search(/^[$a-zA-Z_0-9.]+$/);
			if (pos >= 0)
				return text;

			throw "NumJS.Parse parser error.";
		}
	}

	text = text.replace(new RegExp("[ \t\r\n]", "g"), "");

	var result = reduce(text);
	code += "return " + result + ";\n})";

	return code;
};

/*
 *  NumJS -- a JavaScript library for numerical computing
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


NumJS.Cmplx = function(re, im) {
	this.re = re;
	this.im = im;
};

NumJS.C = function(re, im) {
	return new NumJS.Cmplx(re, im);
};

NumJS.P = function(abs, arg) {
	var re = abs * Math.cos(arg);
	var im = abs * Math.sin(arg);
	return new NumJS.Cmplx(re, im);
};

NumJS.Cmplx.prototype =
{
	op_add: function(a, b) {
		if ((a instanceof NumJS.Cmplx) && (b instanceof NumJS.Cmplx))
			return NumJS.C(a.re + b.re, a.im + b.im);
		if ((a instanceof NumJS.Cmplx) && (typeof(b) == "number"))
			return NumJS.C(a.re + b, a.im);
		if ((typeof(a) == "number") && (b instanceof NumJS.Cmplx))
			return NumJS.C(a + b.re, b.im);
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_add) == "function"))
			return b.op_add(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_sub: function(a, b) {
		if ((a instanceof NumJS.Cmplx) && (b instanceof NumJS.Cmplx))
			return NumJS.C(a.re - b.re, a.im - b.im);
		if ((a instanceof NumJS.Cmplx) && (typeof(b) == "number"))
			return NumJS.C(a.re - b, a.im);
		if ((typeof(a) == "number") && (b instanceof NumJS.Cmplx))
			return NumJS.C(a - b.re, -b.im);
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_sub) == "function"))
			return b.op_sub(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_mul: function(a, b) {
		if ((a instanceof NumJS.Cmplx) && (b instanceof NumJS.Cmplx))
			return NumJS.C(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
		if ((a instanceof NumJS.Cmplx) && (typeof(b) == "number"))
			return NumJS.C(a.re * b, a.im * b);
		if ((typeof(a) == "number") && (b instanceof NumJS.Cmplx))
			return NumJS.C(a * b.re, a * b.im);
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_mul) == "function"))
			return b.op_mul(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_dot: function(a, b) {
		if ((a instanceof NumJS.Cmplx) && (b instanceof NumJS.Cmplx))
			return NumJS.C(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
		if ((a instanceof NumJS.Cmplx) && (typeof(b) == "number"))
			return NumJS.C(a.re * b, a.im * b);
		if ((typeof(a) == "number") && (b instanceof NumJS.Cmplx))
			return NumJS.C(a * b.re, a * b.im);
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_dot) == "function"))
			return b.op_dot(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_div: function(a, b) {
		if ((a instanceof NumJS.Cmplx) && (b instanceof NumJS.Cmplx))
			return NumJS.C((a.re*b.re + a.im*b.im) / (b.re*b.re + b.im*b.im),
					(a.im*b.re - a.re*b.im) / (b.re*b.re + b.im*b.im));
		if ((a instanceof NumJS.Cmplx) && (typeof(b) == "number"))
			return NumJS.C(a.re / b, a.im / b);
		if ((typeof(a) == "number") && (b instanceof NumJS.Cmplx))
			return NumJS.C((a*b.re) / (b.re*b.re + b.im*b.im), (-a*b.im) / (b.re*b.re + b.im*b.im));
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_div) == "function"))
			return b.op_div(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_solve: function(a, b) {
		var aIsScalar = typeof(a) == "number" || a instanceof NumJS.Cmplx;
		var bIsScalar = typeof(b) == "number" || b instanceof NumJS.Cmplx;
		if (aIsScalar && bIsScalar)
			return this.op_div(b, a);
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_solve) == "function"))
			return b.op_solve(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_pow: function(a, b) {
		var aIsScalar = typeof(a) == "number" || a instanceof NumJS.Cmplx;
		var bIsScalar = typeof(b) == "number" || b instanceof NumJS.Cmplx;
		if (aIsScalar && bIsScalar) {
			var aIsReal = NumJS.IM(a) == 0;
			var bIsReal = NumJS.IM(b) == 0;
			if (aIsReal && bIsReal)
				return Math.pow(NumJS.RE(a), NumJS.RE(b));
			if (aIsReal)
				return NumJS.EXP(NumJS.MUL(b, NumJS.LOG(a)));
			if (bIsReal)
				return NumJS.P(Math.pow(NumJS.ABS(a), NumJS.RE(b)), NumJS.ARG(a) * NumJS.RE(b));
		}
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_pow) == "function"))
			return b.op_pow(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_inv: function(a) {
		return NumJS.C(1 / (a.re*a.re + a.im*a.im), -a.im / (a.re*a.re + a.im*a.im));
	},
	op_neg: function(a) {
		return NumJS.C(-a.re, -a.im);
	},
	op_abs: function(a) {
		return Math.sqrt(a.re*a.re + a.im*a.im);
	},
	op_norm: function(a) {
		return Math.sqrt(a.re*a.re + a.im*a.im);
	},
	op_arg: function(a) {
		return Math.atan2(a.im, a.re);
	},
	op_conj: function(a) {
		return NumJS.C(a.re, -a.im);
	},
	op_transp: function(a) {
		return a;
	},
	op_exp: function(a) {
		var len = Math.exp(a.re);
		if (a.im == 0)
			return len;
		return NumJS.C(len * Math.cos(a.im), len * Math.sin(a.im));
	},
	op_log: function(a) {
		if (a.im == 0)
			return Math.log(a.re);
		throw "NumJS.Cmplx type error";
	},
	op_det: function(a) {
		return a;
	},
	op_re: function(a) {
		return a.re;
	},
	op_im: function(a) {
		return a.im;
	},
	op_round: function(a, n) {
		return NumJS.C(NumJS.ROUND(a.re, n), NumJS.ROUND(a.im, n));
	},
	op_eq: function(a, b) {
		if ((a instanceof NumJS.Cmplx) && (b instanceof NumJS.Cmplx))
			return a.re == b.re && a.im == b.im;
		if ((a instanceof NumJS.Cmplx) && (typeof(b) == "number"))
			return a.re == b && a.im == 0;
		if ((typeof(a) == "number") && (b instanceof NumJS.Cmplx))
			return a == b.re && 0 == b.im;
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_eq) == "function"))
			return b.op_eq(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_eq_abs: function(a, b, d) {
		var aIsScalar = typeof(a) == "number" || a instanceof NumJS.Cmplx;
		var bIsScalar = typeof(b) == "number" || b instanceof NumJS.Cmplx;
		if (aIsScalar && bIsScalar)
			return NumJS.ABS(NumJS.SUB(a, b)) <= d;
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_eq_abs) == "function"))
			return b.op_eq_abs(a, b);
		throw "NumJS.Cmplx type error";
	},
	op_eq_rel: function(a, b, d) {
		var aIsScalar = typeof(a) == "number" || a instanceof NumJS.Cmplx;
		var bIsScalar = typeof(b) == "number" || b instanceof NumJS.Cmplx;
		if (aIsScalar && bIsScalar)
			return NumJS.ABS(NumJS.SUB(a, b)) <= d * (NumJS.ABS(a) + NumJS.ABS(b)) * 0.5;
		if (!(b instanceof NumJS.Cmplx) && (typeof(b.op_eq_rel) == "function"))
			return b.op_eq_rel(a, b);
		throw "NumJS.Cmplx type error";
	},
	toString: function(n) {
		return "(" + this.re.toString(n) + (this.im >= 0 ? "+" : "") +
				this.im.toString(n) + "i)";
	},
	toFixed: function(n) {
		return "(" + this.re.toFixed(n) + (this.im >= 0 ? "+" : "") +
				this.im.toFixed(n) + "i)";
	},
	toPrecision: function(n) {
		return "(" + this.re.toPrecision(n) + (this.im >= 0 ? "+" : "") +
				this.im.toPrecision(n) + "i)";
	}
};

/*
 *  NumJS -- a JavaScript library for numerical computing
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


// Generic Matrix -- do not instanciate directly!

NumJS.GenericMatrix = function() {
};

NumJS.GenericMatrix.prototype =
{
	get: function(i, j) {
		throw "NumJS.Matrix called virtual function from NumJS.GenericMatrix";
	},
	set: function(i, j, v) {
		throw "NumJS.Matrix called virtual function from NumJS.GenericMatrix";
	},
	copy: function() {
		var result = new NumJS.Matrix(this.rows, this.cols);
		for (var i=0; i < this.rows; i++)
		for (var j=0; j < this.cols; j++)
			result.set(i, j, this.get(i, j));
		return result;
	},
	clone: function() {
		return this.copy();
	},
	cut: function(i, j, rows, cols) {
		i = +i, j = +j, rows = +rows, cols = +cols;
		var result = new Array();
		for (var y=0; y < rows; y++)
		for (var x=0; x < cols; x++)
			result.push(this.get(i+y, j+x));
		return result;
	},
	cut_cm: function(i, j, rows, cols) {
		i = +i, j = +j, rows = +rows, cols = +cols;
		var result = new Array();
		for (var x=0; x < cols; x++)
		for (var y=0; y < rows; y++)
			result.push(this.get(i+y, j+x));
		return result;
	},
	paste: function(i, j, rows, cols, data) {
		i = +i, j = +j, rows = +rows, cols = +cols;
		for (var y=0; y < rows; y++)
		for (var x=0; x < cols; x++)
			this.set(i+y, j+x, data.shift());
	},
	paste_cm: function(i, j, rows, cols, data) {
		i = +i, j = +j, rows = +rows, cols = +cols;
		for (var x=0; x < cols; x++)
		for (var y=0; y < rows; y++)
			this.set(i+y, j+x, data.shift());
	},
	op_add: function(a, b) {
		if ((a instanceof NumJS.GenericMatrix) && (b instanceof NumJS.GenericMatrix))
		{
			if (a.rows != b.rows || a.cols != b.cols)
				throw "NumJS.Matrix dimension mismatch";
			var result = new NumJS.Matrix(a.rows, a.cols);
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < a.cols; j++)
				result.set(i, j, NumJS.ADD(a.get(i, j), b.get(i, j)));
			return result;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_add) == "function"))
			return b.op_add(a, b);
		throw "NumJS.Matrix type error";
	},
	op_sub: function(a, b) {
		if ((a instanceof NumJS.GenericMatrix) && (b instanceof NumJS.GenericMatrix))
		{
			if (a.rows != b.rows || a.cols != b.cols)
				throw "NumJS.Matrix dimension mismatch";
			var result = new NumJS.Matrix(a.rows, a.cols);
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < a.cols; j++)
				result.set(i, j, NumJS.SUB(a.get(i, j), b.get(i, j)));
			return result;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_sub) == "function"))
			return b.op_sub(a, b);
		throw "NumJS.Matrix type error";
	},
	op_mul: function(a, b) {
		var aIsMatrix = a instanceof NumJS.GenericMatrix;
		var bIsMatrix = b instanceof NumJS.GenericMatrix;
		if (aIsMatrix && bIsMatrix)
		{
			if (a.cols != b.rows)
				throw "NumJS.Matrix dimension mismatch";
			var result = new NumJS.Matrix(a.rows, b.cols);
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < b.cols; j++)
			{
				var v = result.get(i, j);
				for (var k=0; k < a.cols; k++)
					v = NumJS.ADD(v, NumJS.MUL(a.get(i, k), b.get(k, j)));
				result.set(i, j, v);
			}
			return result;
		}
		var aIsScalar = typeof(a) == "number" || a instanceof NumJS.Cmplx;
		var bIsScalar = typeof(b) == "number" || b instanceof NumJS.Cmplx;
		if (aIsMatrix && bIsScalar)
		{
			var result = new NumJS.Matrix(a.rows, a.cols);
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < a.cols; j++)
				result.set(i, j, NumJS.MUL(a.get(i, j), b));
			return result;
		}
		if (aIsScalar && bIsMatrix)
		{
			var result = new NumJS.Matrix(b.rows, b.cols);
			for (var i=0; i < b.rows; i++)
			for (var j=0; j < b.cols; j++)
				result.set(i, j, NumJS.MUL(b.get(i, j), a));
			return result;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_mul) == "function"))
			return b.op_mul(a, b);
		throw "NumJS.Matrix type error";
	},
	op_dot: function(a, b) {
		if ((a instanceof NumJS.GenericMatrix) && (b instanceof NumJS.GenericMatrix))
		{
			var result = 0;
			if (a.cols != b.cols || a.rows != b.rows)
				throw "NumJS.Matrix dimension mismatch";
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < b.cols; j++)
				result = NumJS.ADD(result, NumJS.MUL(a.get(i, j), b.get(i, j)));
			return result;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_dot) == "function"))
			return b.op_dot(a, b);
		throw "NumJS.Matrix type error";
	},
	op_div: function(a, b) {
		var aIsMatrix = a instanceof NumJS.GenericMatrix;
		var bIsScalar = typeof(b) == "number" || b instanceof NumJS.Cmplx;
		if (aIsMatrix && bIsScalar)
		{
			var result = new NumJS.Matrix(a.rows, a.cols);
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < a.cols; j++)
				result.set(i, j, NumJS.DIV(a.get(i, j), b));
			return result;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_div) == "function"))
			return b.op_div(a, b);
		throw "NumJS.Matrix type error";
	},
	op_solve: function(a, b) {
		var aIsMatrix = a instanceof NumJS.GenericMatrix;
		var bIsMatrix = b instanceof NumJS.GenericMatrix;
		if (aIsMatrix && bIsMatrix) {
			var PLU = a.PLU();
			if (PLU == null)
				return null;
			return PLU.solve(b);
		}
		var aIsScalar = typeof(a) == "number" || a instanceof NumJS.Cmplx;
		if (aIsScalar && bIsMatrix)
			return this.op_div(b, a);
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_solve) == "function"))
			return b.op_solve(a, b);
		throw "NumJS.Matrix type error";
	},
	op_pow: function(a, b) {
		var aIsMatrix = a instanceof NumJS.GenericMatrix;
		var bIsScalar = typeof(b) == "number" || b instanceof NumJS.Cmplx;
		if (aIsMatrix && bIsScalar)
		{
			if (a.cols != a.rows)
				throw "NumJS.Matrix dimension mismatch";
			if (NumJS.IM(b) != 0 || NumJS.RE(b) != Math.round(NumJS.RE(b)) || NumJS.RE(b) < 0)
				throw "NumJS.Matrix negative, fractional or complex matrix power";
			var result = new NumJS.Matrix(a.rows, a.cols);
			for (var i=0; i < a.rows; i++)
				result.set(i, i, 1);
			for (var i=0; i < NumJS.RE(b); i++)
				result = NumJS.MUL(result, a);
			return result;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_div) == "function"))
			return b.op_div(a, b);
		throw "NumJS.Matrix type error";
	},
	op_neg: function(a) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, NumJS.NEG(a.get(j, i)));
		return result;
	},
	op_abs: function(a) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, NumJS.ABS(a.get(j, i)));
		return result;
	},
	op_norm: function(a) {
		var norm2 = 0;
		if (a.cols != 1 && a.rows != 1)
			throw "NumJS.Matrix dimension mismatch";
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++) {
			var v = a.get(j, i);
			if (NumJS.IM(v) != 0)
				throw "NumJS.Matrix type error";
			v = NumJS.RE(v);
			norm2 += v*v;
		}
		return Math.sqrt(norm2);
	},
	op_arg: function(a) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, NumJS.ARG(a.get(j, i)));
		return result;
	},
	op_conj: function(a) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, NumJS.CONJ(a.get(j, i)));
		return result;
	},
	op_transp: function(a) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, a.get(j, i));
		return result;
	},
	op_det: function(a) {
		var PLU = a.PLU();
		if (PLU == null)
			return 0;
		return PLU.det();
	},
	op_re: function(a) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, NumJS.RE(a.get(j, i)));
		return result;
	},
	op_im: function(a) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, NumJS.IM(a.get(j, i)));
		return result;
	},
	op_round: function(a, n) {
		var result = new NumJS.Matrix(a.rows, a.cols);
		for (var i=0; i < a.rows; i++)
		for (var j=0; j < a.cols; j++)
			result.set(i, j, NumJS.ROUND(a.get(j, i), +n));
		return result;
	},
	op_eq: function(a, b) {
		if ((a instanceof NumJS.GenericMatrix) && (b instanceof NumJS.GenericMatrix))
		{
			if (a.cols != b.cols || a.rows != b.rows)
				throw "NumJS.Matrix dimension mismatch";
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < b.cols; j++)
				if (!NumJS.EQ(a.get(i, j), b.get(i, j)))
					return false;
			return true;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_eq) == "function"))
			return b.op_eq(a, b);
		throw "NumJS.Matrix type error";
	},
	op_eq_abs: function(a, b, d) {
		if ((a instanceof NumJS.GenericMatrix) && (b instanceof NumJS.GenericMatrix))
		{
			if (a.cols != b.cols || a.rows != b.rows)
				throw "NumJS.Matrix dimension mismatch";
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < b.cols; j++)
				if (!NumJS.EQ_ABS(a.get(i, j), b.get(i, j), +d))
					return false;
			return true;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_eq_abs) == "function"))
			return b.op_eq_abs(a, b, +d);
		throw "NumJS.Matrix type error";
	},
	op_eq_rel: function(a, b, d) {
		if ((a instanceof NumJS.GenericMatrix) && (b instanceof NumJS.GenericMatrix))
		{
			if (a.cols != b.cols || a.rows != b.rows)
				throw "NumJS.Matrix dimension mismatch";
			for (var i=0; i < a.rows; i++)
			for (var j=0; j < b.cols; j++)
				if (!NumJS.EQ_REL(a.get(i, j), b.get(i, j), +d))
					return false;
			return true;
		}
		if (!(b instanceof NumJS.GenericMatrix) && (typeof(b.op_eq_rel) == "function"))
			return b.op_eq_rel(a, b, +d);
		throw "NumJS.Matrix type error";
	},
	toStringWorker: function(f) {
		var str = "[ ";
		for (var i=0; i < this.rows; i++)
		{
			for (var j=0; j < this.cols; j++)
				str += (j > 0 ? ", " : "") + f(this.get(i, j));
			if (i < this.rows-1)
				str += "; "
		}
		str += " ]";
		return str;
	},
	toString: function(n) {
		return this.toStringWorker(function(v){
			return v.toString(n);
		});
	},
	toFixed: function(n) {
		return this.toStringWorker(function(v){
			return v.toFixed(n);
		});
	},
	toPrecision: function(n) {
		return this.toStringWorker(function(v){
			return v.toPrecision(n);
		});
	}
};

// Just a normal Matrix

NumJS.Matrix = function(rows, cols, initdata) {
	rows = +rows, cols = +cols;
	this.rows = rows;
	this.cols = cols;
	this.data = Array();
	this.cache = new Object();
	for (var i=0; i < this.rows; i++)
	for (var j=0; j < this.cols; j++)
		this.data[i*this.cols + j] = 0;
	if (initdata instanceof Array) {
		if (initdata.length != this.rows*this.cols)
			throw "NumJS.Matrix initdata dimension mismatch";
		for (var i = 0; i < initdata.length; i++)
			this.data[i] = initdata[i];
	}
};

NumJS.MAT = function(rows, cols, initdata) {
	return new NumJS.Matrix(rows, cols, initdata);
};

NumJS.Matrix.prototype = new NumJS.GenericMatrix();

NumJS.Matrix.prototype.get = function(i, j) {
	i = +i, j = +j;
	var idx = i*this.cols + j;
	return this.data[idx];
};

NumJS.Matrix.prototype.set = function(i, j, v) {
	i = +i, j = +j;
	var idx = i*this.cols + j;
	this.data[idx] = v;
	this.cache = new Object();
};

// Permutation Matrix

NumJS.PMatrix = function(dim, initdata) {
	dim = +dim;
	this.sign = 1;
	this.rows = dim;
	this.cols = dim;
	this.data = Array();
	this.cache = new Object();
	for (var i=0; i < dim; i++)
		this.data[i] = i;
	if (initdata instanceof Array)
	{
		var tmp = new Array();
		if (initdata.length != dim)
			throw "NumJS.Matrix initdata dimension mismatch";
		for (var i = 0; i < initdata.length; i++)
			tmp[i] = this.data[i] = initdata[i];

		// detecting sign: shakersort and count number of exchanges
		var sort_done = 0;
		while (!sort_done) {
			sort_done = 1;
			for (var i = 1; i < dim; i++)
				if (tmp[i-1] > tmp[i]) {
					sort_done = 0;
					this.sign *= -1;
					var t = tmp[i-1];
					tmp[i-1] = tmp[i];
					tmp[i] = t;
				}
			for (var i = dim-1; i > 0; i--)
				if (tmp[i-1] > tmp[i]) {
					sort_done = 0;
					this.sign *= -1;
					var t = tmp[i-1];
					tmp[i-1] = tmp[i];
					tmp[i] = t;
				}
		}

		for (var i = 0; i < dim; i++)
			if (tmp[i] != i)
				throw "NumJS.Matrix initdata not a permutation";
	}
};

NumJS.PMAT = function(dim, initdata) {
	return new NumJS.PMatrix(dim, initdata);
};

NumJS.PMatrix.prototype = new NumJS.GenericMatrix();

NumJS.PMatrix.prototype.get = function(i, j) {
	i = +i, j = +j;
	if (this.data[j] == i)
		return 1;
	return 0;
};

// The PMatrix .data array stores the row number of the '1' in each column.
// therefore column pivoting is trivial and row pivoting must be performed indirectly

NumJS.PMatrix.prototype.pivot_col = function(i, j) {
	i = +i, j = +j;
	if (i != j) {
		var tmp = this.data[i];
		this.data[i] = this.data[j];
		this.data[j] = tmp;
		this.sign *= -1;
		this.cache = new Object();
	}
};

NumJS.PMatrix.prototype.pivot_row = function(i, j) {
	i = +i, j = +j;
	if (i != j) {
		i = this.data.indexOf(i);
		j = this.data.indexOf(j);
		this.pivot_col(i, j);
	}
};

NumJS.PMatrix.prototype.clone = function() {
	var result = new NumJS.PMatrix(this.rows);
	for (var i = 0; i < this.rows; i++)
		result.data[i] = this.data[i];
	result.sign = this.sign;
	return result;
};

NumJS.PMatrix.prototype.op_transp = function(a) {
	if (a instanceof NumJS.PMatrix)
	{
		var result = new NumJS.PMatrix(a.rows);
		for (var i = 0; i < a.rows; i++) {
			var j = a.data[i];
			result.data[j] = i;
		}
		result.sign = a.sign;
		return result;
	}
	throw "NumJS.Matrix type error";
};

NumJS.PMatrix.prototype.op_mul = function(a, b) {
	if ((a instanceof NumJS.PMatrix) && (b instanceof NumJS.PMatrix))
	{
		if (a.cols != b.rows)
			throw "NumJS.Matrix dimension mismatch";
		var result = new NumJS.PMatrix(a.rows);
		for (var i = 0; i < a.rows; i++)
			result.data[i] = a.data[b.data[i]];
		result.sign = a.sign * b.sign;
		return result;
	}
	// Fallback to normal matrix-matrix multiplikation
	return NumJS.GenericMatrix.prototype.op_mul(a, b);
};

NumJS.PMatrix.prototype.op_det = function(a) {
	return a.sign;
};

NumJS.PMatrix.prototype.op_round = function(a, n) {
	if (+n >= 0)
		return a.clone();
	return NumJS.GenericMatrix.prototype.op_round(a, n);
};

NumJS.PMatrix.prototype.op_conj = NumJS.PMatrix.prototype.op_transp;
NumJS.PMatrix.prototype.op_inv = NumJS.PMatrix.prototype.op_transp;

/*
 *  NumJS -- a JavaScript library for numerical computing
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


NumJS.GenericMatrix.prototype.rref = function()
{
	if ("rref" in this.cache)
		return this.cache["rref"];

	var R = this.copy();
	var pivcols = new Array();

	var rank = 0;
	for (var j = 0; j < R.cols && rank < R.rows; j++)
	{
		var i = rank;
		var i_abs = NumJS.ABS(R.get(i, j));
		for (var k = rank+1; k < R.rows; k++) {
			var k_abs = NumJS.ABS(R.get(k, j));
			if (k_abs < i_abs)
				i = k, i_abs = k_abs;
		}

		if (i_abs < NumJS.eps)
			continue;

		var pivot = R.get(i, j);
		for (var l = j; l < R.cols; l++) {
			var val = l != j ? NumJS.DIV(R.get(i, l), pivot) : 1;
			if (i != rank)
				R.set(i, l, R.get(rank, l));
			R.set(rank, l, val);
		}

		for (i = 0; i < R.rows; i++)
		{
			if (i == rank)
				continue;
			var factor = R.get(i, j);
			for (var l = j+1; l < R.cols; l++)
				R.set(i, l, NumJS.SUB(R.get(i, l), NumJS.MUL(R.get(rank, l), factor)));
			R.set(i, j, 0);
		}

		pivcols.push(j);
		rank++;
	}

	R.pivcols = pivcols;
	this.cache["rref"] = R;
	return this.cache["rref"];
};

NumJS.GenericMatrix.prototype.rank = function()
{
	return this.rref().pivcols.length;
};

// invert generic matrices using rref()
NumJS.GenericMatrix.prototype.op_inv = function(a)
{
	if (a.rows != a.cols)
		throw "NumJS.RREF dimension mismatch";

	var work = NumJS.MAT(a.rows, 2*a.rows);
	work.paste(0, 0, a.rows, a.rows, a.cut(0, 0, a.rows, a.rows));
	for (var i = 0; i < a.rows; i++)
		work.set(i, a.rows+i, 1);

	work = work.rref();

	if (work.pivcols[a.rows-1] != a.rows-1)
		return null;

	var result = NumJS.MAT(a.rows, a.rows);
	result.paste(0, 0, a.rows, a.rows, work.cut(0, a.rows, a.rows, a.rows));
	return result;
};

/*
 *  NumJS -- a JavaScript library for numerical computing
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


NumJS.GenericPLU = function(P, L, U)
{
	this.P = P;
	this.L = L;
	this.U = U;
};

NumJS.GenericMatrix.prototype.LU = function()
{
	if (this.rows != this.cols)
		throw "NumJS.MatLU dimension mismatch";

	if ("LU" in this.cache)
		return this.cache["LU"];

	var A, P, L, U;
	var n = this.rows;

	// in this LU solver P is just the identity
	P = NumJS.PMAT(n);
	L = NumJS.MAT(n, n);
	A = this.copy();
	U = this.copy();

	for (var k = 0; k < n; k++)
	{
		var pivot = U.get(k, k);
		if (NumJS.ABS(pivot) < NumJS.eps)
			return null;

		L.set(k, k, 1);
		for (var i = k+1; i < n; i++) {
			L.set(i, k, NumJS.DIV(U.get(i, k), pivot));
			for (var j = k+1; j < n; j++)
				U.set(i, j, NumJS.SUB(U.get(i, j), NumJS.MUL(L.get(i, k), U.get(k, j))));
			U.set(i, k, 0);
		}
	}

	this.cache["LU"] = new NumJS.GenericPLU(P, L, U);
	return this.cache["LU"];
};

NumJS.GenericMatrix.prototype.PLU = function()
{
	if (this.rows != this.cols)
		throw "NumJS.MatLU dimension mismatch";

	if ("PLU" in this.cache)
		return this.cache["PLU"];

	var A, P, L, U;
	var n = this.rows;

	P = NumJS.PMAT(n);
	L = NumJS.MAT(n, n);
	A = this.copy();
	U = this.copy();

	for (var k = 0; k < n; k++)
	{
		// find the correct row
		var pivot = U.get(k, k);
		var pivot_i = k;
		var pivot_abs = NumJS.ABS(pivot);
		for (var i = k+1; i < n; i++) {
			var pv = U.get(i, k)
			var pa = NumJS.ABS(pv);
			if (pa > pivot_abs) {
				pivot = pv;
				pivot_i = i;
				pivot_abs = pa;
			}
		}

		if (pivot_i != k) {
			// perform row pivoting in L
			for (var j = 0; j < k; j++) {
				var t1 = L.get(k, j);
				var t2 = L.get(pivot_i, j);
				L.set(k, j, t2);
				L.set(pivot_i, j, t1);
			}
			// perform row pivoting in U
			for (var j = k; j < n; j++) {
				var t1 = U.get(k, j);
				var t2 = U.get(pivot_i, j);
				U.set(k, j, t2);
				U.set(pivot_i, j, t1);
			}
			P.pivot_row(pivot_i, k);
		}

		if (NumJS.ABS(pivot) < NumJS.eps)
			return null;

		L.set(k, k, 1);
		for (var i = k+1; i < n; i++) {
			L.set(i, k, NumJS.DIV(U.get(i, k), pivot));
			for (var j = k+1; j < n; j++)
				U.set(i, j, NumJS.SUB(U.get(i, j), NumJS.MUL(L.get(i, k), U.get(k, j))));
			U.set(i, k, 0);
		}
	}

	this.cache["PLU"] = new NumJS.GenericPLU(P, L, U);
	return this.cache["PLU"];
};

NumJS.GenericPLU.prototype.det = function()
{
	var result = this.P.sign;
	for (var k = 0; k < this.U.rows; k++)
		result = NumJS.MUL(result, this.U.get(k, k));
	return result;
}

NumJS.GenericPLU.prototype.solve = function(Y)
{
	if (!(Y instanceof NumJS.GenericMatrix))
		throw "NumJS.MatLU type error";

	var n = this.P.rows;
	if (n != Y.rows)
		throw "NumJS.MatLU dimension mismatch";

	var X = NumJS.MAT(n, Y.cols);

	for (var k = 0; k < Y.cols; k++)
	{
		// LU*x = y  =>  L*c = y  and  U*x = c
		var y = new Array();
		var c = new Array();
		var x = new Array();

		// Initialize y, c and x
		for (var i = 0; i < n; i++) {
			y[this.P.data[i]] = Y.get(i, k);
			c[i] = x[i] = 0;
		}

		// Solve L*c = y
		for (var i = 0; i < n; i++) {
			var pivot = this.L.get(i, i);
			c[i] = NumJS.DIV(y[i], pivot);
			for (var j = i+1; j < n; j++)
				y[j] = NumJS.SUB(y[j], NumJS.MUL(c[i], this.L.get(j, i)));
		}

		// Solve U*x = c
		for (var i = n-1; i >= 0; i--) {
			var pivot = this.U.get(i, i);
			x[i] = NumJS.DIV(c[i], pivot);
			for (var j = i-1; j >= 0; j--)
				c[j] = NumJS.SUB(c[j], NumJS.MUL(x[i], this.U.get(j, i)));
		}

		// Store x in X
		for (var i = 0; i < n; i++)
			X.set(i, k, x[i]);
	}

	return X;
}

