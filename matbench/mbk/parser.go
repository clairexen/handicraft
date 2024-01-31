
package mbk
import "unicode"
import "regexp"

var re_num = regexp.MustCompile("^[+\\-]?[0-9]+(/[0-9]+|\\.[0-9]*)?")

func Parser(input string, pipe_delim int) (a Atom, r string) {
	var idx, end, ch int;

	next := func() {
		idx++;
		for idx < end && (unicode.IsSpace(int(input[idx])) || input[idx] == ';') {
			if input[idx] == ';' {
				for idx < end && input[idx] != '\n' { idx++; }
			}
			idx++;
		}
		ch = int(input[idx]);
	};

	reindex := func() {
		idx, end = 0, len(input);
		for idx < end && (unicode.IsSpace(int(input[idx])) || input[idx] == ';') {
			if input[idx] == ';' {
				for idx < end && input[idx] != '\n' { idx++; }
			}
			idx++;
		}
		ch = int(input[idx]);
	};

	reindex();

	if ch == '(' || ch == '[' || ch == '|' {
		delim := ch;
		dot_mode := false;
		tail_idx_inc := 1;
		next();
		if delim == '|' {
			delim = pipe_delim;
			tail_idx_inc = 0;
		}
		var el = make([]Atom, 0, 100);
		for (delim == '(' && ch != ')') || (delim == '[' && ch != ']') {
			if ch == '.' {
				dot_mode = true;
				next();
				continue;
			}
			c, i := Parser(input[idx:end], delim);
			n := len(el);
			el = el[0:n+1];
			el[n] = c;
			input = i;
			reindex();
		}
		if dot_mode {
			a = NewACons(el[0], el[1]);
		} else {
			a = NewANil();
			for i := len(el)-1; i >= 0; i-- {
				a = NewACons(el[i], a);
			}
		}
		return a, input[idx+tail_idx_inc:end];
	}

	if ch == '{' {
		next();
		// FIXME!
		return a, input[idx:end];
	}

	if ch == '\'' {
		next();
		c, i := Parser(input[idx:end], pipe_delim);
		a = NewACons(NewAId("quote"), NewACons(c, NewANil()));
		return a, i;
	}

	if ch == '#' {
		next();
		for ch != '(' && ch != '[' && ch != '|' {
			next();
		}
		c, i := Parser(input[idx:end], pipe_delim);
		q := NewACons(NewAId("quote"), NewACons(c, NewANil()));
		a = NewACons(NewAId("function"), NewACons(q, NewANil()));
		return a, i;
	}

	if match := re_num.ExecuteString(input[idx:end]); len(match) > 0 {
		r = input[idx:end];
		a = NewANumber(r[match[0]:match[1]]);
		r = r[match[1]:len(r)];
		return a, r;
	}

	toklen := 0;
	for idx+toklen < end && !unicode.IsSpace(int(input[idx+toklen])) {
		if input[idx+toklen] == '(' || input[idx+toklen] == ')' { break; }
		if input[idx+toklen] == '[' || input[idx+toklen] == ']' { break; }
		if input[idx+toklen] == '|' || input[idx+toklen] == '#' { break; }
		if input[idx+toklen] == ';' || input[idx+toklen] == '\'' { break; }
		toklen++;
	}

	if toklen > 0 {
		token := input[idx:idx+toklen];
		switch token {
		case "TRUE":
			a = Atom(NewABool(true));
		case "FALSE":
			a = Atom(NewABool(false));
		case "NIL":
			a = Atom(NewANil());
		default:
			a = Atom(NewAId(token));
		}
		return a, input[idx+toklen:end];
	}

	return nil, input;
}

