#!/usr/bin/env python3
#
# MathGPT -- A math engine based on GPT
#
# Copyright (C) 2023 Claire Xenia Wolf <claire@yosyshq.com>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

import textwrap, hashlib, click, json, re
from types import SimpleNamespace
from click import echo, style
from pprint import pp

config = SimpleNamespace()
config.max_queries = -1
config.short_queries = False
config.print_cache_info = False
config.tuned_model = "davinci:ft-personal-2023-03-11-12-12-19"
#config.tuned_model = "davinci:ft-personal-2023-03-16-15-57-17"
config.model_temperature = 0.0
config.max_tokens = 300

def gpt(*args, **kwargs):
    if config.max_queries >= 0:
        assert config.max_queries
        config.max_queries -= 1

    cacheDir = os.path.expanduser("~/.MathGPT/cache")
    os.makedirs(cacheDir, exist_ok=True)

    key_string = json.dumps((args, kwargs), sort_keys=True)
    key_hash = hashlib.blake2b(key_string.encode()).hexdigest()

    cacheFile = f"{cacheDir}/{key_hash}.json"
    if not os.access(cacheFile, os.R_OK):
        if config.print_cache_info:
            echo(f"<cold:{key_hash}>", nl=False)
        response = openai.Completion.create(*args, **kwargs)
        with open(cacheFile, "w") as f:
            json.dump({"args": args, "kwargs": kwargs, "response": response.to_dict_recursive()},
                    f, sort_keys=True, indent=4)
            print(file=f)
    else:
        if config.print_cache_info:
            echo(f"<hot:{key_hash}>", nl=False)

    with open(cacheFile) as f:
        data = json.load(f)

    # pp(data)
    return SimpleNamespace(**data["response"]["choices"][0])


def evalExprStep(expr):
    def getNumber(token):
        if type(token) is float:
            return token
        if re.match(r"^\.?[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?$", token):
            return float(token)
        return None

    def getOperator(ops):
        def f(token):
            if type(token) is str and token in ops:
                return token
            return None
        return f

    def update(pattern, func):
        cursor = 0
        newExpr = []
        foundMatch = False
        while cursor+len(pattern) <= len(expr):
            values = [pattern[i](expr[cursor+i]) for i in range(len(pattern))]
            if None in values:
                newExpr.append(expr[cursor])
                cursor += 1
            else:
                newExpr.append(func(*values))
                cursor += len(pattern)
                foundMatch = True
        while foundMatch and cursor < len(expr):
            newExpr.append(expr[cursor])
            cursor += 1
        if foundMatch:
            return newExpr
        return None

    if newExpr := update([getNumber, getOperator("*/"), getNumber], lambda a,b,c: a*c if b == "*" else a/c):
        expr = newExpr

    elif newExpr := update([getNumber, getOperator("+-"), getNumber], lambda a,b,c: a+c if b == "+" else a-c):
        expr = newExpr

    while newExpr := update([getOperator("("), getNumber, getOperator(")")], lambda a,b,c: b):
        expr = newExpr

    return expr


class MathGPT:
    def __init__(self, text="", verbose=True):
        self.verbose = verbose
        self.text = ""
        self.state = dict()
        self.nextState = dict()
        self.varNames = list()
        self.appendText(text)

    def getTrainingJSONL(self):
        data = list()
        txt = self.text
        cursor = 0

        # find end of problem statement
        while not txt[:cursor].endswith("## Analysis\n") and cursor < len(txt): cursor += 1

        if not config.short_queries:
            p, c = txt[:cursor], txt[cursor:]
            return f'{{ "prompt": {json.dumps(p)}, "completion": {json.dumps(c)} }}'

        while cursor < len(txt):
            clen = 0
            parcnt = 0

            while cursor+clen < len(txt):
                clen += 1
                if txt[cursor+clen-1] == "[":
                    parcnt += 1
                elif txt[cursor+clen-1] == "]":
                    parcnt -= 1
                    if parcnt == 0: break

            data.append((txt[:cursor], txt[cursor:cursor+clen]))

            cursor += clen
            while cursor < len(txt) and txt[cursor-1] != "\n": cursor += 1

        return "\n".join([f'{{ "prompt": {json.dumps(p)}, "completion": {json.dumps(c)} }}' for p, c in data])

    def appendText(self, text):
        parts = re.split(r"(?<= \])(?: [^\n]+)?(?:\n|$)", text)

        for part in parts:
            if self.verbose:
                echo(styleMathGPT(part), nl=False)
            self.text += part
            if (output := self.evalCommand()) is not None:
                if self.verbose:
                    echo(styleMathGPT(output), nl=False)
                self.text += output

    def evalCommand(self):
        command = self.text.rsplit("\n", 1)[-1]
        if command.startswith("[") and command.endswith("]"):
            command = command[1:-1].strip()
            text = ""

            if command == "next.":
                for k, v in self.nextState.items():
                    self.state[k] = v
                self.nextState = dict()
                for k in self.varNames:
                    if text != "": text += ","
                    if type(self.state[k]) is float:
                        text += f" {k}={self.state[k]:g}"
                    else:
                        text += f" {k}={self.state[k]}"

            elif command == "resetall.":
                self.state = dict()
                self.nextState = dict()
                self.varNames = list()

            elif m := re.match(r"(\w+)('?) *:= *(.*)", command):
                newVar, tick, expr = m[1], m[2], m[3]
                expr = expr.replace(" ", "")
                expr = re.findall(r"(?:\.?[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?|[a-zA-Z_]+|.)", expr)
                expr = [self.state.get(t, t) for t in expr if t is not None and t != ""]
                text = f" {''.join([f'{v:g}' if type(v) is float else v for v in expr]).replace(' ','')}"
                while expr != (newExpr := evalExprStep(expr)):
                    expr = newExpr
                    text += f"={''.join([f'{v:g}' if type(v) is float else v for v in expr]).replace(' ','')}"
                assert len(expr) == 1
                if tick == "'":
                    if newVar not in self.state and newVar not in self.nextState:
                        self.varNames.append(newVar)
                    self.nextState[newVar] = expr[0]
                else:
                    assert newVar not in self.state and newVar not in self.nextState
                    self.varNames.append(newVar)
                    self.state[newVar] = expr[0]

            elif m := re.match(r"(.*)\.$", command):
                expr = m[1].replace(" ", "")
                expr = re.findall(r"(?:\.?[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?|[a-zA-Z_]+|.)", expr)
                expr = [self.state.get(t, t) for t in expr if t is not None and t != ""]
                text = f" {''.join([f'{v:g}' if type(v) is float else v for v in expr]).replace(' ','')}"
                while expr != (newExpr := evalExprStep(expr)):
                    expr = newExpr
                    text += f"={''.join([f'{v:g}' if type(v) is float else v for v in expr]).replace(' ','')}"

            else:
                text = " ???"
            return text + "\n"
        return None

    def run(self):
        while not self.text.endswith(" resetall. ]\n"):
            completion = gpt(model=config.tuned_model, prompt=self.text,
                    temperature=config.model_temperature, max_tokens=config.max_tokens,
                    stop=(" ]\n", " ] ") if config.short_queries else ("[ resetall. ]",))
            if completion.finish_reason == "length":
                self.appendText(completion.text)
            else:
                if config.short_queries:
                    self.appendText(completion.text + " ]")
                else:
                    self.appendText(completion.text + "[ resetall. ]")

    def __repr__(self):
        return styleMathGPT(f"--MathGPT-BEGIN--\n{self.text}\n--MathGPT-END--")


def styleMathGPT(text):
    lines = list()
    for line in text.split("\n"):

        if line.startswith("#"):
            lines.append(style(line, fg="magenta", bold=True, underline=True))
            continue

        if m := re.match(r"^(-|\(\d+\))(.*)", line):
            lines.append(style(m[1], fg="yellow", bold=True) + style(m[2], fg="blue"))
            continue

        if m := re.match(r"^(\[.*\])(.*)", line):
            lines.append(style(m[1], fg="green", bold=True) + style(m[2], fg="bright_black"))
            continue

        if line.startswith("--MathGPT-"):
            lines.append(style(line, fg="red"))
            continue

        lines.append(line)

    return "\n".join(lines)


def readMathGptFile(filename, verbose=False):
    instances, text = list(), list()
    if verbose:
        echo()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "----":
                text = "\n".join(text).strip()
                if text != "":
                    instances.append(MathGPT(text, verbose))
                if verbose:
                    echo()
                    echo("----")
                    echo()
                text = list()
            else:
                text.append(line)
    text = "\n".join(text).strip()
    if text != "":
        instances.append(MathGPT(text, verbose))
    if verbose:
        echo()
    return instances


@click.group()
def cli():
    """MathGPT command line interface

    Example:
    python3 mathgpt.py query "Alice has ten apples. Bob has seven apples. Alice trades half of her apples with Bob. Bob later eats a third of his apples. How many apples has Bob left?"
    """


@cli.command()
@click.argument('outfile')
@click.argument('files', nargs=-1)
def tunedata(outfile, files):
    """Generate fine-tune JSONL data file

    This command reads all fine-tune/training files and produces
    a fine-tune JSONL data file. (Note that each .mathgpt input
    file is turned into multiple prompt-completion pars, bc we
    do not fine-tune the model to generate the math code output
    text. So we produce a completion that stops right beore the
    math code output, and the we create a prompt that includes
    the entire math code output, and continue completing after
    the math code output.
    """

    with open(outfile, "w") as f:
        for filename in files:
            for instance in readMathGptFile(filename):
                print(instance.getTrainingJSONL(), file=f)


@cli.command()
@click.argument('prompt', nargs=-1)
def query(prompt):
    """Make a MathGPT query"""

    echo()
    prompt = textwrap.fill(" ".join(prompt).strip())

    txt = f"""
A list of math problems and puzzle questions, and one-line summaries that can
be used as problem titles. Note that such summaries must never give away the
solution of the problem or puzzle! One-line summaries should use title case and
should not end with a period.

----

Problem Statement:
Alice buys paint for 100 EUR. She paints the walls on her 230 cm tall 4 meter
times 5 meter room, and gives the leftover paint to Bob. It is exactly enough
paint for Bob that to paint a single 4 meter wide wall in his 255 cm room.
Assuming the same amount of paint per square meter wall in both rooms, how much
money does Bob owe to Alice, if he wants to pay her for the exact share of
the paint he was using?

One-Line Summary:
Calculating Paint Costs for Two Different Rooms

----

Problem Statement:
Alice goes to the market. She buys 5 liter milk for 0.7 EUR / liter and 2 kg
bread for 3 EUR / kg. Half the milk and a third of the bread are for Bob. How
much does Bob owe Alice?

One-Line Summary:
Buying Groceries

----

Problem Statement:
{prompt}

One-Line Summary:
"""

    title = gpt(model="text-davinci-003", prompt=txt, temperature=0.3, max_tokens=100, stop="\n").text
    instance = MathGPT(f"""# {title}\n\n## Problem Statement\n{prompt}\n\n## Analysis\n""")
    instance.run()
    echo()


@cli.command()
@click.argument('filename')
def complete(filename):
    """Complete a MathGPT query from an arbitrary point"""

    with open(filename) as f:
        instance = MathGPT(f.read())
    instance.run()
    echo()


@cli.command()
@click.argument('filename')
def check(filename):
    """Re-run all code in a MathGPT file"""
    readMathGptFile(filename, True)


if __name__ == "__main__":
    cli()
