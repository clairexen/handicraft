#!/usr/bin/env python3

import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")
tuned_model = "davinci:ft-personal-2023-03-11-12-12-19"

import textwrap, hashlib, click, json, re
from click import echo, style
from pprint import pp


def complete(*args, **kwargs):
    cacheDir = os.path.expanduser("~/.MathGPT/cache")
    os.makedirs(cacheDir, exist_ok=True)

    key_string = json.dumps((args, kwargs), sort_keys=True)
    key_hash = hashlib.blake2b(key_string.encode()).hexdigest()

    cacheFile = f"{cacheDir}/{key_hash}.json"
    if not os.access(cacheFile, os.R_OK):
        response = openai.Completion.create(*args, **kwargs)
        with open(cacheFile, "w") as f:
            json.dump([args, kwargs, response.to_dict_recursive()], f, sort_keys=True)

    with open(cacheFile) as f:
        data = json.load(f)

    # pp(data)
    return data[2]["choices"][0]["text"]


class MathGPT:
    def __init__(self, text):
        self.text = text

    def getTrainingJSONL(self):
        data = list()
        txt = self.text
        cursor = 0

        # find end of problem statement
        while not txt[:cursor].endswith("## Analysis\n") and cursor < len(txt): cursor += 1

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

    def __repr__(self):
        return styleMathGPT(f"--MathGPT-BEGIN--\n{lines}\n--MathGPT-END--")


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


def readMathGptFile(filename):
    instances, text = list(), list()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "----":
                text = "\n".join(text).strip()
                if text != "":
                    instances.append(MathGPT(text))
                text = list()
            else:
                text.append(line)
        text = "\n".join(text).strip()
        if text != "":
            instances.append(MathGPT(text))
    return instances


@click.group()
def cli(): pass


@cli.command()
@click.argument('files', nargs=-1)
def tunedata(files):
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

    for filename in files:
        for instance in readMathGptFile(filename):
            print(instance.getTrainingJSONL())


@cli.command()
@click.argument('prompt', nargs=-1)
def query(prompt):
    """Make a MathGPT query"""

    echo()
    prompt = textwrap.fill(" ".join(prompt).strip())

    txt = f"""
A list of math problems and puzzle questions, and one-line summaries that can
be used as problem titles. Note that such summaries must not give away the
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
{prompt}

One-Line Summary:
"""

    title = complete(model="text-davinci-003", prompt=txt, temperature=0, max_tokens=20, stop="\n")

    txt = f"""# {title}\n\n## Problem Statement\n{prompt}\n\n## Analysis\n"""
    echo(styleMathGPT(txt), nl=False)

    while True:
        response = complete(model=tuned_model, prompt=txt, temperature=0, max_tokens=200, stop=(" ]\n", " ] "))

        completion = response + " ]"
        echo(styleMathGPT(completion))

        txt += f"{completion}\n"
        if completion.endswith("[ resetall. ]"): break

    echo()

if __name__ == "__main__":
    cli()
