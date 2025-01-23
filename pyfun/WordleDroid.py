#!/usr/bin/env python3
#
# The New York Times "WordleBot" is behnd a paywall.  :/
# So I wrote my own "WordleDroid" which I can run locally.
#
# Copyright (C) 2025  Claire Xenia Wolf <claire@clairexen.net>
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

import numpy as np
from collections import defaultdict
from types import SimpleNamespace
import sys, copy, math

shuffle = True
usegood = False
subexpand = False
subsample = 100
tgsamples = 1000
hardmode = True
playmode = False
autoplaysecret = None
wordledump = False
cmdname = sys.argv[0]
args = sys.argv[1:]

def getWordleWordOfTheDay(pastDays=0, quiet=False):
    global totalOfficialWordles
    import requests, datetime

    date = (datetime.datetime.today() - datetime.timedelta(days=pastDays)).strftime('%Y-%m-%d')
    url = f'https://www.nytimes.com/svc/wordle/v2/{date}.json'

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve Wordle solution. Status code: {response.status_code}")
        assert False

    data = response.json()
    if "editor" not in data:
        data["editor"] = "Unknown"
    if "days_since_launch" not in data:
        data["days_since_launch"] = 0
    if pastDays == 0:
        totalOfficialWordles = data['days_since_launch']

    if not quiet:
        print(f"Using official NYT Wordle word #{data['days_since_launch']+1} " +
              f"for {data['print_date']}, edited by {data['editor']}.")
    return data['solution']

while len(args):
    match args[0]:
        case "-s": # sort word lists
            del args[0]
            shuffle = False
            continue
        case "-H": # disable "hard mode"
            del args[0]
            hardmode = False
            continue
        case "-g": # use pre-generated "good" (first) words
            del args[0]
            usegood = True
            continue
        case "-n": # use N random samples when len(candidates) > N
            del args[0]
            subsample = int(args[0])
            del args[0]
            continue
        case "-N": # use a total of N random samples for each guess
            del args[0]
            tgsamples = int(args[0])
            del args[0]
            continue
        case "-e": # expand number of subsamples in pre-defined steps
            del args[0]
            subexpand = True
            continue
        case "-p": # no analysis, just play a normal game of wordl
            assert not playmode and autoplaysecret is None
            del args[0]
            playmode = True
            continue
        case "-P": # play todays official (NYT) Wordle
            assert not playmode and autoplaysecret is None
            del args[0]
            playmode = True
            autoplaysecret = getWordleWordOfTheDay()
            continue
        case "-w": # play (using the given arg as secret word)
            assert not playmode and autoplaysecret is None
            del args[0]
            playmode = True
            autoplaysecret = args[0]
            del args[0]
            continue
        case "-a": # auto-play (using the given arg as secret word)
            assert not playmode and autoplaysecret is None
            del args[0]
            autoplaysecret = args[0]
            del args[0]
            continue
        case "-A": # auto-play todays official (NYT) Wordle
            assert not playmode and autoplaysecret is None
            del args[0]
            autoplaysecret = getWordleWordOfTheDay()
            continue
        case "-W": # produce list of (new) wordle words (hidden from help())
            del args[0]
            wordledump = True
            continue
        case _:
            break

# This class is not very efficient, but it gets the job done.
#
# TODO: Re-implement this class using a faster approach.
# For example, .clues[idx] and .stops could be 26-bit wide bitmaps,
# and each letter could be represented by such a bitmap with exactly
# one bit set.
#
class Wordle:
    def __init__(self, pattern=None):
        self.clues = [
                set("abcdefghijklmnopqrstuvwxyz"),
                set("abcdefghijklmnopqrstuvwxyz"),
                set("abcdefghijklmnopqrstuvwxyz"),
                set("abcdefghijklmnopqrstuvwxyz"),
                set("abcdefghijklmnopqrstuvwxyz")
        ]
        self.stops = set()
        self.letters = list()
        self.candidates = None

    def __str__(self):
        s = "Wordle(\"/"
        for i in range(5):
            if len(self.clues[i]) < 14:
                s += "".join(sorted(self.clues[i])).upper()
            else:
                s += "".join(sorted(set("abcdefghijklmnopqrstuvwxyz").difference(self.clues[i])))
            s += "/"
        s += "".join(sorted(self.letters)).upper()
        s += "".join(sorted(self.stops))
        return s + "\")"

    def loadPattern(self, pattern):
        assert pattern.startswith("/")
        A = (pattern[1:]+"////////").split("/")

        for i in range(5):
            c = A[i]
            if c in ("", "_", "."):
                pass
            elif c == c.upper():
                self.clues[i] &= set(c.lower())
            elif c == c.lower():
                self.clues[i] = self.clues[i].difference(set(c))
            else:
                assert False

        for c in A[5]:
            if c == c.upper():
                self.letters += c.lower()
            else:
                self.stops.add(c)

        self.candidates = None

    def processGuess(self, guess, result):
        oldLettersCnt = defaultdict(int)
        for c in self.letters:
            oldLettersCnt[c] += 1

        self.letters.clear()

        for i in range(5):
            g, r = guess[i], result[i]
            assert g == g.lower()
            if g.upper() == r:
                oldLettersCnt[g] -= 1
                self.letters.append(g)
                self.clues[i] &= set(g)
            elif g == r:
                oldLettersCnt[g] -= 1
                self.letters.append(g)
                self.clues[i].discard(g)
            else:
                self.stops.add(g)

        for c in oldLettersCnt:
            for i in range(oldLettersCnt[c]):
                self.letters.append(c)

        self.candidates = None

    def findCandidates(self):
        if self.candidates is not None:
            return

        self.candidates = list()
        for word in words:
            letters = self.letters.copy()
            for i in range(5):
                if word[i] not in self.clues[i]:
                    break
                if word[i] in letters:
                    idx = letters.index(word[i])
                    letters = letters[:idx] + letters[idx+1:]
                elif word[i] in self.stops:
                    break
            else:
                if len(letters) == 0:
                    self.candidates.append(word)

    def resultCase(self, guess, secret):
        if hardmode:
            self.findCandidates()
            assert guess in self.candidates
            assert secret in self.candidates

        result = list("_____")
        secretCopy = list(secret)

        for i in range(5):
            g, s = guess[i], secret[i]
            if g == s:
                result[i] = g.upper()
                secretCopy[i] = None

        for i in range(5):
            g, s = guess[i], secret[i]
            if g == s or g not in secretCopy: continue
            k = secretCopy.index(g)
            secretCopy[k] = None
            result[i] = g

        return "".join(result)

    def assumeCase(self, guess, secret):
        if hardmode:
            self.findCandidates()
            assert guess in self.candidates
            assert secret in self.candidates

        w = copy.deepcopy(self)
        result = w.resultCase(guess, secret)
        w.processGuess(guess, result)
        w.findCandidates()
        return w, result

    def analyze(self, batch, quiet=False, progress=False):
        progsnippets = [
                "1\b", "2\b", "3\b", "4\b", "5\b",
                "6\b", "7\b", "8\b", "9\b", "."
        ]
        andat = SimpleNamespace(
                    guesses=set(),
                    avgdata=dict(),
                    msrdata=dict(),
                    maxdata=dict(),
                )

        if not quiet and not progress:
            batchsizedigits = len(f"{len(batch):d}")
            print(f"+ {' '*(2*batchsizedigits+1)} GUESS    AVG    MSR    MAX")

        N = subsample if not tgsamples else int(math.ceil(min(subsample, tgsamples/len(batch))))

        secrets = self.candidates
        if N and len(secrets) > N:
            secrets = list(np.random.choice(secrets, N, False))

        for i, guess in enumerate(batch):
            if progress:
                print(progsnippets[0], end="", flush=True)
                progsnippets = progsnippets[1:] + progsnippets[:1]

            if not quiet and not progress:
                print(f"- {i+1:{batchsizedigits}d}/{len(batch):d} {guess}", end="", flush=True)

            results = list()
            for secret in secrets:
                w, _ = self.assumeCase(guess, secret)
                results.append(len(w.candidates))

            if not quiet and not progress:
                print(f" {avg(results):6.1f} {msr(results):6.1f} {max(results):6.1f}", end="")
                print(f"   {results}" if len(results) < 15 else "")

            andat.guesses.add(guess)
            andat.avgdata[guess] = avg(results)
            andat.msrdata[guess] = msr(results)
            andat.maxdata[guess] = max(results)

        avgmax = avg(andat.maxdata.values())
        g1 = set([g for g, v in andat.maxdata.items() if v <= avgmax])
        minmsr = min([andat.msrdata[g] for g in g1])
        g2 = set([g for g in g1 if andat.msrdata[g] <= minmsr+0.05])
        minavg = min([andat.avgdata[g] for g in g2])
        g3 = set([g for g in g2 if andat.avgdata[g] <= minavg+0.05])
        minmax = min([andat.maxdata[g] for g in g2])
        andat.best = set([g for g in g2 if andat.maxdata[g] <= minmax+0.05])

        if not quiet and not progress:
            print(f"Best guesses: {' '.join(sorted(andat.best))}")

        return andat

    def autoplay(self, secret):
        self.findCandidates()
        if hardmode:
            assert secret in self.candidates

        guess = None
        while len(self.candidates) > 1 or guess != secret:
            if len(self.candidates) == len(words):
                andat = None
                guess = np.random.choice(list(goodwords))
            else:
                n = len(self.candidates)
                print(f"Analyzing {n} remaining candidates..", end="")
                andat = self.analyze(self.candidates, progress=True)
                print(".")
                guess = np.random.choice(list(andat.best))

            result = self.resultCase(guess, secret)
            self.processGuess(guess, result)
            self.findCandidates()

            if andat is None:
                print(f"Automatic Guess: {guess}/{result} -> {self}")
            else:
                print(f"Automatic Guess: {guess}/{result} ({andat.avgdata[guess]:.1f}/" +
                      f"{andat.msrdata[guess]:.1f}/{andat.maxdata[guess]:.1f}) -> {self}")

        if len(self.candidates) == 0:
            print(f"[NO REMAINIG CANDIDATES!]  (secret word was \"{secret}\")")
        else:
            guess = self.candidates[0]
            if guess == secret:
                print(f"Success: {guess}  \\o/")
            else:
                print(f"Fail: {guess}  [NOT THE SECRET!]  (secret word was \"{secret}\")")

def help():
    print()
    print(f"Usage:")
    print(f"  {cmdname} [options...] args...")
    print()
    print(f"Usage examples:")
    print(f"  {cmdname} /s/a/s/_/_/AShrugmsy")
    print(f"  {cmdname} shrug/s____ massy/_as__ ideas/___AS novas/___AS")
    print(f"  {cmdname} -n 0 aloes reals tales tares saner stare plate")
    print(f"  {cmdname} fraud/___uD upend/u___D squid")
    print(f"  {cmdname} -s //t//it//ITadelrs joint")
    print(f"  {cmdname} -H -a point stair delta")
    print()
    print(f"Options:")
    print(f"  -s  ........  sort (do not shuffle) word lists")
    print(f"  -H  ........  disable \"hard mode\" (allow all guess words)")
    print(f"  -g  ........  use pre-generated set of \"good\" (first) words")
    print(f"  -n N  ......  use N random samples when len(candidates) > N")
    print(f"  -N N  ......  use at most N total samples for creating a guess")
    print(f"  -e  ........  expand number of subsamples in pre-defined steps")
    print()
    print(f"Play and Autoplay Options (use max. 1):")
    print(f"  -p .........  no analysis, just play a normal game of wordl")
    print(f"  -w word  ...  play (using the given arg as secret word)")
    print(f"  -a word  ...  auto-play (using the given arg as secret word)")
    print(f"  -P .........  play todays official (NYT) Wordle")
    print(f"  -A .........  auto-play todays official (NYT) Wordle")
    print()
    print(f"State Arg (only valid as first arg):")
    print(f"  [...TBD...]")
    print()
    print(f"Guess Arg (encodes a single guess and the results it produced):")
    print(f"  [...TBD...]")
    print()
    print(f"Query Arg (a single 5-letter word wthout results):")
    print(f"  [...TBD...]")
    print()
    sys.exit(1)

def main():
    global subsample

    if not args and not playmode and not wordledump and autoplaysecret is None:
        help()

    if args and args[0].startswith("-"):
        help()

    # create/download the list of all official wordle words
    if wordledump:
        print()
        print(f"Wordlist Stats:")
        print(f"    {len(words)=:4d}")
        print(f"    {len(words)-len(dictwords  )=:4d}, {len(wordlewords)=:4d}")
        print(f"    {len(words)-len(wordlewords)=:4d}, {len(dictwords  )=:4d}")
        print()
        print(f"Wordle-Only Words:")
        outwords = sorted(wordlewords-dictwords)
        for i in range(0, len(outwords), 12):
            print(" ".join(outwords[i:i+12]))
        print()
        i = 0
        xtrawords = list()
        while i == 0 or i <= totalOfficialWordles:
            w = getWordleWordOfTheDay(i)
            print(f"+ {w}", flush=True)
            if w in wordlewords: break
            xtrawords.append(w)
            i += 1
        print()
        print(f"Got {len(xtrawords)} new Wordle words{':' if xtrawords else '.'}")
        xtrawords = list(reversed(xtrawords))
        for i in range(0, len(xtrawords), 12):
            print(" ".join(xtrawords[i:i+12]))
        print()
        sys.exit(0)

    if len(args) and "/" not in args[0] and autoplaysecret is None:
        wordle = Wordle()
        wordle.findCandidates()
        wordle.analyze(args)
        if subexpand and subsample:
            for step in (50, 100, 200, 400, 800, 1600, 3200):
                if subsample < step:
                    subsample = step
                    print()
                    print(f"with n={subsample}:")
                    wordle.analyze(args)
        sys.exit(0)

    wordle = Wordle()
    if len(args) and args[0].startswith("/"):
        wordle.loadPattern(args[0])
        print(wordle)
        del args[0]

    lastquery = False
    for guess in args:
        assert not guess.startswith("/")
        if "/" in guess:
            guess, result = guess.split("/")
            wordle.processGuess(guess, result)
        elif autoplaysecret is not None:
            result = wordle.resultCase(guess, autoplaysecret)
            wordle.processGuess(guess, result)
        else:
            print("_"*50)
            print(f"+       {guess}")
            wordle.findCandidates()
            for secret in wordle.candidates:
                w, r = wordle.assumeCase(guess, secret)
                print(f"- {secret}/{r} -> {' '.join(sorted(w.candidates))}")
            lastquery = True
            continue
        if lastquery:
            print("_"*50)
            lastquery = False
        print(f"Cmdline Guess: {guess}/{result} -> {wordle}")

    if lastquery:
        return

    if playmode:
        wordle.findCandidates()
        if autoplaysecret is not None:
            secret = autoplaysecret
        else:
            secret = np.random.choice(wordle.candidates)

        for i in range(1, 7):
            while True:
                guess = input(f"guess {i}> ")
                print("\033[F\033[2K", end="", flush=True) # Go up and clear line

                # cheat codes
                if guess == "?":
                    hints = sorted(wordle.candidates)
                    if len(hints) > 10:
                        short = sorted(np.random.choice(hints, 10, False))
                        input(f"hints> {' '.join(short)} (+{len(hints)-len(short)})")
                    else:
                        input(f"hints> {' '.join(hints)}")
                    print("\033[F\033[2K", end="", flush=True) # Go up and clear line
                    continue
                if guess == "??":
                    input(f"state> {wordle}")
                    print("\033[F\033[2K", end="", flush=True) # Go up and clear line
                    continue
                if guess == "???":
                    input(f"secret> {secret}")
                    print("\033[F\033[2K", end="", flush=True) # Go up and clear line
                    continue

                if len(guess) != 5 or not guess.islower():
                    input(f"error> guess \"{guess}\" is malformed (must be 5 lower case chars)")
                    print("\033[F\033[2K", end="", flush=True) # Go up and clear line
                    continue

                if hardmode:
                    if guess not in words:
                        input(f"error> guess \"{guess}\" not found in word list (hard mode)")
                        print("\033[F\033[2K", end="", flush=True) # Go up and clear line
                        continue
                    if guess not in wordle.candidates:
                        input(f"error> guess \"{guess}\" is not allowed in this game anymore (hard mode)")
                        print("\033[F\033[2K", end="", flush=True) # Go up and clear line
                        continue

                break

            result = wordle.resultCase(guess, secret)
            wordle.processGuess(guess, result)
            wordle.findCandidates()

            print(f"guess {i}> ", end="")
            for j in range(5):
                if guess[j].upper() == result[j]:
                    print("\033[37m\033[42m", end="") # White text, green background
                elif guess[j] == result[j]:
                    print("\033[37m\033[43m", end="") # White text, yellow background
                else:
                    print("\033[37m\033[100m", end="") # White text, gray background
                print(guess[j].upper(), end="")
            print("\033[0m") # Reset both foreground and background to default

            if guess == secret:
                print("[YOU HAVE WON!]")
                break
        else:
            print(f"[GAME OVER!]  (secret word was \"{secret}\")")
        return

    wordle.findCandidates()
    out = wordle.candidates
    if len(wordle.candidates) > 100:
        out = sorted(np.random.choice(wordle.candidates, 100))
        out.append("...")
    print(f"{len(wordle.candidates)} remaining candidates: {" ".join(out)}")

    if autoplaysecret is not None:
        wordle.autoplay(autoplaysecret)
    elif usegood:
        wordle.analyze([c for c in wordle.candidates if c in goodwords])
    else:
        wordle.analyze(wordle.candidates)

def avg(data):
    return sum(data)/len(data)

def msr(data):
    return avg([x*x for x in data])**0.5

dictwords = set("""
abaci aback abaft abase abash abate abbey abbot abeam abets abhor abide
abler abode abort about above abuse abuts abuzz abyss ached aches achoo
acids acing acmes acorn acres acrid acted actor acute adage adapt added
adder addle adept adieu adman admen admit adobe adopt adore adorn adult
adzes aegis aeons aerie affix afire afoot afoul after again agape agate
agave agent agile aging agism aglow agony agree ahead aided aides ailed
aimed aired aisle alarm album alder alert algae alias alibi alien align
alike aline alive allay alley allot allow alloy aloes aloft aloha alone
along aloof aloud alpha altar alter altho altos alums amass amaze amber
amble ameba ameer amend amigo amino amirs amiss amity among amour ample
amply ampul amuck amuse angel anger angle angry angst anime anion anise
ankhs ankle annex annoy annul anode anons anted antes antic antis anvil
aorta apace apart aphid aping appal apple apply apron apses apter aptly
aquae aquas arbor arced ardor areas arena argon argot argue arias arise
armed armor aroma arose array arrow arson artsy ascot ashed ashen ashes
aside asked askew aspen aspic assay asses asset aster astir atlas atoll
atoms atone atria attar attic audio audit auger aught augur aunts aurae
aural auras autos avail avast avers avert avian avoid avows await awake
award aware awash awful awing awoke axial axing axiom axles axons azure
baaed babel babes backs bacon badge badly bagel baggy bails baits baize
baked baker bakes balds baled bales balks balky balls balms balmy balsa
banal bands bandy banes bangs banjo banks banns barbs bards bared barer
bares barfs barge barks barns baron basal based baser bases basic basil
basin basis basks bassi basso baste batch bated bates bathe baths batik
baton batty bauds bawdy bawls bayed bayou beach beads beady beaks beams
beans beard bears beast beats beaus beaux bebop becks beech beefs beefy
beeps beers beets befit befog began begat beget begin begot begun beige
being belay belch belie belle bells belly below belts bench bends bents
beret bergs berms berry berth beryl beset besom besot bests betas bevel
bible bicep biddy bided bides bidet biers bight bigot biked biker bikes
bilge bilks bills billy bimbo binds binge bingo biped birch birds birth
bison bitch bites blabs black blade blame bland blank blare blast blaze
bleak bleat bleed bleep blend blent bless blest blimp blind bling blink
blips bliss blitz bloat blobs block blocs blogs blond blood bloom blots
blown blows blued bluer blues bluff blunt blurb blurs blurt blush board
boars boast boats bobby boded bodes bogey boggy bogie bogus boils bolas
boles bolls bolts bombs bonds boned boner bones boney bongo bongs bonny
bonus boobs booby booed books booms boons boors boost booth boots booty
booze boozy borax bored borer bores borne boron bosom bossy bosun botch
bough bound bouts bowed bowel bower bowls boxed boxer boxes bozos brace
bract brads brags braid brain brake brand brash brass brats brave bravo
brawl brawn brays bread break breed brews briar bribe brick bride brief
brier brigs brims brine bring brink briny brisk broad broil broke brood
brook broom broth brown brows bruin brunt brush brusk brute bucks buddy
budge buffs buggy bugle build built bulbs bulge bulgy bulks bulky bulls
bully bumps bumpy bunch bungs bunks bunny bunts buoys burgs burka burly
burns burnt burps burro burrs burst busby bused buses bushy busts butch
butte butts buxom buyer bylaw bytes byway cabal cabby cabin cable cacao
cache cacti caddy cadet cadge cadre caged cages cagey cairn caked cakes
calfs calif calks calls calms calve calyx camel cameo camps campy canal
candy caned canes canny canoe canon canto cants caped caper capes capon
carat carbs cards cared cares caret cargo carol carom carpi carps carry
carts carve cased cases casks caste casts catch cater catty caulk cause
caved caves cavil cawed cease cedar ceded cedes celli cello cells cents
chafe chaff chain chair chalk champ chant chaos chaps chapt charm chars
chart chary chase chasm chats cheap cheat check cheek cheep cheer chefs
chess chest chews chewy chick chide chief child chile chili chill chime
chimp china chink chino chins chips chirp chits chive chock choir choke
chomp chops chord chore chose chows chuck chugs chump chums chunk churl
churn chute cider cigar cilia cinch circa cited cites civet civic civil
clack claim clamp clams clang clank clans claps clash clasp class claws
clean clear cleat clefs cleft clerk clews click cliff climb clime cling
clink clips clipt clits cloak clock clods clogs clomp clone clops close
cloth clots cloud clout clove clown cloys clubs cluck clued clues clump
clung clunk coach coals coast coats cobra cocci cocks cocky cocoa codas
coded codes codex coeds coifs coils coins coked cokes colas colds colic
colon color colts comas combo combs comer comes comet comfy comic comma
conch condo cones conga conic conks cooed cooks cooky cools coons coops
coots coped copes copra copse coral cords cored cores corks corms corns
corny corps costs cotes couch cough could count coupe coups court coven
cover coves covet covey cowed cower cowls coyer coyly cozen crabs crack
craft crags cramp crams crane crank crape craps crash crass crate crave
crawl craws craze crazy creak cream credo creed creek creel creep crepe
crept cress crest crews cribs crick cried crier cries crime crimp crisp
croak croci crock crone crony crook croon crops cross croup crowd crown
crows crude cruel cruet crumb crush crust crypt cubed cubes cubic cubit
cuffs cuing culls cults cumin cunts cupid curbs curds cured curer cures
curie curio curls curly curry curse curst curve curvy cushy cusps cuter
cutup cycle cynic cysts czars dacha daddy dados daffy daily dairy daisy
dales dally dames damns damps dance dandy dared dares darns darts dated
dates datum daubs daunt davit dawns dazed dazes deals dealt deans dears
death debar debit debts debug debut decaf decal decay decks decor decoy
decry deeds deems deeps deers defer deice deify deign deism deity delay
delis dells delta delve demon demos demur denim dense dents depot depth
derby desks deter detox deuce devil dhoti dials diary diced dices dicey
dicks dicky dicta diets digit diked dikes dills dilly dimer dimes dimly
dined diner dines dingo dings dingy dinky diode direr dirge dirks dirty
disco discs disks ditch ditto ditty divan divas dived diver dives divot
divvy dizzy djinn docks dodge dodos doers doffs doggy dogie dogma doily
doing doled doles dolls dolly dolts domed domes donor donut dooms doors
doped dopes dopey dorks dorky dorms dosed doses doted dotes dotty doubt
dough douse doves dowdy dowel downs downy dowry dowse doyen dozed dozen
dozes drabs draft drags drain drake drama drams drank drape drawl drawn
draws drays dread dream dregs dress dried drier dries drift drill drily
drink drips drive droll drone drool droop drops dross drove drown drubs
drugs druid drums drunk dryad dryer dryly ducal ducat duchy ducks ducts
duded dudes duels duets dukes dulls dully dummy dumps dumpy dunce dunes
dungs dunks dunno duped dupes dusky dusts dusty duvet dwarf dweeb dwell
dwelt dyers dying dykes eager eagle earls early earns earth eased easel
eases eaten eater eaves ebbed ebony echos edema edged edger edges edict
edify edits eerie egged egret eider eight eject eking elate elbow elder
elect elegy elfin elide elite elope elude elves email embed ember emcee
emend emery emirs emits emoji emote empty enact ended endow endue enema
enemy enjoy ennui enrol ensue enter entry enure envoy epics epoch epoxy
equal equip erase erect erode erred error erupt essay ester ether ethic
ethos euros evade evens event every evict evils evoke ewers exact exalt
exams excel execs exert exile exist exits expel expos extol extra exude
exult eying eyrie fable faced faces facet facts faded fades fagot fails
faint fairs fairy faith faked faker fakes fakir falls false famed fancy
fangs fanny farce fared fares farms farts fasts fatal fated fates fatty
fault fauna fauns favor fawns faxed faxes fazed fazes fears feast feats
fecal feces feeds feels feign feint fells felon felts femur fence fends
feral ferns ferry fests fetal fetch feted fetid fetus feuds fever fewer
fezes fiats fiber fiche fiefs field fiend fiery fifes fifth fifty fight
filch filed files filet fills filly films filmy filth final finch finds
fined finer fines finis finks finny fiord fired fires firms first firth
fishy fists fitly fiver fives fixed fixer fixes fizzy fjord flack flags
flail flair flake flaky flame flank flaps flare flash flask flats flaws
flays fleas fleck flees fleet flesh flick flied flier flies fling flint
flips flirt flits float flock floes flogs flood floor flops flora floss
flour flout flown flows flubs flues fluff fluid fluke fluky flume flung
flunk flush flute flyby flyer foals foams foamy focal focus fogey foggy
foils foist folds folio folks folly fondu fonts foods fools foots foray
force fords fores forge forgo forks forms forte forth forts forty forum
fouls found fount fours fowls foxed foxes foyer frack frail frame franc
frank frats fraud frays freak freed freer frees fresh frets friar fried
frier fries frill frisk frizz frock frogs frond front frost froth frown
froze fruit frump fryer fucks fudge fuels fugue fulls fully fumed fumes
funds fungi funks funky funny furls furor furry furze fused fuses fussy
fusty futon fuzed fuzes fuzzy gabby gable gaffe gaffs gaged gages gaily
gains gaits galas gales galls gamed gamer games gamey gamin gamma gamut
gangs gaped gapes garbs gases gasps gassy gated gates gaudy gauge gaunt
gauze gauzy gavel gawks gawky gayer gayly gazed gazer gazes gears gecko
geeks geeky geese gelds gelid genes genie genii genre gents genus geode
germs getup ghost ghoul giant gibed gibes giddy gifts gilds gills gilts
gimme gimpy gipsy girds girls girth girts gismo given gives gizmo glade
glads gland glare glass glaze gleam glean glens glide glint glitz gloat
globe globs gloom glory gloss glove glows glued glues gluey gluts glyph
gnarl gnash gnats gnawn gnaws gnome goads goals goats godly gofer going
golds golfs golly gonad goner gongs gonna goods goody gooey goofs goofy
gooks goons goose gored gores gorge gorse gotta gouge gourd gouty gowns
grabs grace grade grads graft grail grain grams grand grant grape graph
grasp grass grate grave gravy grays graze great grebe greed green greet
greys grids grief grill grime grimy grind grins gripe grips grist grits
groan groin groom grope gross group grout grove growl grown grows grubs
gruel gruff grunt guano guard guava guess guest guide guild guile guilt
guise gulag gulch gulfs gulls gully gulps gumbo gummy gunny guppy gurus
gushy gusto gusts gusty gutsy guyed gybed gybes gypsy gyros habit hacks
hafts haiku hails hairs hairy hakes haled haler hales halls halon halos
halts halve hands handy hangs hanks hanky happy hardy hared harem hares
harks harms harps harpy harry harsh harts hasps haste hasty hatch hated
hater hates hauls haunt haven haves havoc hawed hawks hayed hazed hazel
hazes heads heady heals heaps heard hears heart heath heats heave heavy
hedge heeds heels hefts hefty heirs heist helix hello helms helot helps
hence henna herbs herds heron heros hertz hewed hewer hexed hexes hicks
hided hides highs hiked hiker hikes hills hilly hilts hinds hinge hints
hippo hippy hired hires hitch hived hives hoagy hoard hoary hobby hobos
hocks hogan hoist hokey hokum holds holed holes holly homed homer homes
homey homie honed hones honey honks honor hooch hoods hooey hoofs hooks
hooky hoops hoots hoped hopes horde horns horny horse horsy hosed hoses
hosts hotel hotly hound hours house hovel hover howdy howls hubby huffs
huffy huger hulas hulks hulls human humid humor humps humus hunch hunks
hunts hurls hurry hurts husks husky hussy hutch hydra hyena hying hymen
hymns hyped hyper hypes hypos iambs icier icily icing icons ideal ideas
idiom idiot idled idler idles idols idyll idyls igloo ikons image imams
imbed imbue impel imply inane inapt inbox incur index indue inept inert
infer infix ingot inked inlay inlet inner input inset inter inure iotas
irate irked irons irony isles islet issue itchy items ivies ivory jabot
jacks jaded jades jails jambs japan japed japes jaunt jawed jazzy jeans
jeeps jeers jehad jello jells jelly jerks jerky jests jetty jewel jibed
jibes jiffy jihad jilts jimmy jinni jinns jived jives jocks johns joins
joint joist joked joker jokes jolly jolts joule joust jowls joyed judge
juice juicy julep jumbo jumps jumpy junco junks junky junta juror kabob
kapok kaput karat karma kayak kazoo kebab kebob keels keens keeps ketch
keyed khaki khans kicks kicky kiddo kiddy kills kilns kilos kilts kinda
kinds kings kinks kinky kiosk kited kites kitty kiwis klutz knack knave
knead kneed kneel knees knell knelt knife knits knobs knock knoll knots
known knows koala kooks kooky kopek krone kudos kudzu label labia labor
laced laces lacks laded laden lades ladle lager lairs laity lakes lamas
lambs lamed lamer lames lamps lance lands lanes lanky lapel lapse larch
lards large largo larks larva laser lasso lasts latch later latex lathe
laths latte lauds laugh lawns laxer laxly layer lazed lazes leach leads
leafs leafy leaks leaky leans leaps leapt learn lease leash least leave
ledge leech leeks leers leery lefts lefty legal leggy legit lemma lemme
lemon lemur lends leper letup levee level lever liars libel licit licks
liege liens lifer lifts light liked liken liker likes lilac lilts limbo
limbs limed limes limit limns limos limps lined linen liner lines lingo
links lions lipid liras lisle lisps lists liter lithe lived liven liver
lives livid llama llano loads loafs loamy loans loath lobby lobed lobes
local locks locus lodes lodge lofts lofty loges logic login logon logos
loins lolls loner longs looks looms loons loony loops loopy loose loots
loped lopes lords lorry loser loses lotto lotus louse lousy louts loved
lover loves lowed lower lowly loxes loyal luaus lubed lubes lucid lucks
lucky lucre lulls lumps lumpy lunar lunch lunge lungs lupin lupus lurch
lured lures lurid lurks lusts lusty lutes lying lymph lynch lyres lyric
macaw maced maces macho macro madam madly magic magma maids mails maims
mains maize major maker makes males malls malts mamas mambo mamma manes
manga mange mango mangy mania manic manly manna manor manse maple march
mares maria marks marry marsh marts masks mason masts match mated mates
matte matts matzo mauls mauve maven mavin maxed maxes maxim maybe mayor
mazes meals mealy means meant meats meaty mecca medal media medic meets
melds melon melts memes memos mends menus meows mercy meres merge merit
merry mesas messy metal meted meter metes metro mewed mewls miaow micra
middy midge midst miens miffs might miked mikes milch miler miles milfs
milks milky mills mimed mimes mimic mince minds mined miner mines minim
minis minks minor mints minty minus mired mires mirth misdo miser mists
misty miter mites mitts mixed mixer mixes moans moats mocha mocks modal
model modem modes mogul moire moist molar molds moldy moles molls molts
momma mommy money monks month mooch moods moody mooed moons moors moose
moots moped mopes moral moray mores morns moron mosey mossy motel motes
moths motif motor motto mound mount mourn mouse mousy mouth moved mover
moves movie mowed mower mucks mucky mucus muddy muffs mufti muggy mulch
mules mulls multi mummy mumps munch mural murks murky mused muses mushy
music musky mussy musts musty muted muter mutes mutts mynah mynas myrrh
myths nabob nacho nacre nadir naiad nails naive naked named names nanny
napes nappy narcs narks nasal nasty natal natty naval navel naves nears
neath necks needs needy neigh nerds nerdy nerve nervy nests never newel
newer newly newsy newts nexus nicer niche nicks niece nifty nigga night
nimbi nines ninja ninny ninth nippy niter nites nixed nixes noble nobly
nodal noddy nodes noels noise noisy nomad nonce nooks noose norms north
nosed noses nosey notch noted notes nouns novae novas novel noway nuder
nudes nudge nuked nukes nulls numbs nurse nutty nylon nymph oaken oakum
oared oases oasis oaten oaths obese obeys obits oboes occur ocean ocher
ochre octal octet odder oddly odium odors offal offed offer often ogled
ogles ogres oiled oinks okays okras olden older oldie olive omega omens
omits onion onset oozed oozes opals opens opera opine opium opted optic
orals orate orbit order organ osier other otter ought ounce ousts outdo
outed outer outgo ovals ovary ovens overs overt ovoid ovule owing owlet
owned owner oxbow oxide ozone paced paces packs pacts paddy padre paean
pagan paged pager pages pails pains paint pairs paled paler pales palls
palms palmy palsy panda panel panes pangs panic pansy pants panty papal
papas papaw paper parch pared pares parka parks parry parse parts party
pasha pasta paste pasts pasty patch pates paths patio patsy patty pause
paved paves pawed pawls pawns payed payee payer peace peach peaks peals
pearl pears pease pecan pecks pedal peeks peels peeps peers peeve pekoe
pelts penal pence pends penes penis penny peons peony peppy perch peril
perks perky perms pesky pesos pests petal peter petty pewee phase phial
phish phlox phone phony photo phyla piano picks picky piece piers piety
piggy pigmy piing piked piker pikes pilaf pilau pilaw piled piles pills
pilot pimps pinch pined pines pings pinks pinky pinto pints pinup pious
piped piper pipes pipit pique pitch pithy piton pivot pixel pixie pizza
place plaid plain plait plane plank plans plant plate plays plaza plead
pleas pleat plied plies plods plops plots plows ploys pluck plugs plumb
plume plump plums plunk plush poach pocks podia poems poesy poets point
poise poked poker pokes pokey polar poled poles polio polka polls polyp
ponds pones pooch poohs pools poops popes poppa poppy porch pored pores
porno ports posed poser poses posit posse posts potty pouch pound pours
pouts power poxes prank prate prawn prays preen preps press preys price
prick pricy pride pried pries prigs prime primp print prior prism privy
prize probe prods profs promo proms prone prong proof props prose prosy
proud prove prowl prows proxy prude prune psalm pshaw psych pubic pucks
pudgy puffs puffy puked pukes pulls pulps pulpy pulse pumas pumps punch
punks punts pupae pupal pupas pupil puppy puree purer purge purls purrs
purse pushy pussy putts putty pwned pygmy pylon pyres pyxes quack quads
quaff quail quake qualm quark quart quash quasi quays queen queer quell
query quest queue quick quids quiet quill quilt quips quire quirk quite
quits quoit quota quote quoth rabbi rabid raced racer races racks radar
radii radio radon rafts ragas raged rages raids rails rains rainy raise
rajah rajas raked rakes rally ramps ranch randy range rangy ranks rants
raped rapes rapid rared rarer rares rasps raspy rated rates ratio ratty
raved ravel raven raves rawer rayon razed razes razor reach react reads
ready realm reals reams reaps rearm rears rebel rebus rebut recap recta
recur redid reeds reedy reefs reeks reels reeve refer refit regal rehab
reign reins relax relay relic remit renal rends renew rents reorg repay
repel reply reran rerun reset resin rests retch retry reuse revel revue
rheas rheum rhino rhyme riced rices ricks rider rides ridge rifer riffs
rifle rifts right rigid rigor riled riles rills rimed rimes rinds rings
rinks rinse riots ripen riper risen riser rises risks risky rites ritzy
rival riven river rivet roach roads roams roans roars roast robed robes
robin robot rocks rocky rodeo roger rogue roils roles rolls roman romps
roods roofs rooks rooms roomy roost roots roped ropes roses rosin rotor
rouge rough round rouse route routs roved rover roves rowdy rowed rowel
rower royal rubes ruble ruddy ruder ruffs rugby ruing ruins ruled ruler
rules rumba rummy rumor rumps runes rungs runny runts rupee rural ruses
rusks rusts rusty saber sable sabre sacks sades sadly safer safes sagas
sager sages sahib sails saint saith salad sales sally salon salsa salts
salty salve salvo samba sames sands sandy saner sappy saree saris sassy
satay sated sates satin satyr sauce saucy sauna saved saver saves savor
savvy sawed saxes scabs scads scald scale scalp scaly scamp scams scans
scant scare scarf scars scary scats scene scent schwa scion scoff scold
scone scoop scoot scope score scorn scour scout scowl scows scram scrap
screw scrip scrod scrub scuba scuds scuff scull scums scurf seals seams
seamy sears seats sects sedan sedge seeds seedy seeks seems seeps seers
segue seize sells semen semis sends senna sense sepal sepia septa serer
serfs serge serum serve servo setup seven sever sewed sewer sexed sexes
shack shade shads shady shaft shags shahs shake shaky shale shall shalt
shame shams shank shape shard share shark sharp shave shawl sheaf shear
sheds sheen sheep sheer sheet sheik shelf shell sherd shied shies shift
shill shims shine shins shiny ships shire shirk shirr shirt shits shlep
shoal shock shoed shoes shone shook shoon shoos shoot shops shore shorn
short shots shout shove shown shows showy shred shrew shrub shrug shtik
shuck shuns shunt shush shuts shyer shyly sibyl sicks sided sides sidle
siege sieve sifts sighs sight sigma signs silks silky sills silly silos
silts since sinew singe sings sinks sinus sired siren sires sirup sisal
sises sissy sitar sited sites sixes sixth sixty sized sizer sizes skate
skeet skein skews skids skied skier skies skiff skill skimp skims skins
skips skirt skits skulk skull skunk skyed slabs slack slags slain slake
slams slang slant slaps slash slate slats slave slays sleds sleek sleep
sleet slept slews slice slick slide slier slily slime slims slimy sling
slink slips slits slobs sloes slogs sloop slope slops slosh sloth slots
slows slued slues slugs slump slums slung slunk slurp slurs slush sluts
slyer slyly smack small smart smash smear smell smelt smile smirk smite
smith smock smoke smoky smote smuts snack snafu snags snail snake snaky
snaps snare snarl sneak sneer snide sniff snipe snips snits snobs snoop
snoot snore snort snots snout snows snowy snubs snuck snuff snugs soaks
soaps soapy soars sober socks sodas sofas softy soggy soils solar soled
soles solid solos solve sonar songs sonic sonny sooth sooty soppy sorer
sores sorry sorta sorts sough souls sound soups soupy sours souse south
sowed sower space spacy spade spake spams spank spans spare spark spars
spasm spate spats spawn spays speak spear speck specs speed spell spelt
spend spent sperm spews spice spicy spied spiel spies spike spiky spill
spilt spine spins spiny spire spite spits splat splay split spoil spoke
spoof spook spool spoon spoor spore sport spots spout sprat spray spree
sprig spuds spume spunk spurn spurs spurt squab squad squat squaw squid
stabs stack staff stage stags staid stain stair stake stale stalk stall
stamp stand stank staph stare stark stars start stash state stats stave
stays stead steak steal steam steed steel steep steer stein stems stent
steps stern stews stick sties stiff stile still stilt sting stink stint
stirs stoat stock stoic stoke stole stomp stone stony stood stool stoop
stops store stork storm story stout stove stows strap straw stray strep
strew strip strop strum strut stubs stuck studs study stuff stump stung
stunk stuns stunt styes style styli suave sucks sudsy suede sugar suing
suite suits sulks sulky sully sumac sumps sunny sunup super surer surfs
surge surly sushi swabs swags swain swami swamp swank swans swaps sward
swarm swash swath swats sways swear sweat sweep sweet swell swept swift
swigs swill swims swine swing swipe swirl swish swoon swoop swops sword
swore sworn swung sylph synch syncs synod syrup tabby table taboo tabus
tacit tacks tacky tacos taffy tails taint taken taker takes tales talks
tally talon tamed tamer tames tamps tango tangs tangy tanks tansy taped
taper tapes tapir tardy tared tares taros tarot tarps tarry tarts taser
tasks taste tasty tatty taunt taupe tawny taxed taxes taxis teach teaks
teals teams tears teary tease teats techs teems teens teeny teeth telex
tells tempi tempo temps tempt tends tenet tenon tenor tense tenth tents
tepee tepid terms terns terry terse tests testy texts thank thaws thees
theft their theme there these theta thick thief thigh thine thing think
thins third thong thorn those thous three threw throb throe throw thrum
thuds thugs thumb thump thyme thymi tiara tibia ticks tidal tided tides
tiers tiffs tiger tight tikes tilde tiled tiles tills tilts timed timer
times timid tines tinge tings tinny tints tipis tipsy tired tires tiros
titan tithe title tizzy toads toady toast today toddy toffy togae togas
toils toked token tokes tolls tombs tomes tonal toned toner tones tongs
tonic tonne tools tooth toots topaz topic toque torch torsi torso torte
torts torus total toted totem totes touch tough tours touts towed towel
tower towns toxic toxin toyed trace track tract trade trail train trait
tramp trams traps trash trawl trays tread treat treed trees treks trend
tress triad trial tribe trice trick tried tries trike trill trims trios
tripe trips trite troll tromp troop trope troth trots trout troys truce
truck trued truer trues truly trump trunk truss trust truth tryst tsars
tubas tubby tubed tuber tubes tucks tufts tulip tulle tumid tummy tumor
tunas tuned tuner tunes tunic tunny turds turfs turns tusks tutor tutus
tuxes twain twang tweak tweed tweet twerk twerp twice twigs twill twine
twins twirl twist twits tying tykes typed types typos tyros tzars udder
ulcer ulnae ulnas ultra umbel umber umiak umped unbar uncle uncut under
undid undue unfit unify union unite units unity unman unpin unsay unset
untie until unwed unzip upend upped upper upset urban urged urges urine
usage users usher using usual usurp usury uteri utter uvula vacua vague
vales valet valid valor value valve vamps vanes vaped vapes vapid vapor
vases vasts vault vaunt veeps veers vegan veils veins velds veldt venal
vends venom vents venue verbs verge verse verve vests vetch vexed vexes
vials viand vibes vicar viced vices video views vigil vigor viler villa
vines vinyl viola viols viper viral vireo virus visas vised vises visit
visor vista vital vivas vivid vixen vizor vocal vodka vogue voice voids
voile voles volts vomit voted voter votes vouch vowed vowel vulva vying
wacko wacks wacky waded wader wades wadis wafer wafts waged wager wages
wagon waifs wails waist waits waive waked waken wakes waled wales walks
walls waltz wands waned wanes wanly wanna wants wards wares warms warns
warps warts warty wasps waste watch water watts waved waver waves waxed
waxen waxes weals weans wears weary weave wedge weeds weedy weeks weeps
weepy weest wefts weigh weird weirs welch welds wells welsh welts wench
wends wetly whack whale whams wharf whats wheal wheat wheel whelk whelp
whens where whets which whiff while whims whine whiny whips whirl whirr
whirs whisk whist white whits whizz whole whoop whore whorl whose wicks
widen wider widow width wield wight wikis wilds wiled wiles wills wilts
wimps wimpy wince winch winds windy wined wines wings winks winos wiped
wiper wipes wired wires wiser wises wisps wispy witch witty wives wizes
woken wolfs woman wombs women woods woody wooed wooer woofs wooly woozy
words wordy works world worms wormy worry worse worst worth would wound
woven wowed wrack wraps wrapt wrath wreak wreck wrens wrest wrier wring
wrist write writs wrong wrote wroth wrung wryer wryly xenon xylem yacht
yacks yahoo yanks yards yarns yawed yawls yawns yeahs yearn years yeast
yells yelps yeses yield yocks yodel yogin yogis yoked yokel yokes yolks
young yours youth yowls yucca yucks yucky yummy yuppy zebra zebus zeros
zests zilch zincs zings zippy zombi zonal zoned zones zooms""".split())

wordlewords = set("""
cigar rebut sissy humph awake blush focal evade naval serve heath dwarf
model karma stink grade quiet bench abate feign major death fresh crust
stool colon abase marry react batty pride floss helix croak staff paper
unfed whelp trawl outdo adobe crazy sower repay digit crate cluck spike
mimic pound maxim linen unmet flesh booby forth first stand belly ivory
seedy print yearn drain bribe stout panel crass flume offal agree error
swirl argue bleed delta flick totem wooer front shrub parry biome lapel
start greet goner golem lusty loopy round audit lying gamma labor islet
civic forge corny moult basic salad agate spicy spray essay fjord spend
kebab guild aback motor alone hatch hyper thumb dowry ought belch dutch
pilot tweed comet jaunt enema steed abyss growl fling dozen boozy erode
world gouge click briar great altar pulpy blurt coast duchy groin fixer
group rogue badly smart pithy gaudy chill heron vodka finer surer radio
rouge perch retch wrote clock tilde store prove bring solve cheat grime
exult usher epoch triad break rhino viral conic masse sonic vital trace
using peach champ baton brake pluck craze gripe weary picky acute ferry
aside tapir troll unify rebus boost truss siege tiger banal slump crank
gorge query drink favor abbey tangy panic solar shire proxy point robot
prick wince crimp knoll sugar whack mount perky could wrung light those
moist shard pleat aloft skill elder frame humor pause ulcer ultra robin
cynic aroma caulk shake dodge swill tacit other thorn trove bloke vivid
spill chant choke rupee nasty mourn ahead brine cloth hoard sweet month
lapse watch today focus smelt tease cater movie saute allow renew their
slosh purge chest depot epoxy nymph found shall stove lowly snout trope
fewer shawl natal comma foray scare stair black squad royal chunk mince
shame cheek ample flair foyer cargo oxide plant olive inert askew heist
shown zesty trash larva forgo story hairy train homer badge midst canny
shine gecko farce slung tipsy metal yield delve being scour glass gamer
scrap money hinge album vouch asset tiara crept bayou atoll manor creak
showy phase froth depth gloom flood trait girth piety goose float donor
atone primo apron blown cacao loser input gloat awful brink smite beady
rusty retro droll gawky hutch pinto egret lilac sever field fluff agape
voice stead berth madam night bland liver wedge roomy wacky flock angry
trite aphid tryst midge power elope cinch motto stomp upset bluff cramp
quart coyly youth rhyme buggy alien smear unfit patty cling glean label
hunky khaki poker gruel twice twang shrug treat waste merit woven needy
clown irony ruder gauze chief onset prize fungi charm gully inter whoop
taunt leery class theme lofty tibia booze alpha thyme doubt parer chute
stick trice alike recap saint glory grate admit brisk soggy usurp scald
scorn leave twine sting bough marsh sloth dandy vigor howdy enjoy valid
ionic equal floor catch spade stein exist quirk denim grove spiel mummy
fault foggy flout carry sneak libel waltz aptly piney inept aloud photo
dream stale begin spell rainy unite medal valet inane maple snarl baker
there glyph avert brave axiom prime drive feast itchy clean happy tepid
undue study eject chafe torso adore woken amber joust infer braid knock
naive apply spoke usual rival probe chord taper slate third lunar excel
aorta poise extra judge condo impel havoc molar manly whine skirt antic
layer sleek belie lemon opera pixie grimy sedan leapt human koala spire
frock adopt chard mucky alter blurb matey elude count maize beefy worry
flirt fishy crave cross scold shirk tasty unlit dance ninth apple flail
stage heady debug giant usage sound salsa magic cache avail kiosk sweat
ruddy riper vague arbor fifty syrup worse polka moose above squat trend
toxic pinky horse regal where revel email birth blame surly sweep cider
mealy yacht credo glove tough duvet staid grout voter untie guano hurry
beset bread every march stock flora ratio smash leafy locus ledge snafu
under qualm borax carat thief agony dwelt whiff hound thump plate kayak
broke unzip ditto joker metro logic circa cedar plaza range sulky horde
guppy below anger ghoul aglow cocoa ethic broom snack acrid scarf canoe
latte plank shorn grief flask brash igloo clerk utter bagel swine ramen
skimp mouse kneel agile jazzy humid nanny beast ennui scout hater crumb
balsa again guard wrong plunk crime maybe strap ranch shyly kazoo frost
crane taste covet grand rodeo guest about tract diner straw bleep mossy
hotel irate venom windy donut cower enter folly earth whirl barge fiend
crone topaz droop flyer tonic flank burly froze whale hobby wheel heart
disco ethos curly bathe style tenth beget party chart anode polyp brook
bully lover empty hello quick wrath snaky index scrub amiss exact magma
quest beach spice verve wordy ocean choir peace write caper audio bride
space onion await giddy birch gnash dwell rouse lucky quote older whisk
clear rayon exert angel music frank close snare stone brush carol right
rocky loyal smile coach azure daddy beret merry while spurt bunch chime
viola binge truth snail skunk knelt uncle agent leaky graph adult mercy
splat occur smirk given tempo cause retry pique noble mason phony grail
bleak noise until ardor mania flare trade limit ninja glaze leash actor
meant green sassy sight trust tardy think queue candy piano pixel queen
throw guide solid tawny scope sushi resin taken genre adapt worst young
woman sleep sharp shift chain house these spent would topic globe bacon
funny table small built touch slope grace evoke phone daisy learn child
three salty mural aging twirl scant lunge cable stony final liner threw
brief route heard doing lunch blond court stole thing large north tweak
still relic block aloof snake ember leggy expel bulky alive cleft micro
verge repel which after place stiff fried never pasta scram talon ascot
stash psalm ridge price match build heavy apart piper smith often sense
devil image forty urban state flame hunch teary clone early cheer grasp
pesky heave local since erupt toxin snort spelt abide lingo shade decay
risen towel sally mayor stung speak realm force taboo frond serum plait
climb wrist finch voila breed merge broth louse whiny steel blimp equip
shank tithe facet raise lucid jolly laser rover overt intro vapid gleam
prune craft prowl diary slice ebony value decal shave musty pious jerky
media tidal outer cumin amass pinch stall tutor briny hitch nicer dingo
exalt swish glide titan bevel skier minus papal gummy chaos basin bravo
stark groom organ ether melon hence crowd manga swung deter angst vault
proud grind prior cover terse scent paint edict bugle dolly savor knead
order drove zebra buddy adage inlay thigh debut crush scoff canon shape
blare gaunt cameo jiffy enact video swoon decoy quite nerdy refer shaft
speck cadet prong forte porch awash juice smock super feral penne chalk
flake scale lower ensue anvil macaw saucy ounce medic scone skiff neigh
shore acorn brace storm lanky meter delay mulch brute leech filet skate
stake crown lithe flunk knave spout mushy camel faint stern widen rerun
owner drawn debit rebel aisle brass harsh broad recur honey beaut fully
press smoke seven teach steam handy torch thank faith brain rider cloud
modem shell wagon title miner lager flour joint mommy carve gusty stain
prone gamut corer grant halve stint fiber dicey spoon shout goofy bossy
frown wreak sandy bawdy tunic easel weird sixth snoop blaze vinyl octet
truly event ready swell inner stoic flown primp uvula tacky visor tally
frail going niche spine pearl jelly twist brown witch slang chock hippo
dogma mauve guile shaky crypt endow shove hilly hyena flung patio plumb
vying boxer drool funky boast scowl hefty stray flash blade brawn sauna
eagle share affix grain decry mambo stare lemur nerve chose cheap relax
cyber sprig atlas draft wafer crawl dingy total cloak fancy knack flint
prose silly rower squid icing reach upper""".split())

goodwords = set("""
acres aegis aeons aides aisle aloes antes arise arose bales bares bates
canes cares cores cotes cries dales dares dates deals deans dears doles
earls earns fares gales gates hales hares hates holes ideas laces lades
lairs lames lanes laser leads leans least liars lines liras loser lures
males manes mares mates names napes nears notes oases orals pales panes
pares pates pears pores races rages rails raise rakes rapes rates reads
reals reams reaps reins rents rices riles rites roles rules saber sabre
safer sager saner sated scare score sepia share shear siren slate slier
smear snare snore solar spare spear stale stare stead stile stole store
tales tames tapes tares taros taser teals tears terns tiers tiles tires
tones tries wares""".split())

words = sorted(dictwords | wordlewords)

if shuffle:
    np.random.shuffle(words)

if __name__ == "__main__":
    main()
