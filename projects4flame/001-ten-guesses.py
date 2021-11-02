#!/usr/bin/env python

import random

counter = 0
lower_bound = 1
upper_bound = 99
sekret = random.randint(lower_bound, upper_bound)

print("I have a sekret? It's a number in the range %d-%d. "
      "Can you guess it?" % (lower_bound, upper_bound))
while True:
    counter += 1
    guess = int(input("your %d%s guess: " % (counter,
            "st" if counter == 1 else "nd" if counter == 2 else
            "rd" if counter == 3 else "th")))
    if guess < lower_bound or guess > upper_bound:
        print("You are not very smart, are you?")
        continue
    if guess < sekret:
        print("Nah, that's too low.")
        lower_bound = guess+1
        continue
    if guess > sekret:
        print("Nah, that's too high.")
        upper_bound = guess-1
        continue
    print("Wow! You have found out my sekret!")
    break

