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

import requests, datetime, json

try:
    with open("wordledb.json") as f:
        database = json.load(f)
except FileNotFoundError as ex:
    print()
    print("!!!  Database file not found! Create empty JSON file and retry:")
    print("!!!  $ echo '{}' > wordledb.json")
    print()
    raise ex
except json.decoder.JSONDecodeError as ex:
    print()
    print("!!!  Database file is corrupt! Fix or create a new empty JSON file:")
    print("!!!  $ echo '{}' > wordledb.json")
    print()
    raise ex

print(f"Initial database size: {len(database)}")

day = datetime.date(2021, 6, 18)
while True:
    day = day + datetime.timedelta(days=1)
    date = day.strftime('%Y-%m-%d')
    if date in database: continue

    url = f'https://www.nytimes.com/svc/wordle/v2/{date}.json'
    print(f"Fetching {url}")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data for {date}. Status code: {response.status_code}")
        break

    print(f"  {response.content}")

    database[date] = response.json()

    database_json = "{\n  " + ",\n  ".join([
        f'"{k}": {json.dumps(v)}' for k, v in sorted(database.items())
    ]) + "\n}"

    with open("wordledb.json", "w") as f:
        print(database_json, file=f)

print(f"Final database size: {len(database)}")
