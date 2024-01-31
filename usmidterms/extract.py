#!/usr/bin/env python3

import json

data = dict()

with open("states.txt") as fs:
    for state in fs:
        state = state.strip()
        data[state] = dict()
        with open("%s-context.json" % state) as f:
            data[state]["context"] = json.load(f)
        with open("%s-county.json" % state) as f:
            data[state]["county"] = json.load(f)
        with open("%s-state.json" % state) as f:
            data[state]["state"] = json.load(f)

outdata = dict()
parties = set()

for state in data.keys():
    print("Reading %s db" % state)

    nice_state = list(state)
    nice_state[0] = nice_state[0].upper()
    for i in range(1, len(nice_state)):
        if nice_state[i] == "-":
            nice_state[i] = " "
            nice_state[i+1] = nice_state[i+1].upper()
    nice_state = "".join(nice_state)

    if state not in outdata:
        outdata[nice_state] = dict()

    for entry in data[state]["county"]:
        office = entry["officeid"]

        county = None
        for info in data[state]["context"]["division"]["children"]:
            if info["level"] == "county" and info["code"] == entry["fipscode"]:
                county = "%s, %s (fips:%s)" % (info["label"], entry["statepostal"], entry["fipscode"])

        if state == "alaska" and county is None:
            county = "Alaska"
        assert county is not None

        if county not in outdata[nice_state]:
            outdata[nice_state][county] = dict()

        candidate = None
        votes = int(entry["votecount"])

        for info in data[state]["context"]["elections"]:
            slug = info["office"]["slug"]
            if office == 'H' and "-house-" not in slug:
                continue
            if office == 'G' and not slug.endswith("-governor"):
                continue
            if office == 'S' and "-senate-" not in slug:
                continue

            if candidate is None:
                for info2 in info["candidates"]:
                    if info2["ap_candidate_id"] in [("polid-%d" % int(entry["polid"])),
                                                 ("%d-polid-%d" % (int(entry["raceid"]), int(entry["polid"])))]:
                        candidate = "%s, %s (%s)" % (info2["last_name"], info2["first_name"], info2["party"])
                        if office == 'H':
                            office = "%s-%s-%s" % (office, entry["statepostal"], info["division"]["code"])
                        else:
                            office = "%s-%s" % (office, info["division"]["code"])
                        break

        if candidate is None:
            print(state, json.dumps(entry, indent=4))
            print(json.dumps(data[state]["context"]["elections"], indent=4))

        assert candidate is not None

        if office not in outdata[nice_state][county]:
            outdata[nice_state][county][office] = dict()

        if candidate not in outdata[nice_state][county][office]:
            outdata[nice_state][county][office][candidate] = 0

        outdata[nice_state][county][office][candidate] += votes

print("Writing data.json")
with open("data.json", "w") as f:
    json.dump(outdata, f, indent=4, sort_keys=True)

print("Writing data.txt")
with open("data.txt", "w") as f:
    last_office = None
    for state in sorted(outdata.keys()):
        print(state, file=f)
        for county in sorted(outdata[state].keys()):
            print("  %s" % county, file=f)
            for office in sorted(outdata[state][county].keys()):
                for candidate in sorted(outdata[state][county][office].keys()):
                    line = ["", candidate, outdata[state][county][office][candidate]]

                    if last_office != office:
                        last_office = office
                        line[0] = office

                    print("    %-8s %40s %10d" % tuple(line), file=f)

print("Done.")
