#!/usr/bin/env python3

import json

def collect_data(election_type):
    data = None
    outdata = dict()

    with open("data.json") as f:
        data = json.load(f)

    for state in data.keys():
        for county in data[state].keys():
            for office in data[state][county].keys():
                if office[0] != election_type:
                    continue

                if office not in outdata:
                    outdata[office] = dict()

                for candidate, votes in data[state][county][office].items():
                    if candidate not in outdata[office]:
                        outdata[office][candidate] = 0
                    outdata[office][candidate] += votes

    dem_votes = 0
    gop_votes = 0
    other_votes = 0

    dem_seats = 0
    gop_seats = 0
    other_seats = 0

    print()
    for office in sorted(outdata.keys()):
        second_best_votes = 0
        best_candidate = None
        best_votes = -1
        total = 0

        for candidate, votes in outdata[office].items():
            if "(Dem)" in candidate:
                dem_votes += votes
            elif "(GOP)" in candidate:
                gop_votes += votes
            else:
                other_votes += votes

            if votes > best_votes:
                best_candidate = candidate
                second_best_votes = best_votes
                best_votes = votes
            elif votes > second_best_votes:
                second_best_votes = votes

            total += votes

        if "(Dem)" in best_candidate:
            dem_seats += 1
        elif "(GOP)" in best_candidate:
            gop_seats += 1
        else:
            other_seats += 1

        if (total == 0) or (total == best_votes):
            print("%-8s %40s  -----" % (office, best_candidate))
        else:
            print("%-8s %40s  %4.1f%%  (+%4.1f%%)" % (office,
                    best_candidate, 100 * best_votes / total,
                    100 * (best_votes-second_best_votes) / total))

    total_votes = dem_votes + gop_votes + other_votes
    total_seats = dem_seats + gop_seats + other_seats

    print()
    print("Dem:     %8d (%4.1f%%) votes      %3d (%4.1f%%) seats" % (dem_votes, 100*dem_votes/total_votes, dem_seats, 100*dem_seats/total_seats))
    print("GOP:     %8d (%4.1f%%) votes      %3d (%4.1f%%) seats" % (gop_votes, 100*gop_votes/total_votes, gop_seats, 100*gop_seats/total_seats))
    print("Other:   %8d (%4.1f%%) votes      %3d (%4.1f%%) seats" % (other_votes, 100*other_votes/total_votes, other_seats, 100*other_seats/total_seats))
    print()

print()
print("----------- House Elections -----------")

collect_data('H')

print("----------- Senate Elections -----------")

collect_data('S')
