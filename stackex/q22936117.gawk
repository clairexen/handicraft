# http://stackoverflow.com/questions/22936117/awk-match-column-values-from-2-files-if-their-numerical-values-are-close
ARGIND == 1 {
    heystrack[$1] = $2;
}
ARGIND == 2 {
    bestdiff=-1;
    for (v in heystrack)
        if (bestdiff < 0 || (v-$1)**2 < bestdiff) {
            bestkey=heystrack[v];
            bestdiff=(v-$1)**2;
        }
    if (bestdiff < 1.5**2)
        print $1, $2, bestkey;
}
