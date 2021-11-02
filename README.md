Various unsorted experiments. One branch per year:
- [handicraft-2021](https://github.com/clairexen/handicraft/tree/handicraft-2021)

Checking out one year:
```sh
git clone --single-branch --no-tags -b "handicraft-YYYY" \
    "git@github.com:clairexen/handicraft.git" "handicraft-YYYY"
```

Checking out all years:
```sh
for ((year=2008; year<=2021; year++)); do
  git clone --single-branch --no-tags -b "handicraft-${year}" \
      "git@github.com:clairexen/handicraft.git" "handicraft-${year}"
done
```
