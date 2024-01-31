Various unsorted experiments. One branch per year:
- [handicraft-2008](https://github.com/clairexen/handicraft/tree/handicraft-2008)
- [handicraft-2009](https://github.com/clairexen/handicraft/tree/handicraft-2009)
- [handicraft-2010](https://github.com/clairexen/handicraft/tree/handicraft-2010)
- [handicraft-2011](https://github.com/clairexen/handicraft/tree/handicraft-2011)
- [handicraft-2012](https://github.com/clairexen/handicraft/tree/handicraft-2012)
- [handicraft-2013](https://github.com/clairexen/handicraft/tree/handicraft-2013)
- [handicraft-2014](https://github.com/clairexen/handicraft/tree/handicraft-2014)
- [handicraft-2015](https://github.com/clairexen/handicraft/tree/handicraft-2015)
- [handicraft-2016](https://github.com/clairexen/handicraft/tree/handicraft-2016)
- [handicraft-2017](https://github.com/clairexen/handicraft/tree/handicraft-2017)
- [handicraft-2018](https://github.com/clairexen/handicraft/tree/handicraft-2018)
- [handicraft-2019](https://github.com/clairexen/handicraft/tree/handicraft-2019)
- [handicraft-2020](https://github.com/clairexen/handicraft/tree/handicraft-2020)
- [handicraft-2021](https://github.com/clairexen/handicraft/tree/handicraft-2021)
- [handicraft-2022](https://github.com/clairexen/handicraft/tree/handicraft-2022)
- [handicraft-2023](https://github.com/clairexen/handicraft/tree/handicraft-2023)
- [handicraft-2024](https://github.com/clairexen/handicraft/tree/handicraft-2024)

Checking out one year:
```sh
git clone --single-branch --no-tags -b "handicraft-YYYY" \
    "git@github.com:clairexen/handicraft.git" "handicraft-YYYY"
```

Checking out all years:
```sh
for ((year=2008; year<=2024; year++)); do
  git clone --single-branch --no-tags -b "handicraft-${year}" \
      "git@github.com:clairexen/handicraft.git" "handicraft-${year}"
done
```
