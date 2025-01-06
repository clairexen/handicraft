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
bash -ex clone-year.sh YYYY
```

Checking out all years:
```sh
bash -ex clone-year.sh {2008..2024}
```

Creating a new year:
```sh
create_handicraft_year() { year="$1"; (
	set -ex
	git clone --reference "$PWD" git@github.com:clairexen/handicraft.git handicraft-${year}
	cd handicraft-${year}
	git checkout --orphan handicraft-${year}
	rm *; touch .gitignore; git add -u .gitignore
	git commit -sm "Create handicraft-${year}"
	git push --set-upstream origin handicraft-${year}
	cd ..
	rm -rf handicraft-${year}
	bash -ex clone-year.sh ${year}
); }
create_handicraft_year 2024
```
