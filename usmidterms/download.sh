#!/bin/bash
set -ex
download_state() {
	echo $1 >> states.txt
	wget -O ${1}-context.json https://www.politico.com/election-results/2018/$1/context.json
	wget -O ${1}-county.json https://www.politico.com/election-results/2018/$1/county.json
	wget -O ${1}-state.json https://www.politico.com/election-results/2018/$1/state.json
}
rm -f states.txt
download_state alabama
download_state alaska
download_state arizona
download_state arkansas
download_state california
download_state colorado
download_state connecticut
download_state delaware
download_state florida
download_state georgia
download_state hawaii
download_state idaho
download_state illinois
download_state indiana
download_state iowa
download_state kansas
download_state kentucky
download_state louisiana
download_state maine
download_state maryland
download_state massachusetts
download_state michigan
download_state minnesota
download_state mississippi
download_state missouri
download_state montana
download_state nebraska
download_state nevada
download_state new-hampshire
download_state new-jersey
download_state new-mexico
download_state new-york
download_state north-carolina
download_state north-dakota
download_state ohio
download_state oklahoma
download_state oregon
download_state pennsylvania
download_state rhode-island
download_state south-carolina
download_state south-dakota
download_state tennessee
download_state texas
download_state utah
download_state vermont
download_state virginia
download_state washington
download_state west-virginia
download_state wisconsin
download_state wyoming
: OK
