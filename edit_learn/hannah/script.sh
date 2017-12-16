#!/bin/bash

check_file() {
	[ ! -e "$1" ] && echo "$1"
}

while read -r line; do
	check_file "$line"
done < nefs.lst
