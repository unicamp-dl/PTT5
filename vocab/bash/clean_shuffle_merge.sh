#!/bin/bash

DUMPS_DIR=$1
OUTPUT_FILE=$2

for file in $(find $DUMPS_DIR | grep 'wiki_'| sort -R)
do
    sed $file -e 's/<[^>]*>//g' | sed -e '/^[[:blank:]]*$/ d' | sort -R >> $OUTPUT_FILE
    echo "File $file parsed and appended to $OUTPUT_FILE"
done

