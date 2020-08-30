#!/bin/bash

for f in * ; do
    if [[ $f == *.zst ]]; then
        nocache unzstd -d "$f" 
    fi
done

nocache 7z e *.bz2
nocache 7z e *.xz

for f in * ; do
    if [[ ! -d $f ]]; then
        case $f in *.*) continue;; esac
        nocache mv  -- "$f" "${f}.json"
    fi
done
