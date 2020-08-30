#!/bin/bash

for f in * ; do
    if [[ $f == *.zst ]]; then
        nocache unzstd -d "$f" 
    fi
done

nocache /mnt/c/Program\ Files\ \(x86\)/7-zip/7z.exe e *.bz2
nocache /mnt/c/Program\ Files\ \(x86\)/7-zip/7z.exe e *.xz

for f in * ; do
    if [[ ! -d $f ]]; then
        case $f in *.*) continue;; esac
        nocache mv  -- "$f" "${f}.json"
    fi
done
