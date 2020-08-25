#!/bin/bash

/mnt/c/Program\ Files\ \(x86\)/7-zip/7z.exe e *.bz2
/mnt/c/Program\ Files\ \(x86\)/7-zip/7z.exe e *.zst
/mnt/c/Program\ Files\ \(x86\)/7-zip/7z.exe e *.xz

for f in * ; do
    mv "$f" "$f.json"
done
