#!/usr/bin/env python3 

import urllib.request 

print("Download starting...")

months = ["0"+str(i) for i in range(1, 9)] 
months.extend([str(i) for i in range(10, 13)])
years = list(range(2011, 2020))
extensions = ["bz2", "xz", "bz2"]

urls = [f"https://files.pushshift.io/reddit/submissions/RS_{i}-{j}.{k}" for i
        in years for j in months for k in extensions]

print(urls)

for u in urls:
    try:
        fn = u.split("/")[-1]
        urllib.request.urlretrieve(u, f"./src/data/{fn}")
        print(f"Downloaded data from {u}")
    except:
        print(f"Downloading {u} failed")
