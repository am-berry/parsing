# Simple script to calculate the average scores of the three separate runs of LexRank

import json
import os

files = os.listdir()
relevant_files = [file for file in files if ".json" in file]
print(relevant_files)

r1f = 0
r1r = 0
r1p = 0
r2f = 0
r2r = 0
r2p = 0
rlf = 0
rlr = 0
rlp = 0

for f in relevant_files:
    with open(f, 'r') as handler:
        data = json.load(handler)
    r1f += data['rouge-1']['f']/3 
    r1r += data['rouge-1']['r']/3
    r1p += data['rouge-1']['p']/3 

    r2f += data['rouge-2']['f'] /3 

    r2r += data['rouge-2']['r']/3 

    r2p += data['rouge-2']['p']/3 

    rlf += data['rouge-l']['f'] /3 

    rlr += data['rouge-l']['r']/3 


    rlp += data['rouge-l']['p']/3 

print(r1f, r1r, r1p,r2f, r2r, r2p,rlf, rlr, rlp)
