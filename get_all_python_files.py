import os
import shutil

lst = []

for r, d, f in os.walk('.'):
    for file in f:
        if file.endswith(".rs") or file.endswith(".sh") or file.endswith(".py"):
            lst.append(os.path.join(r, file))
         
for i in lst:
    shutil.copy(i, '../upload/')

