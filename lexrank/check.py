import pandas as pd
import time
import sentence_transformers

model = sentence_transformers.SentenceTransformer('../models/roberta-large-ft/')

asdfsdfsdf = model.encode(['asdfasdfasdfasdf'])
t = time.time()
a = 'abc'
b = 'cde'
c = 'fgh'
ae = model.encode([a])
be = model.encode([b])
ce = model.encode([c])
print(time.time() - t)

t=time.time()
to = [a,b,c]
de = model.encode(to)
print(time.time() -t)
