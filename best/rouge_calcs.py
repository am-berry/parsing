import rouge
import pandas as pd
import json
import string

print('Greedy.')
res = pd.read_csv('basic_results_greedy.csv')
gens = res['pred_summary'].tolist()
refs = res['orig_summary'].tolist()

gens = [str(a) for a in gens]
refs = [str(b) for b in refs]
rouge = rouge.Rouge()
scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
print(scores)

with open('./greedy_results.json', 'w') as f:
    json.dump(scores, f)


print('Beam.')
res = pd.read_csv('basic_results_beam4_len75.csv')
gens = res['pred_summary'].tolist()
refs = res['orig_summary'].tolist()

gens = [str(a) for a in gens]
refs = [str(b) for b in refs]
scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
print(scores)

with open('./beam_results.json', 'w') as f:
    json.dump(scores, f)


print('Nucleus.')
res = pd.read_csv('basic_results_nucl.csv')
gens = res['pred_summary'].tolist()
refs = res['orig_summary'].tolist()

gens = [str(a) for a in gens]
refs = [str(b) for b in refs]
scores = rouge.get_scores(gens, refs, avg=True, ignore_empty=True)
print(scores)

with open('./nucleus_results.json', 'w') as f:
    json.dump(scores, f)

