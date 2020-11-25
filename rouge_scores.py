from rouge import Rouge
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
r=17

df = pd.read_csv('prp.csv', sep = ',')
print('whole csv loaded')

train, test = train_test_split(df, train_size=120000, test_size=60000, shuffle=True, random_state=r)

print('train, test sets created')
# generated summaries are lead-n sentences where n is the number of sentences in the reference summary (i.e. the first n sentences of the original text joined back together

train['tokenized_text'] = train['Text'].apply(nltk.sent_tokenize)

print('tokenized')

train['lead_n'] = train.apply(lambda x: x.tokenized_text[:x.sum_len], axis=1)

print('leads found')

train['temp'] = train['lead_n'].apply(len)
train = train[train['temp'] == train['sum_len']]
train['joined'] = train['lead_n'].apply(lambda x: ''.join([str(a) for a in x]))
train.dropna(inplace=True)
print('done')
test['tokenized_text'] = test['Text'].apply(nltk.sent_tokenize)

print('tokenized')

test['lead_n'] = test.apply(lambda x: x.tokenized_text[:x.sum_len], axis=1)

print('leads found')

test['temp'] = test['lead_n'].apply(len)
test = test[test['temp'] == test['sum_len']]
test['joined'] = test['lead_n'].apply(lambda x: ''.join([str(a) for a in x]))
test.dropna(inplace=True)
print('done')

generated_train = train['joined'].tolist()
generated_test = test['joined'].tolist()

# reference summaries are the sentences in Sum column
reference_train = train['Sum'].tolist()
reference_test = test['Sum'].tolist()

gen_and_ref = zip(generated_train, reference_train)
gen_and_ref = [_ for _ in gen_and_ref if 500 > len(_[0]) > 25 and 500 > len(_[1]) > 25]
generated_train, reference_train = zip(*gen_and_ref)

new_train = pd.DataFrame(
        {'generated':generated_train, 'reference':reference_train
            })[:100000]

del generated_train, reference_train

gen_and_ref = zip(generated_test, reference_test)
gen_and_ref = [_ for _ in gen_and_ref if 500 > len(_[0]) > 25 and 500 > len(_[1]) > 25]
generated_test, reference_test = zip(*gen_and_ref)

new_test = pd.DataFrame(
        {'generated':generated_test, 'reference':reference_test
            })[:50000]

del generated_test, reference_test

new_train.to_csv('./data/train.csv', sep=',', index=False)
new_test.to_csv('./data/test.csv', sep=',', index=False)
'''
evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=2,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=False,
                           apply_best=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

scores = evaluator.get_scores(refs, generated)

def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print(prepare_results(results['p'], results['r'], results['f']))
'''

generated = new_test['generated'].tolist()
refs = new_test['reference'].tolist()
print(len(generated), len(refs))
rouge = Rouge()
scores = rouge.get_scores(generated, refs, avg=True, ignore_empty=True)
print(scores)

import json
with open('./results/lead_n.json', 'w') as handler:
    json.dump(scores, handler)
