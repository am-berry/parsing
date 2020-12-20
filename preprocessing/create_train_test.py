import pandas as pd 


df=pd.read_csv('../data/prp.csv', sep = ',')
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

train_df = train[(train['joined'].apply(lambda x: 500 > len(x) > 25)) & (train['Sum'].apply(lambda x: 500 > len(x) > 25))]
test_df = test[(test['joined'].apply(lambda x: 500 > len(x) > 25)) & (test['Sum'].apply(lambda x: 500 > len(x) > 25))]

print(train_df.head())
print(test_df.head())

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

# want to save original text, reference summary and summary length

train_df.to_csv('../data/train.csv', sep=',', index=False)
test_df.to_csv('../data/test.csv', sep=',', index=False)

