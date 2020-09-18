# %%
import re, collections, itertools

def get_stats(corpus):
    pairs = collections.defaultdict(int)
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_corpus(pair, c_in):
    c_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in c_in:
        w_out = p.sub(''.join(pair), word)
        c_out[w_out] = c_in[word]
    return c_out

corpus = {
    'l o w </w>': 5, 
    'l o w e r </w>': 2, 
    'n e w e s t </w>': 6, 
    'w i d e s t </w>': 3
}
vocab = [e.split()[1:-1] for e in corpus.keys()]
vocab = list(set(itertools.chain.from_iterable(vocab)))

num_merges = 1000
for i in range(num_merges):
    pairs = get_stats(corpus)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab.append(re.escape(''.join(best)))
    corpus = merge_corpus(best, corpus)

# vocab = ['##'+e.rstrip('</w>') \
#          if (e.endswith('</w>') and not e.startswith('<w>')) \
#          else e.lstrip('<w>').rstrip('</w>') for e in vocab]
vocab = sorted(vocab, key=lambda x: len(x), reverse=True)
print(f"Vocabulary: \n{vocab}")


# %%
ddict = collections.defaultdict(int)

ddict[1,2] = 1
ddict[2,3] = '2'
ddict['1','3'] = 2

print(ddict)

# %%
# https://cloud.tencent.com/developer/ask/203218
t = '1 2 3'

p = re.compile(r'(?<!\S)\d(?!\S)')
print(p.findall(t))

p = re.compile(r'(?<=\S)\d(?=\S)')
print(p.findall(t))

# %%
