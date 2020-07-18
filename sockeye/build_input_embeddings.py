import numpy as np
import json

src_embed_file = '/home/msobrevillac/Projects/phd/NLG/sockeye/pre-embeddings/embed-in-src.npy'
src_vocab_file = '/home/msobrevillac/Projects/phd/NLG/sockeye/pre-embeddings/vocab-in-src.json'

vocab = {}
vectors = []

with open('/home/msobrevillac/Projects/phd/Resources/Embeddings/glove/glove.6B.300d.txt', 'rb') as f:
    index = 0
    for line in f:
        fields = line.split()
        word = fields[0].decode('utf-8')
        vocab[word] = index
        index += 1
        vector = np.fromiter((float(x) for x in fields[1:]),
                             dtype=np.float)
        vectors.append(vector)

src_array = np.array(vectors)

np.save(src_embed_file, src_array)

with open(src_vocab_file, "w", encoding="utf-8") as out:
    json.dump(vocab, out, indent=4, ensure_ascii=False) 
