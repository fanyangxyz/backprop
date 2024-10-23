from collections import Counter


def get_stats(ids, counter=None):
    counter = counter if counter is not None else Counter()
    for pair in zip(ids, ids[1:]):
        counter[pair] += 1
    return counter


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i+1 < len(ids) and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class Tokenizer:

    def __init__(self):
        self.merges = {}
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')
        return vocab

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            if verbose:
                print(f'Merge {pair} into {idx}')
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf'))) 
            if pair not in self.merges:
                break # no more merges available
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def decode(self, ids):
        text_bytes = b"".join([self.vocab[idx] for idx in ids])
        text = text_bytes.decode('utf-8', errors="replace")
        return text
    

def main():
    tokenizer = Tokenizer()
    text = "\n".join([
        "The food was absolutely delicious! Great service.",
        "Delicious food and excellent service as always.",
        "The service could be better, but the food is delicious.",
        "Not the best service, but definitely delicious food.",
        "Great atmosphere and delicious meals.",
        "The restaurant has amazing atmosphere.",
        "Best restaurant in town, amazing food!",
        "This place has the best atmosphere.",
        "Such delicious dishes and great service!",
        "Amazing restaurant with excellent food.",
        "The food here is consistently excellent.",
        "Excellent dishes and wonderful atmosphere.",
        "The best restaurant experience ever!",
        "Great food, great service, great place!",
        "Restaurant has amazing dishes and service."
    ])
    print(text)
    
    sentence = "Hello world! How are you? My name is Fan. I'm from China."
    print(sentence)
    print(len(tokenizer.encode(sentence))) 
    
    vocab_size = 290
    tokenizer.train(text, vocab_size)
    print(len(tokenizer.encode(sentence))) 

if __name__ == '__main__':
    main()