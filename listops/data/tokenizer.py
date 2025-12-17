
COT_TOKEN = '='
TOKEN_EOS = '<eos>'
TOKENS_OTHER = list('()=')
SEP=','
SHARED_TOKENS = ['<pad>', '<unk>', '<s>', '</s>', 
                COT_TOKEN, TOKEN_EOS] + TOKENS_OTHER
class Tokenizer:
    def __init__(self, vocab, sep=SEP):
        self.vocab = vocab
        self.sep = sep
        self.tok2index = {ch: i for i, ch in enumerate(vocab)}
    
    def encode(self, s, sep=None):
        if sep is None:
            sep = self.sep
        return [self.tok2index[c] for c in s.split(sep)]  # encoder: take a string, output a list of integers
    
    def decode_raw(self, l, sep=None):
        if sep is None:
            sep = self.sep
        return sep.join([self.vocab[i] for i in l])  # decoder: take a list of integers, output a string
    
    def decode(self, l, sep=None):
        out = self.decode_raw(l, sep)
        out = out.replace(f',{COT_TOKEN},', ' = ').replace(f',{TOKEN_EOS}', '\n').replace('\n,', '\n')
        out = out.replace(',(,', '(').replace(',)', ')').replace('(,', '(')
        return out
    
    def __call__(self, s):
        return self.encode(s)
    
    def decode_batch(self, batch):
        return [self.decode(t.tolist()) for t in batch]