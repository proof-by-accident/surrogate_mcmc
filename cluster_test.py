import jug
import numpy as np

@TaskGenerator
def decrypt(prefix, suffix_size):
    res = []
    for p in product(letters, repeat=suffix_size):
        text = decode(ciphertext, np.concatenate([prefix, p]))
        if isgood(text):
            passwd = "".join(map(chr, p))
            res.append((passwd, text))
    return res

@TaskGenerator
def join(partials):
    return list(chain(*partials))

fullresults = join([decrypt([let], 4) for let in letters])
