from collections import Counter 
import pysnooper
import torch


@pysnooper.snoop()
def debug_test():
    a = 1 + 2
    c = 1 + 3

    b = a / a - a

    return b

if __name__ == '__main__':
    c = Counter()
    c.update('a')

    c.update(['a','c'])


    reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    index2word = reserved[:]

    bbb = {"a":1, "b": 2}

    print(bbb.get('c', "nuk"))

    debug_test()
    
