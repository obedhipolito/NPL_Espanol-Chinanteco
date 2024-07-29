import unicodedata
import re
import random

MAX_LENGTH = 10
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
#def unicodeToAscii(s):
#    return ''.join(
#        c for c in unicodedata.normalize('NFD', s)
#        if unicodedata.category(c) != 'Mn'
#    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?¿¡])", r" \1", s)
    s = re.sub(r"[^a-z0-9ñáéíóúàèìòùâêîôπēīįėäëïöü!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

 #esp_prefixes = (
#    "el ", "la ", "los ", "las ",
#    "un ", "una ", "unos ", "unas ",
#    "es ", "son ", "está ", "están "
#)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH #and \
        #p[0].startswith(esp_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    with open('esp-chi.txt', 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(f"{pair[0]}\t{pair[1]}\n")

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    print("Sample tokenization:")
    for i in range(5):
        print(pair[0], "->", [input_lang.word2index[word] for word in pair[0].split(' ')])
        print(pair[1], "->", [output_lang.word2index[word] for word in pair[1].split(' ')])
    
    return input_lang, output_lang, pairs

#input_lang, output_lang, pairs = prepareData('esp', 'chi', True)
#print(random.choice(pairs))
