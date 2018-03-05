from collections import defaultdict
import sys
from math import log

startsym, stopsym = "<s>", "</s>"


def readfile(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        wordtags = [x.rsplit("/", 1) for x in line.split()]
        ws, ts = [e[0] for e in wordtags], [e[1] for e in wordtags]
        results.append((ws, ts))

    return results  # (word_seq, targ_seq) pair


# to get model -> phi(x,y)
def mle(filename):  # Max Likelihood Estimation of HMM
    twfreq = defaultdict(lambda: defaultdict(int))
    ttfreq = defaultdict(lambda: defaultdict(int))
    tagfreq = defaultdict(int)
    dictionary = defaultdict(set)

    for words, tags in readfile(filename):

        last = startsym

        tagfreq[last] += 1
        for word, tag in [(startsym, startsym)] + list(zip(words, tags)) + [(stopsym, stopsym)]:
            # if tag == "VBP": tag = "VB" # +1 smoothing
            twfreq[tag][word] += 1  # 'tag': {'word': +1, 'word': +1 ...}
            ttfreq[last][tag] += 1  # 'tag': {'tag': +1, 'tag': +1 ...}

            dictionary[word].add(tag)  # 'word': set(['tag', 'tag']), 'word': set(['tag', 'tag', 'tag']), ..."
            tagfreq[tag] += 1  # 'tag': number of tag, 'tag': number of tag, ...
            last = tag

    for key, values in dictionary.items():
        dictionary[key] = sorted(values)

    model = defaultdict(float)
    num_tags = len(tagfreq)
    for tag, freq in tagfreq.items():
        logfreq = log(freq)
        for word, f in twfreq[tag].items():
            model[tag, word] = log(f) - logfreq  # ('tag', 'word': frequency)
        logfreq2 = log(freq + num_tags)
        for t in tagfreq:  # all tags
            model[tag, t] = log(ttfreq[tag][t] + 1) - logfreq2  # +1 smoothing

    return dictionary, model  # dictionary -> "[word]": "[tag],[tag], ..."


# model -> ('tag', 'word': frequency) ...

# (decode): use model to get mytag, (test): then use mytag to caculate the error rate -> phi(x,z)
def decode(words, dictionary, model, MultiGrams):
    def backtrack(i, tag):
        if i == 0:
            return []
        return backtrack(i - 1, back[i][tag]) + [tag]

    words = [startsym] + words + [stopsym]

    best = defaultdict(lambda: defaultdict(lambda: float(
        "-inf")))  # defaultdict(<function <lambda> at 0x10d30c758>, {0: defaultdict(<function <lambda> at 0x10d30c8c0>, {'<s>': 1}), 1: defaultdict(<function <lambda> at 0x10d30c848>, {'VBZ': -4.644563114996229}), 2: defaultdict(<function <lambda> at 0x10d30c938>, {'DT': -9.014706232734948}), 3: defaultdict(<function <lambda> at 0x10d30c9b0>, {'NN': -11.388674150298066}), 4: defaultdict(<function <lambda> at 0x10d30ca28>, {'VB': -18.2413092729159, 'VBP': -19.405840345190537}), 5: defaultdict(<function <lambda> at 0x10d30caa0>, {'NN': -23.85640063894452}), 6: defaultdict(<function <lambda> at 0x10d30cb18>, {'</s>': -31.35326246611646})})
    best[0][startsym] = 1
    back = defaultdict(dict)

    # print " ".join("%s/%s" % wordtag for wordtag in zip(words,tags)[1:-1])
    for i, word in enumerate(words[1:], 1):
        for tag in dictionary[word]:

            # feature templates: (tag, tag[i-1]), (tag, word)
            if not MultiGrams:
                for prev, _ in best[i - 1].items():
                    score = best[i - 1][prev] + model[tag, prev] + model['tw', tag, word] + \
                            model['tt_1w', tag, prev, word] + model[tag, word, words[i - 1]] + \
                            model['tt_1w_1', tag, prev, words[i - 1]] + model[tag, prev, word, words[i - 1]]

                    if score > best[i][tag]:
                        best[i][tag] = score
                        back[i][tag] = prev

            # feature templates: (tag, tag[i-1]), (tag, word), and
            # (tag, tag[i-1], word), (tag, word, word[i-1]), (tag, tag[i-1], word[i-1]), (tag, tag[i-1], word, word[i-1]),
            # (tag, tag[i-2], tag[i-1]), (tag, tag[i-3], tag[i-2], tag[i-1])
            else:
                for prev, _ in best[i - 1].items():
                    score = best[i - 1][prev] + model[tag, prev] + model['tw', tag, word] + \
                            model['tt_1w', tag, prev, word] + model[tag, word, words[i - 1]] + \
                            model['tt_1w_1', tag, prev, words[i - 1]] + model[tag, prev, word, words[i - 1]]
                    if i >= 2:
                        score += model[tag, back[i - 1][prev], prev]

                    if i >= 3:
                        score += model[tag, back[i - 2][back[i - 1][prev]], back[i - 1][prev], prev]

                    if score > best[i][tag]:
                        best[i][tag] = score
                        back[i][tag] = prev

    mytags = backtrack(len(words) - 1, stopsym)[:-1]
    # print " ".join("%s/%s" % wordtag for wordtag in mywordtags)
    return mytags


# (decode): use model to get mytag, (test): then use mytag to caculate the error rate
def test(filename, dictionary, model, MultiGrams=False):
    errors = tot = 0
    for words, tags in readfile(filename):
        mytags = decode(words, dictionary, model, MultiGrams)
        errors += sum(t1 != t2 for (t1, t2) in zip(tags, mytags))
        tot += len(words)

    return errors / tot
