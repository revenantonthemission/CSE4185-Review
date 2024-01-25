import math
from collections import defaultdict, Counter
from math import log
import numpy as np

EPSILON = 1e-5

def smoothed_prob(arr, alpha=1):
    '''
    list of probabilities smoothed by Laplace smoothing
    input: arr (list or numpy.ndarray of integers which are counts of any elements)
           alpha (Laplace smoothing parameter. No smoothing if zero)
    output: list of smoothed probabilities

    E.g., smoothed_prob( arr=[0, 1, 3, 1, 0], alpha=1 ) -> [0.1, 0.2, 0.4, 0.2, 0.1]
          smoothed_prob( arr=[1, 2, 3, 4],    alpha=0 ) -> [0.1, 0.2, 0.3, 0.4]
    '''
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    _sum = arr.sum()
    if _sum:
        return ((arr + alpha) / (_sum + arr.size * alpha)).tolist()
    else:
        return ((arr + 1) / arr.size).tolist()

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # Baseline Tagger : 단어와 품사의 관계를 학습하여, 테스트 데이터에 등장하는 단어의 품사를 예측한다. 모든 단어는 독립적이며, 다른 단어의 영향을 받지 않는다.
    # 학습 데이터에서 단어와 품사의 관계를 학습한다.

    # baseline_model = {word: tag} | 학습 데이터 train을 통해 구한 단어와 품사의 관계를 저장한다.
    baseline_model = {}

    # tag_freq = {tag: freq} | 학습 데이터 train을 통해 구한 품사의 빈도를 저장한다.
    tag_freq = Counter()

    # word_tag_freq = {word: {tag: freq}} | 학습 데이터 train을 통해 구한 단어와 품사의 관계를 저장한다. train을 통해 학습한 주어진 단어 word에 대한 품사 tag의 빈도를 저장한다.
    word_tag_freq = defaultdict(Counter)
    for sentence in train:
        for word, tag in sentence:
            tag_freq[tag] += 1
            word_tag_freq[word][tag] += 1

    # baseline_model을 구한다. 이때 학습 데이터에서 각 단어에 대해 가장 빈도가 높은 품사를 선택한다.
    for word, tag_freq in word_tag_freq.items():
        # most_common(n): [(elem1, freq1), (elem2, freq2), ...] | 가장 빈도가 높은 n개의 원소를 반환한다.
        baseline_model[word] = tag_freq.most_common(1)[0][0]
    
    # 테스트 데이터 test를 예측한다. 테스트 데이터에 등장하는 단어가 학습 데이터에 등장하지 않는 경우, 전체 품사 중에서 가장 빈도가 높은 품사를 선택한다.
    predicted = []
    for sentence in test:
        sentence_tag = []
        for word in sentence:
            if word in baseline_model:
                sentence_tag.append((word, baseline_model[word]))
            else:
                sentence_tag.append((word, tag_freq.most_common(1)[0][0]))
        predicted.append(sentence_tag)

    return predicted

def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # 전체 데이터에서 단어가 등장하는 빈도
    word_freq = Counter()
    # 전체 데이터에서 품사가 등장하는 빈도
    tag_freq = Counter()
    # 품사의 확률 분포 | tag_prob[tag] = P(tag)
    tag_prob = {}

    # 전체 데이터에서 단어와 품사의 관계가 등장하는 빈도
    tag_word_freq = defaultdict(Counter)
    # 전체 데이터에서 품사에 따른 단어의 확률 분포 | tag_word_prob[tag][word] = P(word|tag)
    tag_word_prob = defaultdict(dict)
    # 전체 데이터에서 연이어 등장하는 품사들의 빈도
    tag_tag_freq = defaultdict(Counter)
    # 전체 데이터에서 연이어 등장하는 품사들의 확률 분포 | tag_tag_prob[tag1][tag2] = P(tag2|tag1)
    tag_tag_prob = defaultdict(dict)

    # 모든 데이터는 START에서 시작해서 END로 끝나기 때문에, START와 END는 계산에서 제외한다. <= sentence[1:-1]
    for sentence in train:
        for word, tag in sentence[1:-1]:
            word_freq[word] += 1
            tag_freq[tag] += 1
            tag_word_freq[tag][word] += 1
        for i in range(1, len(sentence) - 1):
            tag_tag_freq[sentence[i][1]][sentence[i+1][1]] += 1

    # Laplace Smoothing을 적용하여 전체 데이터에서 품사의 확률 분포를 구한다. 보정치는 EPSILON으로 설정한다.
    tag_smooth = smoothed_prob(list(tag_freq.values()), alpha=EPSILON)
    tag_prob = dict(zip(tag_freq.keys(), np.log(tag_smooth)))
    
    # train에 없는 단어가 test에 등장할 경우, 해당 단어의 빈도를 0으로 설정한다.
    for sentence in test:
        for word in sentence[1:-1]:
            if word not in word_freq:
                word_freq[word] = 0

    # 모든 경우에 대해 P(tag(t+1)|tag(t))와 P(tag(t)|word(t))를 구한다. 학습된 데이터에 없는 경우, 해당 케이스의 빈도를 0으로 설정한다.
    for tag in tag_freq.keys():
        for word in word_freq.keys():
            tag_word_prob[tag][word] = tag_word_freq[tag].get(word, 0)
        for next_tag in tag_freq.keys():
            tag_tag_prob[tag][next_tag] = tag_tag_freq[tag].get(next_tag, 0)

    # Laplace Smoothing을 적용하여 P(tag(t+1)|tag(t))와 P(tag(t)|word(t))를 구한다. 보정치는 EPSILON으로 설정한다.
    for tag in tag_freq.keys():
        tag_word_smooth = smoothed_prob(list(tag_word_prob[tag].values()), alpha=EPSILON)
        # 작은 확률의 연속적인 곱셈으로 인해 데이터가 0이 되어버릴 수 있으므로, log를 취한다.
        tag_word_prob[tag] = dict(zip(tag_word_prob[tag].keys(), np.log(tag_word_smooth)))

        tag_tag_smooth = smoothed_prob(list(tag_tag_prob[tag].values()), alpha=EPSILON)
        # 작은 확률의 연속적인 곱셈으로 인해 데이터가 0이 되어버릴 수 있으므로, log를 취한다.
        tag_tag_prob[tag] = dict(zip(tag_tag_prob[tag].keys(), np.log(tag_tag_smooth)))

    # 학습 데이터에 단 한 번만 등장한 단어의 품사가 가지는 분포를 구한다.
    one_time_word_tag_freq = Counter()
    for sentence in train:
        for word, tag in sentence[1:-1]:
            if word_freq[word] == 1:
                one_time_word_tag_freq[tag] += 1
        # 전체 품사 중에서 한 번만 등장한 단어의 품사가 아닌 모든 품사의 빈도를 0으로 설정한다.
        for tag in tag_freq.keys():
            if tag not in one_time_word_tag_freq:
                one_time_word_tag_freq[tag] = 0
    
    # Laplace Smoothing을 적용하여 학습 데이터에 단 한 번만 등장한 단어의 품사가 가지는 분포를 구한다. 보정치는 EPSILON으로 설정한다.
    one_time_word_tag_smooth = dict(zip(one_time_word_tag_freq.keys(), np.log(smoothed_prob(list(one_time_word_tag_freq.values()), alpha=EPSILON))))

    # 테스트 데이터 test에 등장하는 단어들의 품사를 예측한다.
    predicted = []
    for sentence in test:
        # sentence_tag = [(word1, tag1), (word2, tag2), ...] | 테스트 데이터 test의 한 문장인 sentence에 등장하는 단어들의 품사를 저장한다.
        sentence_tag = []

        # Initialize Viterbi and Backpointer
        viterbi = defaultdict(dict)
        backpointer = defaultdict(dict)
        for tag in tag_prob.keys():
            # viterbi[tag][t] = P(tag(t), word(t)) | 테스트 데이터 test의 한 문장인 sentence의 t번째 단어의 품사가 tag일 확률을 저장한다.
            viterbi[tag][0] = tag_prob[tag] + tag_word_prob[tag][sentence[1]]

            # 학습 데이터에 단 한 번만 등장한 단어의 품사가 가지는 분포를 추가로 적용하여, 학습된 데이터에 없는 단어의 품사를 더욱 정확하게 예측한다.
            if word_freq[sentence[1]] == 0:
                viterbi[tag][0] += one_time_word_tag_smooth[tag]
                
            # backpointer[tag][t] = tag | 테스트 데이터 test의 한 문장인 sentence의 t번째 단어에 대하여 예측된 품사가 tag일 때, t-1번째 단어의 품사를 저장한다.
            # t = 0일 때, t-1번째 단어가 없으므로 None으로 설정한다.
            backpointer[tag][0] = None

        # Run Viterbi for t > 0
        # START (0) 와 END (len(sentence)-1) 를 제외한 모든 단어에 대해 품사를 예측한다.
        for t in range(1, len(sentence) - 2):
            for tag in tag_prob.keys():
                max_prob = -math.inf
                max_tag = None
                for prev_tag in tag_prob.keys():
                    prob = viterbi[prev_tag][t-1] + tag_tag_prob[prev_tag][tag]
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = prev_tag
                viterbi[tag][t] = max_prob + tag_word_prob[tag][sentence[t+1]]

                # 학습 데이터에 단 한 번만 등장한 단어의 품사가 가지는 분포를 추가로 적용하여, 학습된 데이터에 없는 단어의 품사를 더욱 정확하게 예측한다.
                if word_freq[sentence[t+1]] == 0:
                    viterbi[tag][t] += one_time_word_tag_smooth[tag]

                # 작전에 등장한 단어의 품사로 가장 유력한 후보를 backpointer에 저장한다.
                backpointer[tag][t] = max_tag
                
        # Backtrace
        max_prob = -math.inf
        max_tag = None
        # 문장의 마지막 단어의 가장 유력한 품사를 찾는다.
        for tag in tag_prob.keys():
            prob = viterbi[tag][len(sentence) - 3]
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        sentence_tag.append((sentence[-2], max_tag))

        # backtrace를 이용하여 바로 앞에 있는 단어의 품사로 가장 유력한 후보를 찾는다.
        for t in range(len(sentence) - 3, 0, -1):
            max_tag = backpointer[max_tag][t]
            sentence_tag.append((sentence[t], max_tag))

        # 역순으로 예측했기 때문에 다시 순서를 뒤집고, 조건에 맞게 START와 END를 추가한다.
        sentence_tag.reverse()
        predicted.append([("START", "START")] + sentence_tag + [("END", "END")])
        
    return predicted
    