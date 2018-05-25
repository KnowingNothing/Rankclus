# -*- coding: utf-8 -*-

"""
Created on May 14, 2018
@author: zhengsz@pku.edu.cn
data pre-process reference: https://blog.csdn.net/frontend922/article/details/18552077
Last modify: May 24, 2018
"""

import xml.sax
import sys
import io
import time
import numpy as np
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')  # use UTF-8 encoding

log_file = open("log.txt", "a")  # simple log file

# these codes bellow are modified according to a blog from CSDN
class DBLPHandler(xml.sax.ContentHandler):
    """
    handler for DBLP, a xml dataset
    """
    def __init__(self):
        super(DBLPHandler, self).__init__()
        self._CurrentData = ""
        self._author_list = []
        self._author = ""
        self._journal = ""
        self._X = set()     # to record journals without redundancy
        self._Y = set()     # to record authors without redundancy
        self._Wxy = {}      # to record a link from author to journal and its number
        self._Wyy = {}      # to record a link between two author and its number

    def get_all(self):
        return self._X, self._Y, self._Wxy, self._Wyy

    # start of an element, we focus on author and journal of article
    def startElement(self, tag, attributes):
        self._CurrentData = tag
        if tag == "article":
            self._author_list = []      # there may be more than one author
        elif tag == "author":
            self._author = ""
        elif tag == "journal":
            self._journal = ""

    # end of an element
    def endElement(self, tag):
        if tag == "author":
            self._author_list.append(self._author)
            self._Y.add(self._author)
        elif tag == "journal":
            self._X.add(self._journal)
        elif tag == "article":
            for i, author in enumerate(self._author_list):
                # to us '<=' link a journal and an author, hoping no one use '<=' in his name
                key = self._journal + "<=" + author
                if key in self._Wxy:
                    self._Wxy[key] += 1
                else:
                    self._Wxy[key] = 1
                for next_author in self._author_list[i + 1:]:
                    # frozenset is a good choice in representing link between two authors
                    author_key = frozenset([author, next_author])
                    if author_key in self._Wyy:
                        self._Wyy[author_key] += 1
                    else:
                        if len(author_key) == 2:
                            self._Wyy[author_key] = 1

    # handle the characters
    def characters(self, content):
        if content == '\n':     # unfortunately, the original data has '\n'
            return
        if self._CurrentData == "author":
            self._author += content
        elif self._CurrentData == "journal":
            self._journal += content


"""
    end of codes modified from CSDN, bellow are my own creations
    three parts form my solution:
    --- Fetch data
    --- Computation
    --- Run the algorithm and be robust
"""


class FetchData(object):
    """
    create an object of FetchData with a path where the xml dataset lies
    and use 'fetch' to get the set of journals, authors, links from author to journal and between authors
    """
    def __init__(self, path):
        self._X = {}
        self._Y = {}
        self._Wxy = {}
        self._Wyy = {}
        self._path = path
        self._from_data()

    def _from_data(self):
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        handler = DBLPHandler()
        parser.setContentHandler(handler)
        print("********************************************************************************", file=log_file)
        print("A new log at ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file=log_file)
        print("Start parsing xml data of DBLP at path: {}...".format(self._path), file=log_file)
        log_file.flush()
        beg_time = time.time()
        parser.parse(self._path)
        end_time = time.time()
        print("Parse done! Use {} seconds in total.".format(end_time - beg_time), file=log_file)
        log_file.flush()
        self._X, self._Y, self._Wxy, self._Wyy = handler.get_all()

    def fetch(self):
        return self._X, self._Y, self._Wxy, self._Wyy


class Graph(object):
    """
    this class is used to calculate values of journals and authors under a certain partition
    create an object of Graph with these parameters:
                        --- setX:       a set of all journals
                        --- setSubX:    a set of journals in one partition
                        --- setY:       a set of all authors
                        --- Wxy:        a dict with <journal"<="author>:<number of this link>
                        --- Wyy:        a dict with <frozenset(author1, author2)>:<number of this link>
                        --- max_iter:   how many iterations expected
                        --- epsi:       expected accuracy
                        --- alpha:      weight of influence of link between journal and author to values of authors
    then use calculate to get these results:
                        --- X:          a dict with <journal>:<the value>
                        --- Y:          a dict with <author>:<the value>
                        --- rate:       final accuracy of iteration
    """
    def __init__(self, setX, setSubX, setY, Wxy, Wyy, max_iter=100, epsi=1e-3, alpha=0.7):
        self._setX = setX
        self._setSubX = setSubX
        self._setY = setY
        self._Wxy = Wxy
        self._Wyy = Wyy
        self._max_iter = max_iter
        self._epsi = epsi
        self._alpha = alpha

    def calculate(self):
        tmp_sum_X = 0.0
        rate = -1
        X, subX, Y = {}, {}, {}

        # to give an initial value of journals
        for journal in self._setX:
            X[journal] = 0.0
        for journal in self._setSubX:
            subX[journal] = 1.0     # we choose 1.0 not for a certain reason, just because this is a fine value
        for author in self._setY:
            Y[author] = 1.0

        tmp_X, tmp_Y = subX.copy(), Y.copy()
        # use iterations to solve a equation is a state-of-the-art method
        for i in range(self._max_iter):
            for arc, prop in self._Wxy.items():
                journal, author = arc.split("<=")   # remember our links contains "<="
                if journal in subX:
                    _ = prop * Y[author]
                    tmp_X[journal] += _
                    _ = prop * subX[journal] * self._alpha
                    tmp_Y[author] += _
            for arc, prop in self._Wyy.items():
                # be careful, there will be people with the same names, so the arc has only one element
                if len(arc) < 2:
                    print("Arc in Wyy has only one element, this is abnormal: ", str(arc), file=log_file)
                    log_file.flush()
                    continue
                else:
                    authors = []
                    for author in arc:
                        authors.append(author)
                    _ = prop * Y[authors[1]] * (1 - self._alpha)
                    tmp_Y[authors[0]] += _
                    _ = prop * Y[authors[0]] * (1 - self._alpha)
                    tmp_Y[authors[1]] += _

            sum_X, sum_Y = sum(tmp_X.values()), sum(tmp_Y.values())
            # we use this formula to determine the rate
            rate = abs(sum_X - tmp_sum_X) / sum_X
            if rate < self._epsi:
                break
            tmp_sum_X = sum_X

            # we pretend these values to be probabilities, so need to normalize
            for key in tmp_X:
                tmp_X[key] /= sum_X
            for key in tmp_Y:
                tmp_Y[key] /= sum_Y

            # continue the iteration
            subX, Y = tmp_X.copy(), tmp_Y.copy()

        # at last, we need to calculate all the values of journals, not only those in this partition
        for arc, prop in self._Wxy.items():
            journal, author = arc.split("<=")
            X[journal] += prop * Y[author]
        sum_X = sum(X.values())
        for key in X:
            X[key] /= sum_X

        return X, Y, rate


"""
    following are methods to calculate some important values in this algorithm
"""
def compute_pai(P_Xs, P_Xk):
    """
    this function computes the probability of a single journal to be in a certain partition
    which is a compressive matrix computation, we can do this because number of journals and partitions are small
    parameters:
        --- P_Xs:   ndarray of values of different journals under every partition
        --- P_Xk:   ndarray of probabilities of a partition itself
    returns:
        a ndarray of probabilities of a journal to be in a partition, or to say, a row is a vector called pai_k in
        the original paper
    """
    P_xi_Xk = []
    for item in P_Xs:
        P_xi_Xk.append(list(item.values()))
    P_xi_Xk = np.array(P_xi_Xk)
    raw_result = np.multiply(P_xi_Xk.T, P_Xk)
    sum_val = np.sum(raw_result, 1)
    return (raw_result.T / sum_val).T


def compute_P_Xk(Wxy, P_Xs, P_Ys, P_Xk, K):
    """
    this function gives ndarray of probabilities of every partition
    this counters for the EM step of original paper
    parameters:
        --- Wxy:    a dict with <journal"<="author>:<number of this link>
        --- P_Xs:   ndarray of values of different journals under every partition
        --- P_Ys:   ndarray of values of different authors under every partition
        --- P_Xk:   old ndarray of probabilities of every partition
        --- K:      number of partitions
    returns:
        new ndarray of probabilities of every partition
    """
    new_P_Xk = np.array([0.0] * K)
    for k in range(K):
        for arc, val in Wxy.items():
            journal, author = arc.split("<=")
            new_P_Xk[k] += val * P_Xs[k][journal] * P_Ys[k][author]
        new_P_Xk[k] *= P_Xk[k]
    sum_V = sum(new_P_Xk)
    return new_P_Xk / sum_V


def run(param_iter, param_K, param_epsi):
    """
    this function runs the algorithm in iterations
    given expected number of iterations: param_iter
          expected number of partitions: param_K
          expected accuracy:             param_epsi
    """
    # fetch data
    # f = FetchData("E:/DATASETS/DBLP/dblp.xml")
    f = FetchData("small_dblp.xml")     # to use a small dataset
    # journals in fX, authors in fY, paper records in fWxy, cooperation records in fWyy
    fX, fY, fWxy, fWyy = f.fetch()

    # fine tune parameters
    max_iter = max(param_iter, 1)
    epsi = min(param_epsi, 1e-1)
    K = max(param_K, 2)

    # ready to give an init partition
    print("Begin to run (expected partitions {})".format(K))
    sys.stdout.flush()
    if K > len(fX):
        print("Can't divide! K is larger than total number of journals", file=log_file)
        log_file.flush()
        exit(0)
    partitions = [[] for i in range(K)]
    k = 0
    rand_fX = list(fX)
    np.random.shuffle(rand_fX)      # introduce some random factors at the beginning
    for journal in rand_fX:
        partitions[k].append(journal)
        k += 1
        if k >= K:
            k = 0

    # begin to run algorithm
    p_Xk = np.array([1.0 / K] * K)
    last_dist = 0.0
    iter_rate = float("Inf")
    for i in range(max_iter):
        print("{}th iteration begins.".format(i + 1), file=log_file)
        log_file.flush()
        # sub-graphs vector
        p_Xs = []
        p_Ys = []
        for k in range(K):
            beg_time = time.time()
            sub_g = Graph(fX, partitions[k], fY, fWxy, fWyy, epsi=1e-4)
            X, Y, rate = sub_g.calculate()      # get values according to different partition
            print("     After a calculation on sub_graph {} and rate is {}.".format(k, rate), file=log_file)
            log_file.flush()
            p_Xs.append(X)
            p_Ys.append(Y)
            end_time = time.time()
            print("     **** Use {} seconds. ****".format(end_time - beg_time), file=log_file)
            log_file.flush()

        # update p_Xk
        beg_time = time.time()
        p_Xk = compute_P_Xk(fWxy, p_Xs, p_Ys, p_Xk, K)
        now_time = time.time()
        print("     After calculating p_Xk, use {} seconds.".format(now_time - beg_time), file=log_file)
        log_file.flush()
        # get probabilities of every journal in every partition
        beg_time = time.time()
        pais = compute_pai(p_Xs, p_Xk)
        xi_pai = dict(zip(list(fX), pais))
        now_time = time.time()
        print("     After calculating pais, use {} seconds.".format(now_time - beg_time), file=log_file)
        log_file.flush()
        # begin to generate new partitions
        beg_time = time.time()
        total_dist = 0.0
        centers = []
        for k in range(K):
            center = np.array([0.0] * K)
            for journal in partitions[k]:
                center += xi_pai[journal]
            center /= K
            centers.append(center)
        new_partitions = [[] for i in range(K)]
        for journal, pai in xi_pai.items():
            belong, dist = 0, float("Inf")
            for k in range(K):
                # we use cosine distance
                tmp_dist = 1 - np.sum(centers[k] * pai) / (
                            np.sqrt(np.sum(np.square(centers[k]))) * np.sqrt(np.sum(np.square(pai))))
                if tmp_dist < dist:
                    dist = tmp_dist
                    belong = k
            total_dist += dist
            new_partitions[belong].append(journal)
        partitions = new_partitions
        end_time = time.time()
        print("     After partition, use {} seconds.".format(end_time - beg_time), file=log_file)
        log_file.flush()
        for k, part in enumerate(partitions):
            # we may go into such condition where a partition is empty, report an error code 1
            if len(part) <= 0:
                print("Partition {} becomes empty, ready to restart.[at iteration {}]".format(k, i), file=log_file)
                log_file.flush()
                return 1, iter_rate
        # we can use total distance as a index, whatever, if it converges, this should change in a small range
        iter_rate = abs(total_dist - last_dist) / max(total_dist, last_dist)
        last_dist = total_dist
        if iter_rate < epsi:
            print("Iterations finished early than expected when rate is {} and epsilon is {}.".format(iter_rate, epsi), file=log_file)
            log_file.flush()
            break
        print("end of {}th iteration with iter-rate: {}.".format(i, iter_rate), file=log_file)
        log_file.flush()

    # output the results
    with open("partitions.txt", "w") as f:
        lines = "The partitions are as below [command to have {} partitions]:\n".format(K)
        f.write(lines)
        for i, part in enumerate(partitions):
            lines = str(i) + ": " + str(part) + "\n"
            try:
                f.write(lines)
            except UnicodeEncodeError:
                f.write("Fail to write this partition {} because UnicodeEncodeError.\n".format(i))

    return 0, iter_rate


if __name__ == '__main__':
    iter, K, epsi = 40, 7, 1e-4
    false_trail = [0 for i in range(K + 1)]
    while True:
        if K <= 1:
            print("No need to try more, current K=1, exiting...", file=log_file)
            log_file.flush()
            print("Fail to work! Please check log.txt for detail.")
            sys.stdout.flush()
            break
        code, rate = run(param_iter=iter, param_K=K, param_epsi=epsi)
        if code == 0:
            print("All done! Final rate is ", rate, file=log_file)
            log_file.flush()
            print("Done! The result is in partitions.txt.")
            sys.stdout.flush()
            break
        else:
            print("Try to restart..")
            sys.stdout.flush()
            false_trail[K] += 1
            # if we try 3 times but all failed, we should try a smaller partition number
            if K >= 2 and false_trail[K] >= 3:
                K -= 1
