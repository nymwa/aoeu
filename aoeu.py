import sys
import numpy as np
from collections import Counter
import argparse

def load_words(f):
	words = f.read().splitlines()
	words = list(set(words))
	words.sort()
	return words

def prepare_vocab(words):
	chars = [c for w in words for c in w]
	vlist = ['<s>', '</s>'] + [x for x, _ in Counter(chars).most_common()]
	vdict = {x:i for i, x in enumerate(vlist)}
	return vlist, vdict


class Model:
	def __init__(self, words):
		self.vlist, self.vdict = prepare_vocab(words)
		self.data = [[self.vdict[x] for x in w] for w in words]
		self.prepare()

	def count_ngram(self, n):
		tab = np.zeros([len(self.vlist)] * n)
		for xs in self.data:
			xs = [0] * max(1, n-1) + xs + [1] * max(1, n-1)
			for ys in zip(*[xs[i:] for i in range(n)]):
				tab[ys] += 1
		return tab

	def decode(self, xs):
		xs = [self.vlist[x] for x in xs if x >=2]
		return ''.join(xs)


class LinearInterpolationModel(Model):
	def __init__(self, words, l1=0.5, l2=0.5):
		self.l1 = l1
		self.l2 = l2
		super().__init__(words)

	def prepare(self):
		tab1 = self.count_ngram(1)
		tab2 = self.count_ngram(2) + 1
		prob1 = self.l1 * tab1 / tab1.sum() + (1 - self.l1) / len(self.vlist)
		prob2 = self.l2 * tab2 / tab2.sum(axis=1)[:, np.newaxis] + (1 - self.l2) * prob1
		self.prob = prob2

	def __call__(self, xs):
		return self.prob[xs[-1]]


class Noiser:
	def __init__(self, model, v=0.1):
		self.model = model
		self.v = v

	def decode(self, xs):
		return self.model.decode(xs)

	def __call__(self, x):
		x = self.model(x)
		x = x + np.random.normal(0, self.v, size=len(x))
		return x


class BeamDecoder:
	def __init__(self, model):
		self.model = model

	def cand(self, prob, xs, dist, width):
		indices = (-dist).argsort()[:width]
		values = dist[indices]
		return [(prob + np.log(v), xs + [i]) for v, i in zip(values, indices)]

	def calc_cands(self, beam, dists):
		cands = [
				tup
				for (prob, xs), dist in zip(beam, dists)
				if prob is not None
				for tup in self.cand(prob, xs, dist, len(beam))
			]
		cands.sort(reverse=True)
		return cands[:len(beam)]

	def __call__(self, width, max_len=100):
		beam = [(0, [0])] + [(None, [0])] * (width - 1)
		output = []
		for i in range(max_len):
			if len(beam) == 0:
				break
			dists = [self.model(xs) for prob, xs in beam]
			cands = self.calc_cands(beam, dists)
			beam = [(prob, xs) for prob, xs in cands if xs[-1] >= 2]
			output += [(prob, xs) for prob, xs in cands if xs[-1] < 2]
		return output


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--width', type=int, default=10)
	parser.add_argument('--max_len', type=int, default=100)
	parser.add_argument('--l1', type=float, default=0.8)
	parser.add_argument('--l2', type=float, default=0.8)
	args = parser.parse_args()

	words = load_words(sys.stdin)
	model = LinearInterpolationModel(words, l1=args.l1, l2=args.l2)
	model = Noiser(model)
	bd = BeamDecoder(model)
	for p, x in bd(args.width, args.max_len):
		print(p, '\t', model.decode(x))

