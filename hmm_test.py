# test_bayes_classifier.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pickle
import torch
#import pytest

from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Exponential
from pomegranate.distributions import Normal


#from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


MIN_VALUE = 0
MAX_VALUE = None
VALID_VALUE = 1.2


def X():
	return [[[1.0],
	      [0.0],
	      [1.0],
	      [2.0],
	      [3.0]],
	     [[5.0],
	      [2.0],
	      [1.0],
	      [1.0],
	      [0.0]]]

def BAF():
	with open("/n/fs/ragr-research/projects/melody-hetSNPdetection/data/chr18_BAF.pickle","rb") as f: 
		B = pickle.load(f)
	f.close()

	return B
	

def model():
	starts = [0.2, 0.8]

	edges = [[1-3*10**-5, 3*10**-5],
	         [3*10**-5, 1-3*10**-5]]

#	d = [Normal([2.1],[[0.2]],covariance_type="diag"), Normal([4.5],[[0.2]],covariance_type="diag")]
	d = [Normal([0.2],[[0.2]],covariance_type="diag", min_cov=0.2), Normal([0.5],[[0.2]],covariance_type="diag", min_cov=0.2)]
	model = DenseHMM(distributions=d, edges=edges, starts=starts, 
		random_state=0)
	return model


mod = model()
#Xarr = numpy.array(X()).astype(float)
#Xarr = torch.tensor(Xarr)
#print(Xarr)
#mod.fit(Xarr)
#for d in mod.distributions:
#	print(d.means, d.covs)
import pdb; pdb.set_trace()
hidden_states = mod.predict(BAF())

# print("".join(map(str, list(hidden_states.numpy()[0]))))
