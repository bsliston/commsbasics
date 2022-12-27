import numpy as np
from scipy.signal import correlation_lags
import pdb

arr1 = np.ones((50))
arr2 = np.zeros((200))
arr2[50:100] = 1.0


corr = np.correlate(arr1, arr2, mode="full")
lags = correlation_lags(arr1.size, arr2.size, mode="full")
lag = lags[corr.argmax()]


pdb.set_trace()
