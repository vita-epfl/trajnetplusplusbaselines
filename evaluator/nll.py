from scipy.stats import gaussian_kde
import numpy as np 

np.random.seed(10)

def nll(preds, gt, log_pdf_lower_bound=-20):
    ll = 0.0
    for timestep in range(1, 10):
        curr_gt = gt[timestep]

        scipy_kde = gaussian_kde(preds[timestep].T)

        # We need [0] because it's a (1,)-shaped numpy array.
        log_pdf = np.clip(scipy_kde.logpdf(curr_gt.T), a_min=log_pdf_lower_bound, a_max=None)[0]
        ll += log_pdf/9
    print(ll)

def test():
	pred_list = []
	for i in range(10):
		# print("i: ", i)
		mean = (i, 0)
		cov = [[0, 0], [0, i]] 
		x = np.random.multivariate_normal(mean, cov, 10000)
		# print(x.shape)
		pred_list.append(x)
	preds = np.array(pred_list)
	gt = np.array([[i, ] for i in range(10)])

	print(preds.shape)
	print(gt.shape)

	for j in range(2):
		gt = np.array([[i, -j*i] for i in range(10)])
		nll(preds, gt)
		gt = np.array([[i, j*i] for i in range(10)])
		nll(preds, gt)
test()