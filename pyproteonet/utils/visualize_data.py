from matplotlib import pyplot as plt

def create_hist(feature_matrix, dataset):
	features = feature_matrix.flatten()
	features = features[features != -100]
	plt.hist(features, bins=50)
	plt.gca().set(title=feature_matrix, xlabel = 'Abundance', ylabel='Frequency')
	plt.tight_layout()
	plt.savefig(dataset + '_abundance.png')
	plt.close()
	return 1
