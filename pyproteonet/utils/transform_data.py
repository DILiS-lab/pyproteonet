import logging

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger('transform_data')

def scale_data(features_matrix, replace_nan_with, missing_data_symbol = 'NA', logarithmize = False, normalize = True):
        logger.info('Before replacement', features_matrix)
        features_matrix[features_matrix == missing_data_symbol] = np.nan
        features_matrix = np.array(features_matrix, dtype=np.float32)
        logger.info('After replacement', features_matrix)
        # features_matrix = np.power(2, features_matrix)
        # print('After raising to power', features_matrix)

        if logarithmize:
                logger.info("Performing logarithmization")
                features_matrix = np.log(features_matrix + 1e-8)

        if normalize:
                # dividend = features_matrix - np.nanmin(features_matrix, axis=0)
                # divisor = np.nanmax(features_matrix, axis=0) - np.nanmin(features_matrix, axis=0)
                # scaled_features_matrix = np.true_divide(dividend, divisor, where=(features_matrix != np.nan)) #0-1 scaling

                dividend = features_matrix - np.nanmean(features_matrix, axis=0)
                divisor = np.nanstd(features_matrix, axis=0)
                scaled_features_matrix = np.true_divide(dividend, divisor, where=(features_matrix != np.nan)) #mean std normalization
                logger.info('After scaling', scaled_features_matrix)

                scaled_features_matrix = np.nan_to_num(scaled_features_matrix, nan= replace_nan_with)
                logger.info('After replacement with -10', scaled_features_matrix)
                return scaled_features_matrix
        elif normalize == False:
                features_matrix = np.nan_to_num(features_matrix, nan = replace_nan_with)
                logger.info('After replacement with -10', features_matrix)
                return features_matrix

def discretize_data(features_matrix, molecule_id_list, file_name, num_bins = 10, replace_nan_with = -10,
                    write_debug_artifacts = True):
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
        discretized_features_matrix = []
        
        # for sample_features in features_matrix.T:
        #        discretized_features_matrix.append(discretizer.fit_transform(sample_features))
        # discretized_features_matrix = np.array(discretized_features_matrix).T
        
        print("Discretizing file", file_name)
        bin_edges = []

        for molecule_features, molecule_id in zip(features_matrix, molecule_id_list):
                # print(molecule_id, 'molecule_features', np.unique(molecule_features))
                molecule_features_discretized = molecule_features.copy()
                molecule_features_new = np.delete(molecule_features, np.where(molecule_features == replace_nan_with)[0])

                if len(molecule_features_new)>= 1:
                        if len(molecule_features_new) == 1: #only one non Nan element, so discretization is not possible.
                                molecule_features_new[0] = replace_nan_with
                                molecule_features_new = molecule_features_new.reshape(-1, 1)
                        else:
                                molecule_features_new = molecule_features_new.reshape(-1, 1)
                                molecule_features_new = discretizer.fit_transform(molecule_features_new).squeeze()

                        # print(molecule_id, np.where(molecule_features != replace_nan_with)[0], molecule_features_new)

                        for i, abundance_val in zip(np.where(molecule_features != replace_nan_with)[0], molecule_features_new):
                                molecule_features_discretized[i] = abundance_val

                # print(molecule_id, 'molecule_features_discretized', np.unique(molecule_features_discretized))
                if hasattr(discretizer, 'bin_edges_') and discretizer.bin_edges_[0].shape[0] == 11 : #and discretizer.bin_edges_.size >=10:
                        bin_edges.append(discretizer.bin_edges_[0])
                        # print('1', np.array(discretizer.bin_edges_[0]).shape, np.array(bin_edges).shape)
                else:
                        bin_edges.extend([np.zeros(11)])
                        # print('2')
                discretized_features_matrix.append(molecule_features_discretized)

        print('******** ', np.array(molecule_id_list).shape,  '******** ')
        print('******** ', np.array(bin_edges).shape,  '******** ')

        if write_debug_artifacts:
                with open(file_name + '.csv', 'w') as f:
                        np.savetxt(f, np.c_[molecule_id_list, bin_edges], delimiter=',', fmt='%s')

        discretized_features_matrix = np.array(discretized_features_matrix)
        return discretized_features_matrix
