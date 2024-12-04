import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc
from scipy.optimize import lsq_linear
from sklearn.cluster import KMeans


def clustering_model(in_sample_data, x_columns, no_of_clusters=15):
    matrix = in_sample_data[x_columns].values.T
    in_sample_data[x_columns].describe().to_csv('TrailingPE.csv')
    pdist = spc.distance.pdist(matrix)
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, no_of_clusters, criterion='maxclust')
    clusters = pd.Series(idx, index=x_columns, dtype='float64').sort_values()
    cluster_names = clusters.unique()
    cluster_data_init = pd.DataFrame()
    for cluster in cluster_names:
        cluster_alphas = clusters[clusters == cluster].index.values
        cluster_data_init['cluster_' + str(cluster)] = (in_sample_data[cluster_alphas]).mean(axis=1)
    init = cluster_data_init.values.T
    cluster_labels = KMeans(n_clusters=no_of_clusters, random_state=0, init=init,
                            n_init=1, verbose=True).fit(matrix).labels_
    clusters = pd.Series(cluster_labels, index=x_columns, dtype='float64').sort_values()
    cluster_names = clusters.unique()
    cluster_data = pd.DataFrame()
    for cluster in cluster_names:
        cluster_alphas = clusters[clusters == cluster].index.values
        cluster_data['cluster_' + str(cluster)] = (in_sample_data[cluster_alphas]).mean(axis=1)
    return cluster_data, clusters


def regress_clustered_data(cluster_data, in_sample_data, y_column, clusters, bounds):
    regression_model = lsq_linear(cluster_data, in_sample_data[y_column], bounds=bounds,
                                  verbose=1, method='bvls', tol=0.001)
    coefficients = pd.Series(regression_model['x'], index=cluster_data.columns, dtype='float64').fillna(0)
    print(coefficients)
    coefficients = coefficients / coefficients.abs().sum()
    alpha_weights = pd.Series(dtype='float64')
    cluster_names = clusters.unique()
    for cluster in cluster_names:
        cluster_alphas = clusters[clusters == cluster].index.values
        cluster_weight = coefficients['cluster_' + str(cluster)]
        cluster_alpha_weights = pd.Series(1 / len(cluster_alphas), index=cluster_alphas, dtype='float64')
        cluster_alpha_weights = cluster_alpha_weights * cluster_weight
        alpha_weights = pd.concat([alpha_weights, cluster_alpha_weights], axis=0)
    return alpha_weights


def regress_alphas(independent_variables, dependent_variable, bounds, model):
    regression_model = lsq_linear(independent_variables, dependent_variable, bounds=bounds,
                                  verbose=1, method='bvls', tol=0.001)
    alpha_weights = pd.Series(regression_model['x'], index=independent_variables.columns, dtype='float64').fillna(0)
    alpha_weights = alpha_weights / alpha_weights.abs().sum()
    alpha_weights = independent_regression_weights(alpha_weights, model)
    return alpha_weights


def independent_regression_weights(alpha_weights, model):
    alpha_df = pd.DataFrame()
    for alpha in alpha_weights.index:
        if abs(alpha_weights[alpha]) > 0.01:
            alpha_series = pd.Series(dtype='object')
            if '_conj' in alpha:
                multiplier = -1
            else:
                multiplier = 1
            alpha_series['pulse'] = alpha.replace('_conj', '')
            alpha_series['long_short'] = multiplier * alpha_weights[alpha]
            alpha_df = pd.concat([alpha_df, alpha_series.to_frame().T], axis=0)
    alpha_df['model'] = model
    alpha_df['model_type'] = 'independent'
    return alpha_df


def hybrid_regression_weights(alpha_weights, model):
    alpha_df = pd.DataFrame()
    for alpha in alpha_weights.index:
        if abs(alpha_weights[alpha]) > 0.01:
            alpha_series = pd.Series(dtype='object')
            alpha_series['pulse'] = alpha
            alpha_series['long_short'] = alpha_weights[alpha]
            alpha_df = alpha_df.append(alpha_series, ignore_index=True)
    alpha_df['model'] = model
    alpha_df['model_type'] = 'hybrid'
    return alpha_df


def independent_regression(turnover, in_sample_data, frequency='1D', no_of_clusters=15):
    turnover_buckets = np.linspace(0, 1, 11)
    combined_weights = pd.DataFrame()
    for i in range(0, len(turnover_buckets) - 1):
        turnover_alphas = turnover[(turnover >= turnover.quantile(q=turnover_buckets[i])) &
                                   (turnover < turnover.quantile(q=turnover_buckets[i + 1]))].index

        model = 'p_' + str(int(turnover_buckets[i] * 100)) + \
                '_' + str(int(turnover_buckets[i + 1] * 100)) + '_independent'
        if len(turnover_alphas) <= 15:
            no_of_clusters = min(int(len(turnover_alphas) / 3), 15)
        clustered_data, clusters = clustering_model(in_sample_data, turnover_alphas, no_of_clusters=no_of_clusters)
        alpha_weights = regress_clustered_data(clustered_data, in_sample_data, frequency, clusters, (-1, 1))
        weights_df = independent_regression_weights(alpha_weights, model)
        combined_weights = combined_weights.append(weights_df, ignore_index=True)
    return combined_weights


def slow_regression(turnover, fundamental_alphas, in_sample_data):
    slow_alphas = turnover[turnover < turnover.quantile(q=0.25)].index
    x_columns = np.unique(np.append(slow_alphas, fundamental_alphas))
    model = 'slow'
    clustered_data, clusters = clustering_model(in_sample_data, x_columns)
    alpha_weights = regress_clustered_data(clustered_data, in_sample_data, '1D', clusters, (-1, 1))
    return independent_regression_weights(alpha_weights, model)


def hybrid_regression(turnover, style_factors, in_sample_data, frequency='5D'):
    for alpha1 in turnover.index:
        for alpha2 in style_factors:
            if '_conj' in alpha1:
                multiplier = -1
            else:
                multiplier = 1
            in_sample_data[alpha1 + '|' + alpha2] = (multiplier * in_sample_data[alpha1] - in_sample_data[alpha2]) / 2
    turnover_buckets = np.linspace(0, 1, 11)
    combined_weights = pd.DataFrame()
    for i in range(0, len(turnover_buckets) - 1):
        turnover_alphas = turnover[(turnover >= turnover.quantile(q=turnover_buckets[i])) &
                                   (turnover < turnover.quantile(q=turnover_buckets[i + 1]))].index
        turnover_bucket_name = 'p_' + str(int(turnover_buckets[i] * 100)) + '_' + str(
            int(turnover_buckets[i + 1] * 100)) + '_hybrid'
        for style_factor in style_factors:
            x_columns = [pv_alpha + '|' + style_factor for pv_alpha in turnover_alphas]
            turnover_model_name = turnover_bucket_name
            style_model_name = style_factor.replace('_conj', '')
            clustered_data, clusters = clustering_model(in_sample_data, x_columns)
            alpha_weights = regress_clustered_data(clustered_data, in_sample_data, frequency, clusters, (0, 1))
            turnover_weights = hybrid_regression_weights(alpha_weights, turnover_model_name)
            style_weights = hybrid_regression_weights(alpha_weights, style_model_name)
            combined_weights = combined_weights.append(turnover_weights, ignore_index=True)
            combined_weights = combined_weights.append(style_weights, ignore_index=True)
    return combined_weights


def fundamental_regression(in_sample_data, bucket, alphas, bounds=(-1, 1)):
    model = bucket
    if len(alphas) <= 5:
        alpha_weights = pd.Series(1 / len(alphas), index=alphas)
    else:
        no_of_clusters = int(np.floor(len(alphas) / 3))
        clustered_data, clusters = clustering_model(in_sample_data, alphas, no_of_clusters=no_of_clusters)
        alpha_weights = regress_clustered_data(clustered_data, in_sample_data, '1D', clusters, bounds)
    weights_df = independent_regression_weights(alpha_weights, model)
    print(weights_df)
    return weights_df


def hybrid_monthly_regression(turnover, style_factors, in_sample_data, frequency='1M'):
    for alpha1 in turnover.index:
        for alpha2 in style_factors:
            in_sample_data[alpha1 + '|' + alpha2] = (in_sample_data[alpha1] + in_sample_data[alpha2]) / 2
    turnover_buckets = np.linspace(0, 1, 11)
    combined_weights = pd.DataFrame()
    for i in range(0, len(turnover_buckets) - 1):
        turnover_alphas = turnover[(turnover >= turnover.quantile(q=turnover_buckets[i])) &
                                   (turnover < turnover.quantile(q=turnover_buckets[i + 1]))].index
        turnover_bucket_name = 'p_' + str(int(turnover_buckets[i] * 100)) + '_' + str(
            int(turnover_buckets[i + 1] * 100)) + '_hybrid'
        for style_factor in style_factors:
            x_columns = [pv_alpha + '|' + style_factor for pv_alpha in turnover_alphas]
            turnover_model_name = turnover_bucket_name
            no_of_clusters = 15
            if len(turnover_alphas) <= 15:
                no_of_clusters = min(int(len(x_columns) / 3), 15)
            clustered_data, clusters = clustering_model(in_sample_data, x_columns, no_of_clusters=no_of_clusters)
            alpha_weights = regress_clustered_data(clustered_data, in_sample_data, frequency, clusters, (0, 1))
            turnover_weights = hybrid_regression_weights(alpha_weights, turnover_model_name)
            style_weights = hybrid_regression_weights(alpha_weights, style_factor)
            combined_weights = combined_weights.append(turnover_weights, ignore_index=True)
            combined_weights = combined_weights.append(style_weights, ignore_index=True)
    return combined_weights