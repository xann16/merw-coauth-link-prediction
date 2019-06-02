from dataset import DataSet
import metrics as mtr
import kernel_methods as kern
import scipy.sparse.linalg as sla
import math
import json
import sys

KERNELS_BASE = ["CK", "NCK", "MECK", "NMECK", "DK", "NDK", "MEDK", "NMEDK",
                "RK", "NRK", "MERK", "NMERK", "MENK", "NNK", "NMENK"]
KERNELS_EXCLUDE = ["CK", "NCK", "MECK", "NMECK", "DK", "NDK", "MEDK", "NMEDK",
                   "RK", "NRK", "MERK", "NMERK", "NNK", "NMENK"]

CATEGORIES_BASE = ["gr-qc", "import"]
CATEGORIES_EXCLUDE = ["import"]

DSNAMES_BASE = ["eg1k_chr_prc", "std_gr-qc"]
DSNAMES_EXCLUDE = ["std_gr-qc"]


def get_kernel_names():
    result = []
    for kernel_name in KERNELS_BASE:
        if kernel_name not in KERNELS_EXCLUDE:
            result.append(kernel_name)
    return result


def get_categories():
    result = []
    for category in CATEGORIES_BASE:
        if category not in CATEGORIES_EXCLUDE:
            result.append(category)
    return result


def get_dataset_names():
    result = []
    for ds_name in DSNAMES_BASE:
        if ds_name not in DSNAMES_EXCLUDE:
            result.append(ds_name)
    return result


def calculate_kernel(A_csr, A_csc, l_max, v_max, type):
    if type == "CK":
        return kern.commute_time_kernel(kern.laplacian(A_csr), 3)
    elif type == "NCK":
        return kern.commute_time_kernel(
            kern.symmetric_normalized_laplacian(A_csr), 3)
    elif type == "MECK":
        return kern.commute_time_kernel(kern.mecl(A_csr, l_max, v_max), 3)
    elif type == "NMECK":
        return kern.commute_time_kernel(kern.mecl(A_csr, l_max, v_max,
                                                  type='sym'), 3)
    elif type == 'DK':
        return kern.heat_diffusion_kernel(kern.laplacian(A_csr))
    elif type == "NDK":
        return kern.heat_diffusion_kernel(
            kern.symmetric_normalized_laplacian(A_csr))
    elif type == "MEDK":
        return kern.heat_diffusion_kernel(kern.mecl(A_csr, l_max, v_max))
    elif type == "NMEDK":
        return kern.heat_diffusion_kernel(kern.mecl(A_csr, l_max, v_max,
                                                    type='sym'))
    elif type == 'RK':
        return kern.regularized_laplacian_kernel(kern.laplacian(A_csr))
    elif type == "NRK":
        return kern.regularized_laplacian_kernel(
            kern.symmetric_normalized_laplacian(A_csr))
    elif type == "MERK":
        return kern.regularized_laplacian_kernel(kern.mecl(A_csr, l_max,
                                                           v_max))
    elif type == "NMERK":
        return kern.regularized_laplacian_kernel(kern.mecl(A_csr, l_max, v_max,
                                                           type='sym'))
    elif type == "NNK":
        return kern.traditional_normalized_neumann_kernel(A_csr)
    elif type == "MENK":
        return kern.neumann_kernel(A_csr, l_max, v_max)
    elif type == "NMENK":
        return kern.normalized_neumann_kernel(A_csr, l_max, v_max)
    else:
        raise BaseException("Invalid kernel type")


def calculate_metrics(vx_count, training_edges, test_edges, kernel,
                      auc_samples=10000, precision_k=30):
    auc = mtr.auc(vx_count, training_edges, test_edges, kernel, auc_samples)
    prc = mtr.precision(vx_count, training_edges, test_edges, kernel,
                        precision_k)
    return (auc, prc)


def process_subset(base_dataset, index, kernels,
                   auc_samples=10000, precision_k=30):

    # get training and test data edges list, as well as training graph adj mx
    training_edges, test_edges = base_dataset.get_dataset(ds_index=index)
    A_csr = base_dataset.get_training_set(ds_index=index,
                                          mode="adjacency_matrix_csr")
    A_csc = base_dataset.get_training_set(ds_index=index,
                                          mode="adjacency_matrix_csc")

    # init map for storing results
    results = {}

    # calculate dominant eigenvalue and its corresponding eigenvector
    ls, vs = sla.eigsh(A_csr, 1, which='LA')
    l_max = ls[0]
    v_max = vs[:, 0]

    # process all kernels and obtain respective metrics
    for kernel_name in kernels:
        kernel = calculate_kernel(A_csr, A_csc, l_max, v_max, kernel_name)
        auc, prc = calculate_metrics(base_dataset.vx_count,
                                     training_edges,
                                     test_edges,
                                     kernel,
                                     auc_samples,
                                     precision_k)
        results[kernel_name] = (auc, prc)
        print("[{}:{}] {} - AUC: {:.2f}; PRC: {:.2f}".
              format(base_dataset.base_path, index, kernel_name, auc, prc))

    return results


def calculate_results_summary(kernel_names, results):
    ds_count = len(results) - 1
    for kernel_name in kernel_names:
        auc_min = 1.0
        auc_max = 0.0
        auc_sum = 0.0
        prc_min = 1.0
        prc_max = 0.0
        prc_sum = 0.0

        for i in range(1, len(results)):
            auc, prc = results[i][kernel_name]
            auc_min = min(auc, auc_min)
            prc_min = min(prc, prc_min)
            auc_max = max(auc, auc_max)
            prc_max = max(prc, prc_max)
            auc_sum += auc
            prc_sum += prc

        auc_avg = auc_sum / ds_count
        prc_avg = prc_sum / ds_count

        auc_ssq = 0.0
        prc_ssq = 0.0

        for i in range(1, len(results)):
            auc, prc = results[i][kernel_name]
            auc_ssq += (auc - auc_avg) * (auc - auc_avg)
            prc_ssq += (prc - prc_avg) * (prc - prc_avg)

        auc_ssq = auc_ssq / ds_count
        prc_ssq = prc_ssq / ds_count

        auc_sdv = math.sqrt(auc_ssq)
        prc_sdv = math.sqrt(prc_ssq)

        results[0][kernel_name] = {"auc": {"min": auc_min, "max": auc_max,
                                           "mean": auc_avg, "sd": auc_sdv},
                                   "prc": {"min": prc_min, "max": prc_max,
                                           "mean": prc_avg, "sd": prc_sdv}}


def write_results(filepath, results):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)


def process_dataset(category, dsname, kernel_names, output_prefix,
                    auc_samples=10000, precision_k=30):
    ds = DataSet('../datasets/', category, dsname)
    results = []
    results.append({})
    for index in range(1, ds.set_count + 1):
        result = process_subset(ds, index, kernel_names, auc_samples,
                                precision_k)
        results.append(result)
    calculate_results_summary(kernel_names, results)
    print("RESULTS:")
    print(results)
    print()
    write_results(output_prefix + category + "_" + dsname + ".json", results)


def run(output_prefix):
    kernel_names = get_kernel_names()

    for category in get_categories():
        for dsname in get_dataset_names():
            process_dataset(category, dsname, kernel_names, output_prefix)


###############################################################################


if __name__ == '__main__':
    output_prefix = ""
    if len(sys.argv) > 1:
        output_prefix = sys.argv[1] + "_"

    run(output_prefix)
