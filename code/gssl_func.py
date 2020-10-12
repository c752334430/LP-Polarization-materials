from label_prop import *
import learn_graph as lg
import pickle as pkl
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import cvxpy as cp
import numpy as np
import os, random
from random import sample
from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

label_percents = [0.3, 0.5]
epsilon = 10
lam = 1
USE_CORA = False
unknown_label = 0


def edit_biased_assimilation(adj_mat, labels, train_index, test_index, eta=1, max_it=1000, b=2, reg=True, tol=0.01,
                             result_graph=False):
    temp_label = np.array(labels)
    temp_label = (temp_label + 1) / 2
    self_weight = np.array([0 if i in train_index else 1 for i in range(labels.shape[0])])
    previous_x = np.array([temp_label[i] if i in train_index else 0.5 for i in range(labels.shape[0])])
    w = normalized(adj_mat)
    it = 0
    while it < max_it:
        s = np.matmul(previous_x, w)
        s[np.where(s == 0)] += 0.0001
        s[np.where(s == 1)] -= 0.0001
        cur_x = []
        for i in range(labels.shape[0]):
            nom = self_weight[i] * previous_x[i] + previous_x[i] ** b * s[i]
            deno = self_weight[i] + previous_x[i] ** b * s[i] + (1 - previous_x[i]) ** b * (1 - s[i])
            cur_x.append(nom / deno)
        cur_x = np.array(cur_x)
        y_pred = (cur_x - 0.5) * 2
        w = update_mat(w, y_pred, eta)
        w = normalized(w)
        if np.abs(previous_x[test_index] - cur_x[test_index]).sum() < tol:
            break
        previous_x = np.array(cur_x)
        it += 1
    y_pred = (cur_x - 0.5) * 2
    cla_acc, reg_err = result_metrics(labels, y_pred, test_index, reg)
    if result_graph:
        return cla_acc, reg_err, y_pred, it, w
    return cla_acc, reg_err, y_pred, it


def label_prop(w_origin, labels, train_index, test_index, max_it=1000, reg=True, tol=0.01):
    y_partial = np.array([labels[i] if i in train_index else 0 for i in range(labels.shape[0])])
    y_pred = np.array(y_partial)
    a = np.zeros(w_origin.shape)
    row, col = np.diag_indices(a.shape[0])
    a[row, col] = np.array(np.sum(w_origin, axis=0))
    D_inv = np.linalg.inv(a)
    random_walk_mat_numbers = np.matmul(D_inv, w_origin)
    y_prev = np.array(y_pred)
    it = 0
    while it < max_it:
        y_pred = np.matmul(random_walk_mat_numbers, y_pred)
        y_pred[train_index] = labels[train_index]
        if np.abs(y_prev[test_index] - y_pred[test_index]).sum() < tol:
            break
        it += 1
        y_prev = np.array(y_pred)
    cla_acc, reg_err = result_metrics(labels, y_pred, test_index, reg)
    #     print('Final accuracy:', cla_acc, 'regression error:', reg_err)
    return cla_acc, reg_err, y_pred, it


def iterate_hebbian(w_origin, labels, train_index, test_index, max_it=1000, eta=1, reg=True, tol=0.01,
                    result_graph=False):
    y_partial = np.array([labels[i] if i in train_index else 0 for i in range(labels.shape[0])])
    previous_y = y_partial
    w = np.array(w_origin)
    w = normalized(w)
    y_pred = np.matmul(previous_y, w).reshape(-1)
    it = 1
    while it < max_it:
        previous_y = y_pred
        previous_y[train_index] = labels[train_index]
        w = update_mat(w, y_pred, eta)
        w = normalized(w)
        y_pred = np.matmul(previous_y, w).reshape(-1)
        y_pred[train_index] = labels[train_index]
        if np.abs(previous_y[test_index] - y_pred[test_index]).sum() < tol:
            break
        it += 1
    cla_acc, reg_err = result_metrics(labels, y_pred, test_index, reg)
    #     print('Final accuracy:', cla_acc, 'regression error:', reg_err)
    if result_graph:
        return cla_acc, reg_err, y_pred, it, w
    return cla_acc, reg_err, y_pred, it


def iterate_optimize(feat_mat, labels, train_index, test_index, max_it=5, lam=1, alpha=1, beta=1,
                     low_bound=-1, up_bound=1, reg=True):
    w_s = []
    y_s = []
    y_partial = np.array([labels[i] if i in train_index else 0 for i in range(labels.shape[0])])
    previous_y = y_partial
    w = opt_L_given_Y(feat_mat, alpha=alpha, beta=beta)
    for i in range(w.shape[0]):
        w[i][np.where(w[i] < 0.0001)[0]] = 0
    w_s.append(w)
    y_pred = opt_Y_given_L(w, y_partial, low_bound, up_bound)
    y_s.append(y_pred)
    it = 1
    while it < max_it and np.linalg.norm(y_pred - previous_y) >= epsilon:
        w = opt_L_given_Y(feat_mat, lam * y_pred, alpha=alpha, beta=beta)
        for i in range(w.shape[0]):
            w[i][np.where(w[i] < 0.0001)[0]] = 0
        w_s.append(w)
        y_pred = opt_Y_given_L(w, y_partial, low_bound, up_bound)
        y_s.append(y_pred)
        cla_acc, reg_err = result_metrics(labels, y_pred, test_index, reg, printing=False)
        print('at iteration', it, ', current accuracy:', cla_acc, 'regression error:', reg_err)
        it += 1
    return cla_acc, reg_err, y_pred


def cgssl(x, labels, y_part_sklearn, w, up_bound, low_bound, ini_w=None, alpha=1, beta=1, epsilon=10):
    previous_y = y_part_sklearn
    cur_y = opt_Y_given_L(w, y_part_sklearn, low_bound, up_bound)
    acc = accuracy(cur_y, labels, y_part_sklearn)
    print('current accuracy:', acc)

    while np.linalg.norm(previous_y - cur_y) >= epsilon:
        w = opt_L_given_Y(x, cur_y, ini_w, alpha, beta)
        for i in range(w.shape[0]):
            w[i][np.where(w[i] < 0.0001)[0]] = 0
        previous_y = cur_y
        cur_y = opt_Y_given_L(w, y_part_sklearn, low_bound, up_bound)
        print('current accuracy:', accuracy(cur_y, labels, y_part_sklearn))
    # accuracy(cur_y, labels, y_part_sklearn, True)
    return cur_y


def johnsen(w, labels, train_index, test_index, max_it=1000, reg=True, tol=0.01):
    previous_y = np.zeros(labels.shape)
    previous_y[train_index] = labels[train_index]
    w_johnson = np.array(w)
    w_johnson, s_toadd = normalize_johnson(w_johnson, previous_y)
    it = 0
    while it < max_it:
        y_pred = np.matmul(previous_y, w_johnson).reshape(-1) + s_toadd
        if np.abs(previous_y[test_index] - y_pred[test_index]).sum() < tol:
            break
        it += 1
        previous_y = y_pred
    cla_acc, reg_err = result_metrics(labels, y_pred, test_index, reg)
    return cla_acc, reg_err, y_pred, it


def biased_assimilation(adj_mat, labels, train_index, test_index, max_it=1000, b=2, reg=True, tol=0.01):
    temp_label = np.array(labels)
    temp_label = (temp_label + 1) / 2
    self_weight = np.array([0 if i in train_index else 1 for i in range(labels.shape[0])])
    previous_x = np.array([temp_label[i] if i in train_index else 0.5 for i in range(labels.shape[0])])

    it = 0
    while it < max_it:
        s = np.matmul(previous_x, normalized(adj_mat))
        s[np.where(s == 0)] += 0.0001
        s[np.where(s == 1)] -= 0.0001
        cur_x = []
        for i in range(labels.shape[0]):
            nom = self_weight[i] * previous_x[i] + previous_x[i] ** b * s[i]
            deno = self_weight[i] + previous_x[i] ** b * s[i] + (1 - previous_x[i]) ** b * (1 - s[i])
            cur_x.append(nom / deno)
        cur_x = np.array(cur_x)
        if np.abs(previous_x[test_index] - cur_x[test_index]).sum() < tol:
            break
        previous_x = np.array(cur_x)
        it += 1
    y_pred = (cur_x - 0.5) * 2
    cla_acc, reg_err = result_metrics(labels, y_pred, test_index, reg)
    return cla_acc, reg_err, y_pred, it


def spielman_voltage(labels, train_index, test_index, dirt, it, reg=True):
    write_bound(labels, train_index, dirt, it)
    it_str = str(it)
    cmd = 'java -cp YINSlex/out/YINSlex_jar/YINSlex.jar CompLexMinimizer dataset/' + dirt + \
          '/edgelist.txt dataset/' + dirt + '/bound' + it_str + '.txt dataset/' + dirt + '/voltages' + it_str + '.txt'
    os.system(cmd)
    result_volt = pd.read_csv('dataset/' + dirt + '/voltages' + str(it) + '.txt', delimiter=' ', names=[1],
                              header=None).values
    cla_acc, reg_err = result_metrics(labels, result_volt, test_index, reg)
    return cla_acc, reg_err, result_volt


def matrix_from_weights(w, size, laplacian=True):
    L = np.zeros((size, size))
    count = 0
    for i in range(0, size):
        for j in range(i + 1, size):
            if w[i, j] < 0:
                w[i, j] = 0
            L[i, j] = -w[i, j] if laplacian else w[i, j]
            L[j, i] = -w[i, j] if laplacian else w[i, j]
            count += 1
        L[i, i] = -sum(L[:, i]) if laplacian else 0
    # print(L)
    return L


def opt_Y_given_L(w, y, low_bound, up_bound):
    print('Optimizing labels based on graph')
    size = y.shape[0]
    Y = cp.Variable(size)
    L = matrix_from_weights(w, size)

    objective = cp.Minimize(cp.quad_form(Y, L))
    constraints = []
    for i in range(size):
        if y[i] != unknown_label:
            constraints.append(Y[i] == y[i])
        else:
            constraints.append(Y[i] >= low_bound)
            constraints.append(Y[i] <= up_bound)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("\nThe optimal value is", prob.value)
    # print("A solution y is")
    # print(Y.value)

    return Y.value


def opt_L_given_Y(x, Y=None, original_w=None, alpha=1, beta=1, maxit=1000):
    print('Constructing graph from features')
    if Y is not None:
        x_prime = np.append(x, lam * Y.reshape((x.shape[0], -1)), 1)
    else:
        x_prime = x
    if original_w is None:
        W, problem = lg.log_degree_barrier(x_prime, dist_type='sqeuclidean', alpha=alpha, beta=beta, retall=True,
                                           verbosity='LOW', maxit=maxit)
    else:
        W, problem = lg.log_degree_barrier_with_w0(x_prime, dist_type='sqeuclidean', alpha=alpha, beta=beta,
                                                   retall=True,
                                                   verbosity='LOW', original_w=original_w, maxit=maxit)
    # print(x.shape, "!!!!")
    return W


# cora or citeseer
def cite_dataset(dataset_name, label_mapping=None, is_connected=False):
    data_dir = os.path.expanduser("/Users/tony/PycharmProjects/cgssl/dataset")
    edgelist = pd.read_csv(os.path.join(data_dir, dataset_name + ".cites"), sep='\t', header=None,
                           names=["target", "source"])
    Gnx = nx.from_pandas_edgelist(edgelist)
    if is_connected:
        largest_cc = max(nx.connected_components(Gnx), key=len)
        Gnx = nx.Graph(Gnx.subgraph(largest_cc))
    data_dir = os.path.expanduser("/Users/tony/PycharmProjects/cgssl/dataset")
    # first read examples from file
    f = open(os.path.join(data_dir, dataset_name + ".content"), 'r')
    line = f.readline()
    d = {}
    # matrix = []
    index = {}
    labels = []
    x = []
    while line:
        temp = line.split('\t')
        nodeid = (temp[0])
        d[nodeid] = {}
        # index[nodeid] = len(matrix)
        # matrix.append([int(f) for f in line.split('\t')[1:-1]])
        d[nodeid]['feature'] = [int(f) for f in line.split('\t')[1:-1]]
        if label_mapping is None:
            d[nodeid]['label'] = line.split('\t')[-1].strip('\n')
        else:
            d[nodeid]['label'] = label_mapping[line.split('\t')[-1].strip('\n')]
        # labels.append(d[nodeid]['label'])
        line = f.readline()
    for i in list(Gnx.nodes):
        if str(i) not in d:
            Gnx.remove_node(i)
        else:
            x.append(d[str(i)]['feature'])
            labels.append(d[str(i)]['label'])
    x = np.array(x)
    labels = np.array(labels)
    weight_mat = nx.adjacency_matrix(Gnx).todense()
    print(weight_mat.shape)
    w_ini = []
    for i in range(0, weight_mat.shape[0]):
        for j in range(i + 1, weight_mat.shape[0]):
            if weight_mat[i, j] < 0:
                weight_mat[i, j] = 0
            w_ini.append(weight_mat[i, j])
    return x, labels, w_ini, weight_mat, Gnx


def mnist_dataset(amount=2000):
    file = open('dataset/mnist_' + str(amount) + '_gist.pkl', 'rb')
    (x, y) = pkl.load(file)
    return x, y


def accuracy(y_pred, y_true, y_partial, conf_mat=True):
    t_c = 0
    c_c = 0
    y_pred_pure = []
    y_true_pure = []
    for i in range(y_pred.shape[0]):
        if y_partial[i] == unknown_label:
            t_c += 1
            y_true_pure.append(y_true[i])
            y_pred_pure.append(round(y_pred[i]))
            if abs(y_pred[i] - y_true[i]) < 0.5:
                c_c += 1
    if conf_mat:
        print(confusion_matrix(y_true_pure, y_pred_pure))
    return c_c / t_c


def cgssl(x, labels, y_part_sklearn, w, up_bound, low_bound, ini_w=None, alpha=1, beta=1, epsilon=10):
    previous_y = y_part_sklearn
    cur_y = opt_Y_given_L(w, y_part_sklearn, low_bound, up_bound)
    acc = accuracy(cur_y, labels, y_part_sklearn)
    print('current accuracy:', acc)

    while np.linalg.norm(previous_y - cur_y) >= epsilon:
        w = opt_L_given_Y(x, cur_y, ini_w, alpha, beta)
        for i in range(w.shape[0]):
            w[i][np.where(w[i] < 0.0001)[0]] = 0
        previous_y = cur_y
        cur_y = opt_Y_given_L(w, y_part_sklearn, low_bound, up_bound)
        print('current accuracy:', accuracy(cur_y, labels, y_part_sklearn))
    # accuracy(cur_y, labels, y_part_sklearn, True)
    return cur_y


def sample_test_train_indices(labels, frac):
    indices = [i for i in range(labels.shape[0])]
    train_index = sample(indices, max(1, round(labels.shape[0] * frac)))
    test_index = list(set(indices) - set(train_index))
    return np.array(labels[train_index]), np.array(train_index), np.array(labels[test_index]), np.array(test_index)


def multi_class_cgssl(x, labels, train_index, class_range, ini_w=None, alpha=1, beta=1, epsilon=10):
    if ini_w is None:
        w = opt_L_given_Y(x, alpha=alpha, beta=beta)
    else:
        w, _ = lg.log_degree_barrier_with_w0(x, dist_type='sqeuclidean', alpha=alpha, beta=beta,
                                             w0=ini_w, retall=True, verbosity='LOW', original_w=ini_w)

    vote = []
    for i in range(1, class_range + 1):
        cur_labels = [1 if label == i else 2 for label in labels]
        y_part_sklearn = np.array([cur_labels[i] if i in train_index else unknown_label for i in range(x.shape[0])])
        #     print('\ncgssl')
        up_bound = 2.5
        low_bound = 0.5
        ys = cgssl(x, cur_labels, y_part_sklearn, w, up_bound, low_bound, ini_w, alpha, beta, epsilon)
        vote.append(abs(ys - 1))
    vote = np.array(vote)
    cgssl_pred = np.argmin(vote, axis=0) + 1
    y_part_sklearn = np.array([labels[i] if i in train_index else unknown_label for i in range(x.shape[0])])
    print('Accuracy is', accuracy(cgssl_pred, labels, y_part_sklearn, True))
    return cgssl_pred, y_part_sklearn


def compare_methods(x, labels, train_labels, train_index, test_labels, test_index):
    y_part_sklearn = np.array([labels[i] if i in train_index else unknown_label for i in range(x.shape[0])])
    print('sklearn label propagation')

    lp = LabelPropagation()

    lp.fit(x, y_part_sklearn)
    y_sklearn_pred = lp.predict(x)
    print('accuracy:', accuracy(y_sklearn_pred, labels, y_part_sklearn))

    print('sklearn label Spreading')

    ls = LabelSpreading()

    ls.fit(x, y_part_sklearn)
    y_sklearn_pred = ls.predict(x)
    print('accuracy:', accuracy(y_sklearn_pred, labels, y_part_sklearn))

    print('\n[Kalofolias, 2016] + label propagation')
    w = opt_L_given_Y(x)
    adj_mat = sp.csr_matrix(matrix_from_weights(w, x.shape[0], False))
    hmn = HMN(adj_mat)
    hmn.fit(train_index, train_labels)
    y_pred = hmn.predict(test_index)

    print('first round accuracy', accuracy_score(test_labels, y_pred))

    y_new = []
    t_count = p_count = 0
    for i in range(x.shape[0]):
        if i in train_index:
            y_new.append(train_labels[t_count])
            t_count += 1
        else:
            y_new.append(y_pred[p_count])
            p_count += 1
    adj_mat = sp.csr_matrix(matrix_from_weights(opt_L_given_Y(x, np.array(y_new)), x.shape[0], False))
    hmn = HMN(adj_mat)
    hmn.fit(train_index, train_labels)
    y_pred = hmn.predict(test_index)
    print('second round accuracy', accuracy_score(test_labels, y_pred))


def write_edgelist(w, dir_name):
    g = nx.from_numpy_matrix(w)
    f = open('dataset/' + dir_name + '/edgelist.txt', "w+")
    f.write(str(len(g.nodes)) + ' ' + str(len(g.edges)) + '\n')
    f.close()
    f = open('dataset/' + dir_name + '/edgelist.txt', "ab")
    nx.write_weighted_edgelist(g, f)
    f.close()


def write_bound(labels, train_index, dir_name, ite=0):
    d = {'0': train_index, '1': labels[train_index]}
    df = pd.DataFrame(data=d)
    f = open('dataset/' + dir_name + '/bound' + str(ite) + '.txt', "w+")
    f.write(str(len(train_index)) + '\n')
    f.close()
    f = open('dataset/' + dir_name + '/bound' + str(ite) + '.txt', "a+")
    df.to_csv(f, sep=' ', header=False, index=False)
    f.close()


def update_mat(m, ls, eta):
    ls = ls.reshape((1, -1))
    update_loc = np.where(m > 0.0001)
    m[update_loc] = (m + eta * np.matmul(ls.T, ls))[update_loc]
    m[np.where(m<0.0001)] = 0
    return m


def normalized(a, axis=0, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def normalize_johnson(m, s):
    s_toadd = []
    np.fill_diagonal(m, 0)
    for i in range(m.shape[0]):
        div = np.sum(m[i]) + 1
        s_toadd.append(s[i] / div)
        for j in range(m.shape[1]):
            m[i][j] = m[i][j] / div
    return m, s_toadd


def result_metrics(y_true, y_pred, test_indices, reg=True, printing=True):
    total = 0.0
    correct = 0.0
    for i in test_indices:
        if y_true[i] != 0:
            total += 1
            if y_pred[i] * y_true[i] > 0:
                correct += 1
    if total == 0:
        acc = 0
    else:
        acc = correct / total
    if printing:
        if not reg:
            print('classification accuracy', acc, correct, total)
        else:
            print('regression mean_absolute_error', mean_absolute_error(y_true[test_indices], y_pred[test_indices]))
    return acc, mean_absolute_error(y_true[test_indices], y_pred[test_indices])
