from Models import *
from timeit import default_timer as timer
from Kernels import *
import logging
import pickle
from operator import itemgetter
from sklearn.model_selection import KFold
import multiprocessing as mp


def get_array(file):
    return np.load(file)


def get_accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def get_error(tp, fp, fn, tn):
    return (fp + fn) / (tp + fp + fn + tn)


def get_recall(tp, fp, fn, tn):
    return tp / (tp + fn) if tp + fn > 0 else 0


def get_specificity(tp, fp, fn, tn):
    return tn / (fp + tn) if fp + tn > 0 else 0


def get_precision(tp, fp, fn, tn):
    return tp / (tp + fp) if tp + fp > 0 else 0


def get_prevalence(tp, fp, fn, tn):
    return (tp + fn) / (tp + fp + fn + tn)


def get_fscore(pre, rec):
    return 2 * ((pre * rec) / (pre + rec)) if (pre + rec) > 0 else 0


def t(data, model, test_x, test_y):
    svm = model
    test_y = np.asarray(test_y).reshape(-1)
    if isinstance(svm, SVM):
        clf = svm.train(data.X, data)
    elif isinstance(svm, KT):
        svm.train(data)
        clf = svm
    else:
        clf = svm.train(data)
    predictions = []
    for test_point in test_x:
        predictions.append(clf.predict(test_point))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        elif test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        elif test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        elif test_y[i] == -1 and predictions[i] == -1:
            tn += 1
        else:
            print("This case should not be reached - if so, error")
    return tp, fp, fn, tn


def comp(clf, prob, test_x, test_y):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    average_time = 0
    start = timer()
    a, b, c, d = t(prob, clf, test_x, test_y)
    average_time += timer() - start
    tp += a
    fp += b
    fn += c
    tn += d
    return tp, fp, fn, tn, average_time


def test_svm(best, train_x, train_xs, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian": Gaussian(),
        "Quadratic": Polynomial(),
        "Linear": Linear()}

    test_prob = SvmProblem(train_x, train_xs, train_y, best[0], 0, 0, kern_options[best[1]], Linear())
    svm = SVM()
    clf = svm.train(train_x, test_prob)
    predictions = []
    for test_point in test_x:
        predictions.append(clf.predict(test_point))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    acc = get_accuracy(tp, fp, fn, tn)
    pre = get_precision(tp, fp, fn, tn)
    rec = get_recall(tp, fp, fn, tn)
    fs = get_fscore(pre, rec)
    return best[0], best[1], acc, fs


def test_svm_plus(best, train_x, train_xs, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian": Gaussian(),
        "Quadratic": Polynomial(),
        "Linear": Linear()}

    test_prob = SvmProblem(train_x, train_xs, train_y, c=best[0], gamma=best[1], delta=0, xk=kern_options[best[2]],
                           xsk=kern_options[best[3]])
    svm = SVMp()
    clf = svm.train(test_prob)
    predictions = []
    for test_point in test_x:
        predictions.append(clf.predict(test_point))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    acc = get_accuracy(tp, fp, fn, tn)
    pre = get_precision(tp, fp, fn, tn)
    rec = get_recall(tp, fp, fn, tn)
    fs = get_fscore(pre, rec)
    return best[0], best[1], best[2], best[3], acc, fs


def test_SVMmt(best, train_x, train_xs, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian": Gaussian(),
        "Quadratic": Polynomial(),
        "Linear": Linear()}

    test_prob = SvmProblem(train_x, train_xs, train_y, best[0], best[1], best[2], kern_options[best[3]],
                           kern_options[best[4]])
    svm = SVMdpSimp()
    clf = svm.train(test_prob)
    predictions = []
    for test_point in test_x:
        predictions.append(clf.predict(test_point))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    acc = get_accuracy(tp, fp, fn, tn)
    pre = get_precision(tp, fp, fn, tn)
    rec = get_recall(tp, fp, fn, tn)
    fs = get_fscore(pre, rec)
    return best[0], best[1], best[2], best[3], best[4], acc, fs


def test_svm_delta_plus(best, train_x, train_xs, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian": Gaussian(),
        "Quadratic": Polynomial(),
        "Linear": Linear()}

    test_prob = SvmProblem(train_x, train_xs, train_y, best[0], best[1], best[2], kern_options[best[3]],
                           kern_options[best[4]])
    svm = SVMdp()
    clf = svm.train(test_prob)
    predictions = []
    for test_point in test_x:
        predictions.append(clf.predict(test_point))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    acc = get_accuracy(tp, fp, fn, tn)
    pre = get_precision(tp, fp, fn, tn)
    rec = get_recall(tp, fp, fn, tn)
    fs = get_fscore(pre, rec)
    return best[0], best[1], best[2], best[3], best[4], acc, fs


def test_svm_u(best, train_x, train_xs, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian": Gaussian(),
        "Quadratic": Polynomial(),
        "Linear": Linear()}

    test_prob = SvmUProblem(train_x, train_xs, train_xs, train_y, best[0], best[1], best[2], best[3],
                            kern_options[best[4]], kern_options[best[5]], kern_options[best[6]])
    svm = SVMu()
    clf = svm.train(test_prob)
    predictions = []
    for test_point in test_x:
        predictions.append(clf.predict(test_point))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    acc = get_accuracy(tp, fp, fn, tn)
    pre = get_precision(tp, fp, fn, tn)
    rec = get_recall(tp, fp, fn, tn)
    fs = get_fscore(pre, rec)
    return best[0], best[1], best[2], best[3], best[4], best[5], best[6], acc, fs


def test_knowledge_transfer(best, train_x, train_xs, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian": Gaussian(),
        "Quadratic": Polynomial(),
        "Linear": Linear()}

    test_prob = SvmProblem(train_x, train_xs, train_y, best[0], best[1], 0, kern_options[best[2]],
                           kern_options[best[3]])
    svm = KT()
    svm.train(test_prob)
    predictions = []
    for test_point in test_x:
        predictions.append(svm.predict(test_point))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and predictions[i] == 1:
            tp += 1
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    acc = get_accuracy(tp, fp, fn, tn)
    pre = get_precision(tp, fp, fn, tn)
    rec = get_recall(tp, fp, fn, tn)
    fs = get_fscore(pre, rec)
    return best[0], best[1], best[2], best[3], acc, fs


def svm_comp(clf, prob, test_x, test_y):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    average_time = 0
    start = timer()
    a, b, c, d = t(prob, clf, test_x, test_y)
    average_time += timer() - start
    tp += a
    fp += b
    fn += c
    tn += d
    return tp, fp, fn, tn, average_time


def grid_search_svm(c_list, dataset):
    xkerns = [Gaussian(), Polynomial(), Linear()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_labels.npy")
        test_x = get_array("Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_normal.npy")
        train_y = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_labels.npy").reshape(-1,1)
        train_x = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_normal.npy")
        train_xS = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_priv.npy")
        prob_data.append([test_labels, test_x, train_y, train_x, train_xS])

        prob_data = np.array(prob_data)
        conc_data = []
        for j in range(len(prob_data)):
            conc_data = np.hstack((train_x, train_xS))
            conc_data = np.hstack((conc_data, train_y))
        np.random.shuffle(conc_data)

        kf = KFold(shuffle=True, n_splits=5)
        inner_folds_train_index = []
        inner_folds_test_index = []
        inner_folds_train = []
        inner_folds_test = []
        for train, test in kf.split(conc_data):
            inner_folds_train_index.append(train)
            inner_folds_test_index.append(test)

        for j in range(5):
            inner_folds_train.append(np.array(conc_data[inner_folds_train_index[j]]))
            inner_folds_test.append(np.array(conc_data[inner_folds_test_index[j]]))
        del conc_data, prob_data, train, test

        inner_probs = []
        for j in range(5):
            inner_x = inner_folds_train[j][:, 0:train_x.shape[1]]
            inner_xs = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1] + train_xS.shape[1]]
            inner_y = inner_folds_train[j][:,
                      train_x.shape[1] + train_xS.shape[1]:train_x.shape[1] + train_xS.shape[1] + train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:,
                           :test_x.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xs, i, j])

        del inner_folds_test, \
            inner_folds_test_index, \
            inner_folds_train, \
            inner_folds_train_index, \
            kf, \
            inner_test_x, \
            inner_test_y, \
            inner_x, \
            inner_xs, \
            inner_y, \
            j

        inner_probs = np.array(inner_probs)

        svm_prob = [(data[3], data[4], data[2], data[1], data[0], c, xk, data[5], data[6])
                    for data in inner_probs
                    for c in c_list for xk in xkerns]
        print(len(svm_prob), dataset, i)
        del inner_probs

        no_cores = 4
        try:
            no_cores = mp.cpu_count()
            logging.info("No of Cores: " + str(no_cores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(no_cores)
        results_svm = pool.map(svm_thread, svm_prob)
        pool.close()
        pool.join()

        kerns = ["Gaussian", "Linear", "Quadratic"]
        inner_svm_results = [(c, xker, np.mean([x[3] for x in results_svm if x[5] == c and x[6] == xker]),
                              np.mean([x[4] for x in results_svm if x[5] == c and x[6] == xker])) for c in c_list for xker in
                             kerns]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(2, 3))
        best_f = max(inner_svm_results, key=itemgetter(3, 2))

        results_acc.append(test_svm(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_svm(best_f, train_x, train_xS, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[2] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[3] for x in results_f]))

    with open('SVM_results_all_' + str(dataset), 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVM_results_acc_' + str(dataset), 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVM_results_f_' + str(dataset), 'wb') as fp:
        pickle.dump(results_f, fp)


def grid_search_knowledge_transfer(c_list, dataset):
    xkerns = [Linear(), Gaussian(), Polynomial()]
    xskerns = [Linear(), Gaussian(), Polynomial()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_labels.npy")
        test_x = get_array("Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_normal.npy")
        train_y = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_labels.npy").reshape(-1,
                                                                                                                   1)
        train_x = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_normal.npy")
        train_xs = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_priv.npy")
        prob_data.append([test_labels, test_x, train_y, train_x, train_xs])

        prob_data = np.array(prob_data)
        conc_data = []
        for j in range(len(prob_data)):
            conc_data = np.hstack((train_x, train_xs))
            conc_data = np.hstack((conc_data, train_y))
        np.random.shuffle(conc_data)

        kf = KFold(shuffle=True, n_splits=5)
        inner_folds_train_index = []
        inner_folds_test_index = []
        inner_folds_train = []
        inner_folds_test = []
        for train, test in kf.split(conc_data):
            inner_folds_train_index.append(train)
            inner_folds_test_index.append(test)

        for j in range(5):
            inner_folds_train.append(np.array(conc_data[inner_folds_train_index[j]]))
            inner_folds_test.append(np.array(conc_data[inner_folds_test_index[j]]))
        del conc_data, prob_data, train, test

        inner_probs = []
        for j in range(5):
            inner_x = inner_folds_train[j][:, 0:train_x.shape[1]]
            inner_xs = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1] + train_xs.shape[1]]
            inner_y = inner_folds_train[j][:,
                      train_x.shape[1] + train_xs.shape[1]:train_x.shape[1] + train_xs.shape[1] + train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:,
                           :test_x.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xs, i, j])

        del inner_folds_test, \
            inner_folds_test_index, \
            inner_folds_train, \
            inner_folds_train_index, \
            kf, \
            inner_test_x, \
            inner_test_y, \
            inner_x, \
            inner_xs, \
            inner_y, \
            j

        inner_probs = np.array(inner_probs)

        svm_prob = [(data[3], data[4], data[2], data[1], data[0], c, c2, xk, xsk, data[5], data[6], 0, 0, 0, 0) for data
                    in inner_probs for c in c_list for c2 in c_list for xk in xkerns for xsk in xskerns]
        print(len(svm_prob), dataset, i)
        del inner_probs

        no_cores = 4
        try:
            no_cores = mp.cpu_count()
            logging.info("No of Cores: " + str(no_cores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(no_cores)
        results_kt = pool.map(kt_thread, svm_prob)
        pool.close()
        pool.join()

        kerns = ["Linear", "Gaussian", "Quadratic"]
        kerns2 = ["Linear", "Gaussian", "Quadratic"]
        inner_kt_results = [(c, c2, xker, xsker, np.mean(
            [x[3] for x in results_kt if x[5] == c and x[6] == c2 and x[7] == xker and x[8] == xsker]), np.mean(
            [x[4] for x in results_kt if x[5] == c and x[6] == c2 and x[7] == xker and x[8] == xsker])) for c in c_list for
                             c2 in c_list for xker in kerns for xsker in kerns2]
        results_to_save.append(inner_kt_results)
        print(inner_kt_results)
        best_acc = max(inner_kt_results, key=itemgetter(4, 5))
        best_f = max(inner_kt_results, key=itemgetter(5, 4))

        results_acc.append(test_knowledge_transfer(best_acc, train_x, train_xs, train_y, test_x, test_labels))
        results_f.append(test_knowledge_transfer(best_f, train_x, train_xs, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[4] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[5] for x in results_f]))

    with open('KT_results_all_' + str(dataset) + '_lin_lin_am', 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('KT_results_acc_' + str(dataset) + '_lin_lin_am', 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('KT_results_f_' + str(dataset) + '_lin_lin_am', 'wb') as fp:
        pickle.dump(results_f, fp)


def grid_search_svm_plus(c_list, Gamma, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Linear(), Gaussian(), Polynomial()]
    xskerns = [Polynomial(), Gaussian(), Linear()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_labels.npy")
        test_x = get_array("Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_normal.npy")
        train_y = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_labels.npy").reshape(-1,
                                                                                                                   1)
        train_x = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_normal.npy")
        train_xs = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_priv.npy")
        prob_data.append([test_labels, test_x, train_y, train_x, train_xs])

        prob_data = np.array(prob_data)
        conc_data = []
        for j in range(len(prob_data)):
            conc_data = np.hstack((train_x, train_xs))
            conc_data = np.hstack((conc_data, train_y))
        np.random.shuffle(conc_data)

        kf = KFold(shuffle=True, n_splits=5)
        inner_folds_train_index = []
        inner_folds_test_index = []
        inner_folds_train = []
        inner_folds_test = []
        for train, test in kf.split(conc_data):
            inner_folds_train_index.append(train)
            inner_folds_test_index.append(test)

        for j in range(5):
            inner_folds_train.append(np.array(conc_data[inner_folds_train_index[j]]))
            inner_folds_test.append(np.array(conc_data[inner_folds_test_index[j]]))
        del conc_data, prob_data, train, test

        inner_probs = []
        for j in range(5):
            inner_x = inner_folds_train[j][:, 0:train_x.shape[1]]
            inner_xs = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1] + train_xs.shape[1]]
            inner_y = inner_folds_train[j][:,
                      train_x.shape[1] + train_xs.shape[1]:train_x.shape[1] + train_xs.shape[1] + train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:,
                           :test_x.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xs, i, j])

        del inner_folds_test, \
            inner_folds_test_index, \
            inner_folds_train, \
            inner_folds_train_index, \
            kf, \
            inner_test_x, \
            inner_test_y, \
            inner_x, \
            inner_xs, \
            inner_y, \
            j

        inner_probs = np.array(inner_probs)

        svmp_prob = [(data[3], data[4], data[2], data[1], data[0], c, g, xk, xsk, data[5], data[6]) for data in
                     inner_probs for c in c_list for g in Gamma for xk in xkerns for xsk in xskerns]
        print(len(svmp_prob), dataset, i)
        del inner_probs

        no_cores = 4
        try:
            no_cores = mp.cpu_count()
            logging.info("No of Cores: " + str(no_cores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(no_cores)
        resultsSVMp = pool.map(svm_plus_thread, svmp_prob)
        pool.close()
        pool.join()

        results_svm_p = sorted(resultsSVMp, key=itemgetter(6, 1, 2, 4, 3))

        kerns = ["Linear", "Quadratic", "Gaussian"]
        kerns2 = ["Quadratic", "Linear", "Gaussian"]
        inner_svm_results = [(c, g, xker, xSk, np.mean(
            [x[3] for x in results_svm_p if x[5] == c and x[6] == g and x[7] == xker and x[8] == xSk]), np.mean(
            [x[4] for x in results_svm_p if x[5] == c and x[6] == g and x[7] == xker and x[8] == xSk])) for c in c_list for g in
                             Gamma for xker in kerns for xSk in kerns2]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(4, 5))
        best_f = max(inner_svm_results, key=itemgetter(4, 5))

        results_acc.append(test_svm_plus(best_acc, train_x, train_xs, train_y, test_x, test_labels))
        results_f.append(test_svm_plus(best_f, train_x, train_xs, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[4] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[5] for x in results_f]))

    with open('SVMp_results_all_' + str(dataset) + '_lin_quad', 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVMp_results_acc_' + str(dataset) + '_lin_quad', 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVMp_results_f_' + str(dataset) + '_lin_quad', 'wb') as fp:
        pickle.dump(results_f, fp)


def grid_search_margin_transfer(c_list, Delta, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Linear(), Gaussian(), Polynomial()]
    xskerns = [Gaussian(), Linear(), Polynomial()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_labels.npy")
        test_x = get_array("Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_normal.npy")
        train_y = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_labels.npy").reshape(-1,
                                                                                                                   1)
        train_x = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_normal.npy")
        train_xs = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_priv.npy")
        prob_data.append([test_labels, test_x, train_y, train_x, train_xs])

        prob_data = np.array(prob_data)
        conc_data = []
        for j in range(len(prob_data)):
            conc_data = np.hstack((train_x, train_xs))
            conc_data = np.hstack((conc_data, train_y))
        np.random.shuffle(conc_data)

        kf = KFold(shuffle=True, n_splits=5)
        inner_folds_train_index = []
        inner_folds_test_index = []
        inner_folds_train = []
        inner_folds_test = []
        for train, test in kf.split(conc_data):
            inner_folds_train_index.append(train)
            inner_folds_test_index.append(test)

        for j in range(5):
            inner_folds_train.append(np.array(conc_data[inner_folds_train_index[j]]))
            inner_folds_test.append(np.array(conc_data[inner_folds_test_index[j]]))
        del conc_data, prob_data, train, test

        inner_probs = []
        for j in range(5):
            inner_x = inner_folds_train[j][:, 0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1] + train_xs.shape[1]]
            inner_y = inner_folds_train[j][:,
                      train_x.shape[1] + train_xs.shape[1]:train_x.shape[1] + train_xs.shape[1] + train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:,
                           :test_x.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svmdps_prob = [(data[3], data[4], data[2], data[1], data[0], c, c2, d, xk, xsk, data[5], data[6]) for data in
                       inner_probs for c in c_list for c2 in c_list for d in Delta for xk in xkerns for xsk in xskerns]
        print(len(svmdps_prob), dataset, i)
        del inner_probs

        no_cores = 4
        try:
            no_cores = mp.cpu_count()
            logging.info("No of Cores: " + str(no_cores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(no_cores)
        results_mt = pool.map(svm_mt_thread, svmdps_prob)
        pool.close()
        pool.join()

        kerns = ["Linear", "Gaussian", "Quadratic"]
        kerns2 = ["Gaussian", "Linear", "Quadratic"]
        inner_svm_results = [(c, c2, d, xker, xsker, np.mean([x[3] for x in results_mt if
                                                              x[5] == c and x[6] == c2 and x[7] == d and x[
                                                                  8] == xker and x[9] == xsker]), np.mean(
            [x[4] for x in results_mt if
             x[5] == c and x[6] == c2 and x[7] == d and x[8] == xker and x[9] == xsker])) for c in c_list for c2 in c_list for d
                             in Delta for xker in kerns for xsker in kerns2]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(4, 5))
        best_f = max(inner_svm_results, key=itemgetter(5, 4))

        results_acc.append(test_SVMmt(best_acc, train_x, train_xs, train_y, test_x, test_labels))
        results_f.append(test_SVMmt(best_f, train_x, train_xs, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[5] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[6] for x in results_f]))

    with open('SVMmt_results_all_' + str(dataset) + "_fixed_delta_lin_gaus_am", 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVMmt_results_acc_' + str(dataset) + "_fixed_delta_lin_gaus_am", 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVMmt_results_f_' + str(dataset) + "_fixed_delta_lin_gaus_am", 'wb') as fp:
        pickle.dump(results_f, fp)


def grid_search_svm_delta_plus(c_list, Delta, Gamma, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Linear(), Gaussian(), Polynomial()]
    xskerns = [Linear(), Gaussian(), Polynomial()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_labels.npy")
        test_x = get_array("Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_normal.npy")
        train_y = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_labels.npy").reshape(-1,
                                                                                                                   1)
        train_x = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_normal.npy")
        train_xs = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_priv.npy")
        prob_data.append([test_labels, test_x, train_y, train_x, train_xs])

        prob_data = np.array(prob_data)
        conc_data = []
        for j in range(len(prob_data)):
            conc_data = np.hstack((train_x, train_xs))
            conc_data = np.hstack((conc_data, train_y))
        np.random.shuffle(conc_data)

        kf = KFold(shuffle=True, n_splits=5)
        inner_folds_train_index = []
        inner_folds_test_index = []
        inner_folds_train = []
        inner_folds_test = []
        for train, test in kf.split(conc_data):
            inner_folds_train_index.append(train)
            inner_folds_test_index.append(test)

        for j in range(5):
            inner_folds_train.append(np.array(conc_data[inner_folds_train_index[j]]))
            inner_folds_test.append(np.array(conc_data[inner_folds_test_index[j]]))
        del conc_data, prob_data, train, test

        inner_probs = []
        for j in range(5):
            inner_x = inner_folds_train[j][:, 0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1] + train_xs.shape[1]]
            inner_y = inner_folds_train[j][:,
                      train_x.shape[1] + train_xs.shape[1]:train_x.shape[1] + train_xs.shape[1] + train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:,
                           :test_x.shape[1]]  # test_x.shape[1]:test_x.shape[1]+train_xs.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svmdp_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, xk, xsk, data[5], data[6], 0) for data in
                      inner_probs for c in c_list for d in Delta for g in Gamma for xk in xkerns for xsk in xskerns]
        print(len(svmdp_prob), dataset, i)
        del inner_probs

        no_cores = 4
        try:
            no_cores = mp.cpu_count()
            logging.info("No of Cores: " + str(no_cores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(no_cores)
        results_svmdp = pool.map(svm_dp_thread, svmdp_prob)
        pool.close()
        pool.join()

        kerns = ["Linear", "Gaussian", "Quadratic"]
        kerns2 = ["Linear", "Gaussian", "Quadratic"]
        inner_svm_results = [(c, g, d, xker, xsker, np.mean(
            [x[3] for x in results_svmdp if x[5] == c and x[6] == d and x[7] == g and x[8] == xker and x[9] == xsker]),
                              np.mean([x[4] for x in results_svmdp if
                                       x[5] == c and x[6] == d and x[7] == g and x[8] == xker and x[9] == xsker])) for c
                             in c_list for d in Delta for g in Gamma for xker in kerns for xsker in kerns2]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(5, 6))
        best_f = max(inner_svm_results, key=itemgetter(6, 5))

        results_acc.append(test_svm_delta_plus(best_acc, train_x, train_xs, train_y, test_x, test_labels))
        results_f.append(test_svm_delta_plus(best_f, train_x, train_xs, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[5] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[6] for x in results_f]))

    with open('SVMdp_results_all_' + str(dataset) + "_fixed_delta_lin_lin", 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVMdp_results_acc_' + str(dataset) + "_fixed_delta_lin_lin", 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVMdp_results_f_' + str(dataset) + "_fixed_delta_lin_lin", 'wb') as fp:
        pickle.dump(results_f, fp)


def grid_search_svm_u(c_list, delta_list, gamma_list, sigma_list, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Gaussian()]
    xskerns = [Polynomial()]
    xsskerns = [Gaussian()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_labels.npy")
        test_x = get_array("Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-test_normal.npy")
        train_y = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_labels.npy").reshape(-1,
                                                                                                                   1)
        train_x = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_normal.npy")
        train_xS = get_array(
            "Data/Dataset" + str(dataset) + "/tech" + str(dataset) + "-0-" + str(i) + "-train_priv.npy")
        prob_data.append([test_labels, test_x, train_y, train_x, train_xS])

        prob_data = np.array(prob_data)
        conc_data = []
        for j in range(len(prob_data)):
            conc_data = np.hstack((train_x, train_xS))
            conc_data = np.hstack((conc_data, train_y))
        np.random.shuffle(conc_data)

        kf = KFold(shuffle=True, n_splits=5)
        inner_folds_train_index = []
        inner_folds_test_index = []
        inner_folds_train = []
        inner_folds_test = []
        for train, test in kf.split(conc_data):
            inner_folds_train_index.append(train)
            inner_folds_test_index.append(test)

        for j in range(5):
            inner_folds_train.append(np.array(conc_data[inner_folds_train_index[j]]))
            inner_folds_test.append(np.array(conc_data[inner_folds_test_index[j]]))
        del conc_data, prob_data, train, test

        inner_probs = []
        for j in range(5):
            inner_x = inner_folds_train[j][:, 0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1] + train_xS.shape[1]]
            inner_y = inner_folds_train[j][:,
                      train_x.shape[1] + train_xS.shape[1]:train_x.shape[1] + train_xS.shape[1] + train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:,
                           :test_x.shape[1]]  # test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svmu_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, S, xk, xsk, xssk, data[5], data[6]) for data
                     in inner_probs for c in c_list for d in delta_list for g in gamma_list for S in sigma_list for xk in xkerns for xsk in
                     xskerns for xssk in xsskerns]
        print(len(svmu_prob), dataset, i)
        del inner_probs

        no_cores = 4
        try:
            no_cores = mp.cpu_count()
            logging.info("No of Cores: " + str(no_cores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(no_cores)
        results_svm_u = pool.map(svm_u_thread, svmu_prob)
        pool.close()
        pool.join()

        kerns = ["Gaussian"]
        kerns2 = ["Quadratic"]
        kerns3 = ["Gaussian"]
        inner_svm_results = [(c, g, s, d, xker, xsker, xssker, np.mean([x[3] for x in results_svm_u if
                                                                        x[5] == c and x[6] == d and x[7] == g and x[
                                                                            8] == s and x[9] == xker and x[
                                                                            10] == xsker and x[11] == xssker]), np.mean(
            [x[4] for x in results_svm_u if
             x[5] == c and x[6] == d and x[7] == g and x[8] == s and x[9] == xker and x[10] == xsker and x[
                 11] == xssker])) for c in c_list for d in delta_list for g in gamma_list for s in sigma_list for xker in kerns for xsker in
                             kerns2 for xssker in kerns3]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(7, 8))
        best_f = max(inner_svm_results, key=itemgetter(8, 7))

        results_acc.append(test_svm_u(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_svm_u(best_f, train_x, train_xS, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[7] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[8] for x in results_f]))

    with open('TEST_SVMu_results_all_' + str(dataset) + "_fixed_delta_gaus_quad_gaus", 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('TEST_SVMu_results_acc_' + str(dataset) + "_fixed_delta_gaus_quad_gaus", 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('TEST_SVMu_results_f_' + str(dataset) + "_fixed_delta_gaus_quad_gaus", 'wb') as fp:
        pickle.dump(results_f, fp)


def svm_thread(p):
    prob = SvmProblemTuple(p)
    logging.info("Entered multicore process")
    svm_tp, svm_fp, svm_fn, svm_tn, svm_avg_time = comp(SVM(), prob, p[3], p[4])
    logging.info("model trained" + " " + str(svm_tp) + " " + str(svm_fp) + " " + str(svm_fn) + " " + str(svm_tn))
    svm_acc = get_accuracy(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_pre = get_precision(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_rec = get_recall(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_fsc = get_fscore(svm_pre, svm_rec)
    return ("SVM", p[7], p[8], svm_acc, svm_fsc, p[5], p[6].get_name(), svm_avg_time)


def svm_plus_thread(p):
    prob = SvmProblemTuple(p)
    logging.info("Entered multicore process")
    svmp_tp, svmp_fp, svmp_fn, svmp_tn, svmp_avg_time = comp(SVMp(), prob, p[3], p[4])
    logging.info("model trained")
    svmp_acc = get_accuracy(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_pre = get_precision(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_rec = get_recall(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_fsc = get_fscore(svmp_pre, svmp_rec)
    logging.info("Completed")
    return ("SVM+", p[9], p[10], svmp_acc, svmp_fsc, p[5], p[6], p[7].get_name(), p[8].get_name(), svmp_avg_time)


def svm_mt_thread(p):
    prob = SvmProblemTuple(p)
    logging.info("Entered multicore process")
    svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn, svmdpsa_avg_time = comp(SVMdpSimp(), prob, p[3], p[4])
    logging.info("model trained")
    svmdpsa_acc = get_accuracy(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_pre = get_precision(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_rec = get_recall(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_fsc = get_fscore(svmdpsa_pre, svmdpsa_rec)
    logging.info("Completed")
    return ("SVMd+ - simp", p[10], p[11], svmdpsa_acc, svmdpsa_fsc, p[5], p[6], p[7], p[8].get_name(), p[9].get_name(),
            svmdpsa_avg_time)


def svm_dp_thread(p):
    prob = SvmProblemTuple(p)
    logging.info("Entered multicore process")
    svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn, svmdp_avg_time = comp(SVMdp(), prob, p[3], p[4])
    logging.info("model trained")
    svmdp_acc = get_accuracy(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_pre = get_precision(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_rec = get_recall(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_fsc = get_fscore(svmdp_pre, svmdp_rec)
    logging.info("Completed")
    return (
        "SVMdp", p[10], p[11], svmdp_acc, svmdp_fsc, p[5], p[6], p[7], p[8].get_name(), p[9].get_name(), svmdp_avg_time)


def svm_u_thread(p):
    logging.info("Entered thread")
    prob = SvmUProblemTuple(p)
    logging.info("Entered multicore process")
    svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn, svmdp_avg_time = comp(SVMu(), prob, p[3], p[4])
    logging.info("model trained")
    svmdp_acc = get_accuracy(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_pre = get_precision(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_rec = get_recall(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_fsc = get_fscore(svmdp_pre, svmdp_rec)
    logging.info("Completed")
    return ("SVMu", p[12], p[13], svmdp_acc, svmdp_fsc, p[5], p[6], p[7], p[8], p[9].get_name(), p[10].get_name(),
            p[11].get_name(), svmdp_avg_time)


def kt_thread(p):
    prob = SvmProblemTuple(p)
    logging.info("Entered multicore process")
    svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn, svmdpsa_avg_time = comp(KT(), prob, p[3], p[4])
    logging.info("model trained")
    svmdpsa_acc = get_accuracy(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_pre = get_precision(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_rec = get_recall(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_fsc = get_fscore(svmdpsa_pre, svmdpsa_rec)
    logging.info("Completed")
    return ("SVMkt - simp", p[9], p[10], svmdpsa_acc, svmdpsa_fsc, p[5], p[6], p[7].get_name(), p[8].get_name(),
            svmdpsa_avg_time)


def main():
    grid_search_svm([0.001, 0.1, 10, 1000], 137)
    grid_search_svm([0.001, 0.1, 10, 1000], 174)
    grid_search_svm([0.001, 0.1, 10, 1000], 197)
    grid_search_svm([0.001, 0.1, 10, 1000], 219)
    grid_search_svm([0.001, 0.1, 10, 1000], 254)
    grid_search_svm_plus([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 137)
    grid_search_svm_plus([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 174)
    grid_search_svm_plus([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 197)
    grid_search_svm_plus([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 219)
    grid_search_svm_plus([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 254)
    grid_search_margin_transfer([0.001, 0.1, 10, 1000], [1000], 137)
    grid_search_margin_transfer([0.001, 0.1, 10, 1000], [1000], 174)
    grid_search_margin_transfer([0.001, 0.1, 10, 1000], [1000], 197)
    grid_search_margin_transfer([0.001, 0.1, 10, 1000], [1000], 219)
    grid_search_margin_transfer([0.001, 0.1, 10, 1000], [1000], 254)
    grid_search_svm_delta_plus([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 137)
    grid_search_svm_delta_plus([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 174)
    grid_search_svm_delta_plus([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 197)
    grid_search_svm_delta_plus([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 219)
    grid_search_svm_delta_plus([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 254)
    grid_search_svm_u([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 137)
    grid_search_svm_u([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 174)
    grid_search_svm_u([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 197)
    grid_search_svm_u([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 219)
    grid_search_svm_u([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 254)
    grid_search_knowledge_transfer([0.001, 0.1, 10, 1000], 137)
    grid_search_knowledge_transfer([0.001, 0.1, 10, 1000], 174)
    grid_search_knowledge_transfer([0.001, 0.1, 10, 1000], 197)
    grid_search_knowledge_transfer([0.001, 0.1, 10, 1000], 219)
    grid_search_knowledge_transfer([0.001, 0.1, 10, 1000], 254)

    print("----- Accuracy -----")
    with open('SVM_results_acc_137_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open('SVMp_results_acc_137_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open('SVMmt_results_acc_137_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMdp_results_acc_137_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMu_results_acc_137_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open('KT_results_acc_137_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open('SVM_results_acc_174_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open('SVMp_results_acc_174_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open('SVMmt_results_acc_174_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMdp_results_acc_174_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMu_results_acc_174_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open('KT_results_acc_174_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open('SVM_results_acc_197_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open('SVMp_results_acc_197_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open('SVMmt_results_acc_197_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMdp_results_acc_197_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMu_results_acc_197_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open('KT_results_acc_197_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open('SVM_results_acc_219_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open('SVMp_results_acc_219_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open('SVMmt_results_acc_219_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMdp_results_acc_219_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMu_results_acc_219_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open('KT_results_acc_219_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open('SVM_results_acc_254_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open('SVMp_results_acc_254_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open('SVMmt_results_acc_254_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMdp_results_acc_254_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMu_results_acc_254_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open('KT_results_acc_254_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    print("----- F-Score -----")
    with open('SVM_results_f_137_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open('SVMp_results_f_137_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMmt_results_f_137_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMdp_results_f_137_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMu_results_f_137_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open('KT_results_f_137_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open('SVM_results_f_174_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open('SVMp_results_f_174_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMmt_results_f_174_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMdp_results_f_174_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMu_results_f_174_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open('KT_results_f_174_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open('SVM_results_f_197_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open('SVMp_results_f_197_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMmt_results_f_197_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMdp_results_f_197_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMu_results_f_197_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open('KT_results_f_197_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open('SVM_results_f_219_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open('SVMp_results_f_219_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMmt_results_f_219_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMdp_results_f_219_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMu_results_f_219_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open('KT_results_f_219_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open('SVM_results_f_254_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open('SVMp_results_f_254_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open('SVMmt_results_f_254_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMdp_results_f_254_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open('SVMu_results_f_254_fixed_delta_lin_quad_gaus_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open('KT_results_f_254_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # mp.set_start_method('spawn')
    # mp.freeze_support()
    print(mp.get_start_method())
    # mp.context
    main()
