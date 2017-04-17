from Models import *
from timeit import default_timer as timer
from Kernels import *
import logging
import pickle
from operator import itemgetter
from sklearn.model_selection import KFold
import time
import multiprocessing as mp


def get_array(file):
    return np.load(file)

def get_accuracy(tp, fp, fn, tn):
    return (tp+tn)/(tp+fp+fn+tn)#+0.000001)

def get_error(tp, fp, fn, tn):
    return (fp+fn)/(tp+fp+fn+tn)#+0.000001)

def get_recall(tp, fp, fn, tn):
    return (tp)/(tp+fn) if tp+fn > 0 else 0

def get_specificity(tp, fp, fn, tn):
    return (tn)/(fp+tn) if fp+tn > 0 else 0

def get_precision(tp, fp, fn, tn):
    return (tp)/(tp+fp) if tp+fp > 0 else 0

def get_prevalence(tp, fp, fn, tn):
    return (tp+fn)/(tp+fp+fn+tn)#+0.000001)

def get_fscore(pre, rec):
    return 2*((pre*rec)/(pre+rec)) if (pre+rec) > 0 else 0

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
        if test_y[i] == -1 and predictions[i] == 1:
            fp += 1
        if test_y[i] == 1 and predictions[i] == -1:
            fn += 1
        if test_y[i] == -1 and predictions[i] == -1:
            tn += 1
    return (tp, fp, fn, tn)

def comp(clf, prob, test_x, test_y):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    average_time = 0
    #for fold in probs:
    start = timer()
    a, b, c, d = t(prob, clf, test_x, test_y)
    average_time += timer() - start
    tp += a
    fp += b
    fn += c
    tn += d
    return tp,fp,fn,tn,average_time

def test_SVM(best, train_x, train_xS, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian" : Gaussian(),
        "Quadratic" : Polynomial(),
        "Linear" : Linear()}

    testprob = svm_problem(train_x, train_xS, train_y, best[0], 0, 0, kern_options[best[1]], Linear())
    svm = SVM()
    clf = svm.train(train_x, testprob)
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
    return(best[0], best[1], acc, fs)

def test_SVMp(best, train_x, train_xS, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian" : Gaussian(),
        "Quadratic" : Polynomial(),
        "Linear" : Linear()}

    testprob = svm_problem(train_x, train_xS, train_y, c=best[0], gamma=best[1], delta=0, xk=kern_options[best[2]], xSk=kern_options[best[3]])
    svm = SVMp()
    clf = svm.train(testprob)
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
    return(best[0], best[1], best[2], best[3], acc, fs)

def test_SVMmt(best, train_x, train_xS, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian" : Gaussian(),
        "Quadratic" : Polynomial(),
        "Linear" : Linear()}

    testprob = svm_problem(train_x, train_xS, train_y, best[0], best[1], best[2], kern_options[best[3]], kern_options[best[4]])
    svm = SVMdp_simp()
    clf = svm.train(testprob)
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
    return(best[0], best[1], best[2], best[3], best[4], acc, fs)

def test_SVMdp(best, train_x, train_xS, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian" : Gaussian(),
        "Quadratic" : Polynomial(),
        "Linear" : Linear()}

    testprob = svm_problem(train_x, train_xS, train_y, best[0], best[1], best[2], kern_options[best[3]], kern_options[best[4]])
    svm = SVMdp()
    clf = svm.train(testprob)
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
    return(best[0], best[1], best[2], best[3], best[4], acc, fs)

def test_SVMu(best, train_x, train_xS, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian" : Gaussian(),
        "Quadratic" : Polynomial(),
        "Linear" : Linear()}

    testprob = svm_u_problem(train_x, train_xS, train_xS, train_y, best[0], best[1], best[2], best[3], kern_options[best[4]], kern_options[best[5]], kern_options[best[6]])
    svm = SVMu()
    clf = svm.train(testprob)
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
    return(best[0], best[1], best[2], best[3], best[4], best[5], best[6], acc, fs)

def test_KT(best, train_x, train_xS, train_y, test_x, test_labels):
    test_y = test_labels
    kern_options = {
        "Gaussian" : Gaussian(),
        "Quadratic" : Polynomial(),
        "Linear" : Linear()}

    testprob = svm_problem(train_x, train_xS, train_y, best[0], best[1], 0, kern_options[best[2]], kern_options[best[3]])
    svm = KT()
    svm.train(testprob)
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
    return(best[0], best[1], best[2], best[3], acc, fs)

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
    return tp,fp,fn,tn,average_time

def grid_search(C, Delta, Gamma, dataset, Sigma=[]):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Gaussian(), Polynomial(), Linear()]
    xskerns = [Gaussian(), Polynomial(), Linear()]
    xsskerns = [Gaussian(), Polynomial(), Linear()]

    results_acc = []
    results_f = []

    for i in range(10):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
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
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svm_prob = [(data[3], data[4], data[2], data[1], data[0], c, xk, data[5], data[6]) for data in inner_probs for c in C for xk in xkerns]
        #svmp_prob = [(data[3], data[4], data[2], data[1], data[0], c, g, xk, xsk, data[5], data[6]) for data in inner_probs for c in C for g in Gamma for xk in xkerns for xsk in xskerns]
        #svmdps_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, xk, xsk, data[5], data[6], 0) for data in inner_probs for c in C for d in Delta for xk in xkerns for xsk in xskerns]
        #svmdp_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, xk, xsk, data[5], data[6], 0) for data in inner_probs for c in C for d in Delta for g in Gamma for xk in xkerns for xsk in xskerns]
        #svmu_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, S, xk, xsk, xssk, data[5], data[6]) for data in inner_probs for c in C for d in Delta for g in Gamma for S in Sigma for xk in xkerns for xsk in xskerns for xssk in xsskerns if S >= 1/g]
        print(len(svm_prob))
        del inner_probs

        noCores = 4
        try:
            noCores = mp.cpu_count()
            logging.info("No of Cores: "+str(noCores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(noCores)
        resultsSVM = pool.map(svmThread, svm_prob)
        #resultsSVMp = pool.map(svmpThread, svmp_prob)
        #resultsSVMdpsa = pool.map(svmdpsaThread, svmdps_prob)
        #resultsSVMdp = pool.map(svmdpThread, svmdp_prob)
        #resultsSVMu = pool.map(svmuThread, svmu_prob)
        pool.close()
        pool.join()

        resultsSVM = sorted(resultsSVM, key=itemgetter(6,1,2,4,3))

        kerns =  ["Gaussian", "Linear", "Quadratic"]
        inner_svm_results = [(c, xker, np.mean([x[3] for x in resultsSVM if x[5] == c and x[6] == xker]), np.mean([x[4] for x in resultsSVM if x[5] == c and x[6] == xker])) for c in C for xker in kerns]
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(2, 3))
        best_f = max(inner_svm_results, key=itemgetter(3, 2))

        results_acc.append(test_SVM(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_SVM(best_f, train_x, train_xS, train_y, test_x, test_labels))

        #results = results+resultsSVM#resultsSVM+resultsSVMp+resultsSVMdpsa+resultsSVMdp
    print(results_acc)
    print("Accuracy: ", np.mean([x[2] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[3] for x in results_f]))

    #with open('SVMuoutfileRTEST', 'wb') as fp:
    #    pickle.dump(results, fp)

def grid_search_SVM(C, dataset):

    xkerns = [Gaussian(), Polynomial(), Linear()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
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
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svm_prob = [(data[3], data[4], data[2], data[1], data[0], c, xk, data[5], data[6]) for data in inner_probs for c in C for xk in xkerns]
        print(len(svm_prob), dataset, i)
        del inner_probs

        noCores = 4
        try:
            noCores = mp.cpu_count()
            logging.info("No of Cores: "+str(noCores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(noCores)
        resultsSVM = pool.map(svmThread, svm_prob)
        pool.close()
        pool.join()

        kerns =  ["Gaussian", "Linear", "Quadratic"]
        inner_svm_results = [(c, xker, np.mean([x[3] for x in resultsSVM if x[5] == c and x[6] == xker]), np.mean([x[4] for x in resultsSVM if x[5] == c and x[6] == xker])) for c in C for xker in kerns]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(2, 3))
        best_f = max(inner_svm_results, key=itemgetter(3, 2))

        results_acc.append(test_SVM(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_SVM(best_f, train_x, train_xS, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[2] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[3] for x in results_f]))

    with open('SVM_results_all_'+str(dataset), 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVM_results_acc_'+str(dataset), 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVM_results_f_'+str(dataset), 'wb') as fp:
        pickle.dump(results_f, fp)

def grid_search_KT(C, dataset):

    xkerns = [Linear()]
    xskerns = [Linear()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
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
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svm_prob = [(data[3], data[4], data[2], data[1], data[0], c, c2, xk, xsk, data[5], data[6], 0, 0, 0, 0) for data in inner_probs for c in C for c2 in C for xk in xkerns for xsk in xskerns]
        print(len(svm_prob), dataset, i)
        del inner_probs

        noCores = 4
        try:
            noCores = mp.cpu_count()
            logging.info("No of Cores: "+str(noCores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(noCores)
        resultsSVM = pool.map(ktThread, svm_prob)
        pool.close()
        pool.join()

        kerns =  ["Linear"]
        kerns2 =  ["Linear"]
        inner_svm_results = [(c, c2, xker, xsker, np.mean([x[3] for x in resultsSVM if x[5] == c and x[6] == c2 and x[7] == xker and x[8] == xsker]), np.mean([x[4] for x in resultsSVM if x[5] == c and x[6] == c2 and x[7] == xker and x[8] == xsker])) for c in C for c2 in C for xker in kerns for xsker in kerns2]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(4, 5))
        best_f = max(inner_svm_results, key=itemgetter(5, 4))

        results_acc.append(test_KT(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_KT(best_f, train_x, train_xS, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[4] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[5] for x in results_f]))

    with open('KT_results_all_'+str(dataset)+'_lin_lin', 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('KT_results_acc_'+str(dataset)+'_lin_lin', 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('KT_results_f_'+str(dataset)+'_lin_lin', 'wb') as fp:
        pickle.dump(results_f, fp)

def grid_search_SVMp(C, Gamma, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Linear()]
    xskerns = [Polynomial()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
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
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svmp_prob = [(data[3], data[4], data[2], data[1], data[0], c, g, xk, xsk, data[5], data[6]) for data in inner_probs for c in C for g in Gamma for xk in xkerns for xsk in xskerns]
        print(len(svmp_prob), dataset, i)
        del inner_probs

        noCores = 4
        try:
            noCores = mp.cpu_count()
            logging.info("No of Cores: "+str(noCores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(noCores)
        resultsSVMp = pool.map(svmpThread, svmp_prob)
        pool.close()
        pool.join()

        resultsSVM = sorted(resultsSVMp, key=itemgetter(6,1,2,4,3))

        kerns = ["Linear"]
        kerns2 = ["Quadratic"]
        inner_svm_results = [(c, g, xker, xSk, np.mean([x[3] for x in resultsSVM if x[5] == c and x[6] == g and x[7] == xker and x[8] == xSk]), np.mean([x[4] for x in resultsSVM if x[5] == c and x[6] == g and x[7] == xker and x[8] == xSk])) for c in C for g in Gamma for xker in kerns for xSk in kerns2]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(4, 5))
        best_f = max(inner_svm_results, key=itemgetter(4, 5))

        results_acc.append(test_SVMp(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_SVMp(best_f, train_x, train_xS, train_y, test_x, test_labels))

        #results = results+resultsSVM#resultsSVM+resultsSVMp+resultsSVMdpsa+resultsSVMdp
    print(results_acc)
    print("Accuracy: ", np.mean([x[4] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[5] for x in results_f]))

    with open('SVMp_results_all_'+str(dataset)+'_lin_quad', 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVMp_results_acc_'+str(dataset)+'_lin_quad', 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVMp_results_f_'+str(dataset)+'_lin_quad', 'wb') as fp:
        pickle.dump(results_f, fp)

def grid_search_SVMmt(C, Delta, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Linear()]
    xskerns = [Gaussian()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
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
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svmdps_prob = [(data[3], data[4], data[2], data[1], data[0], c, c2, d, xk, xsk, data[5], data[6]) for data in inner_probs for c in C for c2 in C for d in Delta for xk in xkerns for xsk in xskerns]
        print(len(svmdps_prob), dataset, i)
        del inner_probs

        noCores = 4
        try:
            noCores = mp.cpu_count()
            logging.info("No of Cores: "+str(noCores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(noCores)
        resultsSVMdpsa = pool.map(svmdpsaThread, svmdps_prob)
        pool.close()
        pool.join()

        kerns =  ["Linear"]
        kerns2 =  ["Gaussian"]
        inner_svm_results = [(c, c2, d, xker, xsker, np.mean([x[3] for x in resultsSVMdpsa if x[5] == c and x[6] == c2 and x[7] == d and x[8] == xker and x[9] == xsker]), np.mean([x[4] for x in resultsSVMdpsa if x[5] == c and x[6] == c2 and x[7] == d and x[8] == xker and x[9] == xsker])) for c in C for c2 in C for d in Delta for xker in kerns for xsker in kerns2]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(4, 5))
        best_f = max(inner_svm_results, key=itemgetter(5, 4))

        results_acc.append(test_SVMmt(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_SVMmt(best_f, train_x, train_xS, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[5] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[6] for x in results_f]))

    with open('SVMmt_results_all_'+str(dataset)+"_fixed_delta_lin_gaus_am", 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVMmt_results_acc_'+str(dataset)+"_fixed_delta_lin_gaus_am", 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVMmt_results_f_'+str(dataset)+"_fixed_delta_lin_gaus_am", 'wb') as fp:
        pickle.dump(results_f, fp)

def grid_search_SVMdp(C, Delta, Gamma, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Linear()]
    xskerns = [Gaussian()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
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
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svmdp_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, xk, xsk, data[5], data[6], 0) for data in inner_probs for c in C for d in Delta for g in Gamma for xk in xkerns for xsk in xskerns]
        print(len(svmdp_prob), dataset, i)
        del inner_probs

        noCores = 4
        try:
            noCores = mp.cpu_count()
            logging.info("No of Cores: "+str(noCores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(noCores)
        resultsSVMdp = pool.map(svmdpThread, svmdp_prob)
        pool.close()
        pool.join()

        kerns =  ["Linear"]
        kerns2 =  ["Gaussian"]
        inner_svm_results = [(c, g, d, xker, xsker, np.mean([x[3] for x in resultsSVMdp if x[5] == c and x[6] == d and x[7] == g and x[8] == xker and x[9] == xsker]), np.mean([x[4] for x in resultsSVMdp if x[5] == c and x[6] == d and x[7] == g and x[8] == xker and x[9] == xsker])) for c in C for d in Delta for g in Gamma for xker in kerns for xsker in kerns2]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(5, 6))
        best_f = max(inner_svm_results, key=itemgetter(6, 5))

        results_acc.append(test_SVMdp(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_SVMdp(best_f, train_x, train_xS, train_y, test_x, test_labels))

        #results = results+resultsSVM#resultsSVM+resultsSVMp+resultsSVMdpsa+resultsSVMdp
    print(results_acc)
    print("Accuracy: ", np.mean([x[5] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[6] for x in results_f]))

    with open('SVMdp_results_all_'+str(dataset)+"_fixed_delta_lin_gaus", 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVMdp_results_acc_'+str(dataset)+"_fixed_delta_lin_gaus", 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVMdp_results_f_'+str(dataset)+"_fixed_delta_lin_gaus", 'wb') as fp:
        pickle.dump(results_f, fp)

def grid_search_SVMu(C, Delta, Gamma, Sigma, dataset):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    xkerns = [Gaussian()]
    xskerns = [Polynomial()]
    xsskerns = [Gaussian()]

    results_acc = []
    results_f = []
    results_to_save = []

    for i in range(10):
        prob_data = []
        test_labels = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_labels.npy")
        test_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-test_normal.npy")
        train_y = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_labels.npy").reshape(-1,1)
        train_x = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_normal.npy")
        train_xS = get_array("Data/Dataset"+str(dataset)+"/tech"+str(dataset)+"-0-"+str(i)+"-train_priv.npy")
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
            inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
            inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
            inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
            inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
            inner_test_y = inner_folds_test[j][:, -1:]
            inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])
        del inner_folds_test, inner_folds_test_index, inner_folds_train, inner_folds_train_index, kf, inner_test_x, inner_test_y, inner_x, inner_xS, inner_y, j

        inner_probs = np.array(inner_probs)

        svmu_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, S, xk, xsk, xssk, data[5], data[6]) for data in inner_probs for c in C for d in Delta for g in Gamma for S in Sigma for xk in xkerns for xsk in xskerns for xssk in xsskerns]
        print(len(svmu_prob), dataset, i)
        del inner_probs

        noCores = 4
        try:
            noCores = mp.cpu_count()
            logging.info("No of Cores: "+str(noCores))
        except:
            logging.info("Number of cores couldn't be determined")

        pool = mp.Pool(noCores)
        resultsSVMu = pool.map(svmuThread, svmu_prob)
        pool.close()
        pool.join()

        kerns = ["Gaussian"]
        kerns2 = ["Quadratic"]
        kerns3 = ["Gaussian"]
        inner_svm_results = [(c, g, s, d, xker, xsker, xssker, np.mean([x[3] for x in resultsSVMu if x[5] == c and x[6] == d and x[7] == g and x[8] == s and x[9] == xker and x[10] == xsker and x[11] == xssker]), np.mean([x[4] for x in resultsSVMu if x[5] == c and x[6] == d and x[7] == g and x[8] == s and x[9] == xker and x[10] == xsker and x[11] == xssker])) for c in C for d in Delta for g in Gamma for s in Sigma for xker in kerns for xsker in kerns2 for xssker in kerns3]
        results_to_save.append(inner_svm_results)
        print(inner_svm_results)
        best_acc = max(inner_svm_results, key=itemgetter(7, 8))
        best_f = max(inner_svm_results, key=itemgetter(8, 7))

        results_acc.append(test_SVMu(best_acc, train_x, train_xS, train_y, test_x, test_labels))
        results_f.append(test_SVMu(best_f, train_x, train_xS, train_y, test_x, test_labels))

    print(results_acc)
    print("Accuracy: ", np.mean([x[7] for x in results_acc]))
    print(results_f)
    print("F-Score: ", np.mean([x[8] for x in results_f]))

    with open('SVMu_results_all_'+str(dataset)+"_fixed_delta_gaus_quad_gaus", 'wb') as fp:
        pickle.dump(results_to_save, fp)
    with open('SVMu_results_acc_'+str(dataset)+"_fixed_delta_gaus_quad_gaus", 'wb') as fp:
        pickle.dump(results_acc, fp)
    with open('SVMu_results_f_'+str(dataset)+"_fixed_delta_gaus_quad_gaus", 'wb') as fp:
        pickle.dump(results_f, fp)


def svmThread(p):
    prob = svm_problem_tuple(p)
    logging.info("Entered multicore process")
    svm_tp, svm_fp, svm_fn, svm_tn, svm_avg_time = comp(SVM(), prob, p[3], p[4])
    logging.info("model trained"+" "+str(svm_tp)+" "+str(svm_fp)+" "+str(svm_fn)+" "+str(svm_tn))
    svm_acc = get_accuracy(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_pre = get_precision(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_rec = get_recall(svm_tp, svm_fp, svm_fn, svm_tn)
    svm_fsc = get_fscore(svm_pre, svm_rec)
    return ("SVM", p[7], p[8], svm_acc, svm_fsc, p[5], p[6].getName(), svm_avg_time)

def svmpThread(p):
    prob = svm_problem_tuple(p)
    logging.info("Entered multicore process")
    svmp_tp, svmp_fp, svmp_fn, svmp_tn, svmp_avg_time = comp(SVMp(), prob, p[3], p[4])
    logging.info("model trained")
    svmp_acc = get_accuracy(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_pre = get_precision(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_rec = get_recall(svmp_tp, svmp_fp, svmp_fn, svmp_tn)
    svmp_fsc = get_fscore(svmp_pre, svmp_rec)
    logging.info("Completed")
    return ("SVM+", p[9], p[10], svmp_acc, svmp_fsc, p[5], p[6], p[7].getName(), p[8].getName(), svmp_avg_time)

def svmdpsaThread(p):
    prob = svm_problem_tuple(p)
    logging.info("Entered multicore process")
    svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn, svmdpsa_avg_time = comp(SVMdp_simp(), prob, p[3], p[4])
    logging.info("model trained")
    svmdpsa_acc = get_accuracy(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_pre = get_precision(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_rec = get_recall(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_fsc = get_fscore(svmdpsa_pre, svmdpsa_rec)
    logging.info("Completed")
    return ("SVMd+ - simp", p[10], p[11], svmdpsa_acc, svmdpsa_fsc, p[5], p[6], p[7], p[8].getName(), p[9].getName(), svmdpsa_avg_time)

def svmdpThread(p):
    prob = svm_problem_tuple(p)
    logging.info("Entered multicore process")
    svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn, svmdp_avg_time = comp(SVMdp(), prob, p[3], p[4])
    logging.info("model trained")
    svmdp_acc = get_accuracy(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_pre = get_precision(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_rec = get_recall(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_fsc = get_fscore(svmdp_pre, svmdp_rec)
    logging.info("Completed")
    return ("SVMdp", p[10], p[11], svmdp_acc, svmdp_fsc, p[5], p[6], p[7], p[8].getName(), p[9].getName(), svmdp_avg_time)

def svmuThread(p):
    logging.info("Entered thread")
    prob = svm_u_problem_tuple(p)
    logging.info("Entered multicore process")
    svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn, svmdp_avg_time = comp(SVMu(), prob, p[3], p[4])
    logging.info("model trained")
    svmdp_acc = get_accuracy(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_pre = get_precision(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_rec = get_recall(svmdp_tp, svmdp_fp, svmdp_fn, svmdp_tn)
    svmdp_fsc = get_fscore(svmdp_pre, svmdp_rec)
    logging.info("Completed")
    return ("SVMu", p[12], p[13], svmdp_acc, svmdp_fsc, p[5], p[6], p[7], p[8], p[9].getName(), p[10].getName(), p[11].getName(), svmdp_avg_time)

def ktThread(p):
    prob = svm_problem_tuple(p)
    logging.info("Entered multicore process")
    svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn, svmdpsa_avg_time = comp(KT(), prob, p[3], p[4])
    logging.info("model trained")
    svmdpsa_acc = get_accuracy(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_pre = get_precision(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_rec = get_recall(svmdpsa_tp, svmdpsa_fp, svmdpsa_fn, svmdpsa_tn)
    svmdpsa_fsc = get_fscore(svmdpsa_pre, svmdpsa_rec)
    logging.info("Completed")
    return ("SVMkt - simp", p[9], p[10], svmdpsa_acc, svmdpsa_fsc, p[5], p[6], p[7].getName(), p[8].getName(), svmdpsa_avg_time)


def main():
    #grid_search_SVM([0.001, 0.01, 0.1, 1, 10, 100, 1000], 137) # done
    #grid_search_SVM([0.001, 0.01, 0.1, 1, 10, 100, 1000], 174)
    #grid_search_SVM([0.001, 0.01, 0.1, 1, 10, 100, 1000], 197)
    #grid_search_SVM([0.001, 0.01, 0.1, 1, 10, 100, 1000], 219)
    #grid_search_SVM([0.001, 0.01, 0.1, 1, 10, 100, 1000], 254)
    #grid_search_SVMp([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 137)
    #grid_search_SVMp([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 174)
    #grid_search_SVMp([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 197)
    #grid_search_SVMp([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 219)
    #grid_search_SVMp([0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 254)
    #grid_search_SVMmt([0.001, 0.1, 10, 1000], [1000], 137) # done
    #grid_search_SVMmt([0.001, 0.1, 10, 1000], [1000], 174)
    #grid_search_SVMmt([0.001, 0.1, 10, 1000], [1000], 197)
    #grid_search_SVMmt([0.001, 0.1, 10, 1000], [1000], 219)
    #grid_search_SVMmt([0.001, 0.1, 10, 1000], [1000], 254)
    ##grid_search_SVMdp([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 137)
    ##grid_search_SVMdp([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 174)
    #grid_search_SVMdp([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 197)
    #grid_search_SVMdp([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 219)
    #grid_search_SVMdp([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], 254)
    #grid_search_SVMu([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 137)
    #grid_search_SVMu([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 174)
    #grid_search_SVMu([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 197)
    #grid_search_SVMu([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 219)
    #grid_search_SVMu([0.001, 0.1, 10, 1000], [1000], [0.001, 0.1, 10, 1000], [0.001, 0.1, 10, 1000], 254)
    #grid_search_KT([0.001, 0.1, 10, 1000], 137)



    print("----- Reduced Search Accuracy -----")
    with open ('SVM_results_acc_137_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_137_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_137_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMmt Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_137_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_137_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_137_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_174_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_174_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_174_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMmt Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_174_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_174_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_174_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_197_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_197_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_197_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMmt Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_197_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_197_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_197_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_219_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_219_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_219_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMmt Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_219_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_219_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_219_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_254_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_254_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_254_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMmt Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_254_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_254_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_254_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))


    print("----- Reduced Search F-Score -----")
    with open ('SVM_results_f_137_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_137_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_137_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMmt F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_f_137_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_137_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_137_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_174_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_174_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_174_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMmt F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_f_174_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_174_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_174_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_197_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_197_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_197_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMmt F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_f_197_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_197_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_197_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_219_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_219_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_219_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMmt F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_f_219_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_219_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_219_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_254_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_254_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_254_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMmt F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_f_254_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_254_fixed_delta_reduced_K', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_254_reduced', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))


    print("----- Accuracy -----")
    with open ('SVM_results_acc_137_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_137_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_137_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_137_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_137_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_137_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_174_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_174_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_174_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_174_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_174_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_174_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_197_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_197_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_197_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_197_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_197_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_197_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_219_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_219_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_219_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_219_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_219_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_219_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    with open ('SVM_results_acc_254_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM Accuracy:", np.mean([item[2] for item in itemlistsvm]))
    with open ('SVMp_results_acc_254_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM+ Accuracy:", np.mean([item[4] for item in itemlistsvm]))
    with open ('SVMmt_results_acc_254_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMmt Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMdp_results_acc_254_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMd+ Accuracy:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMu_results_acc_254_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMu Accuracy:", np.mean([item[7] for item in itemlistsvm]))
    with open ('KT_results_acc_254_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 KT Accuracy:", np.mean([item[4] for item in itemlistsvm]))

    print("----- F-Score -----")
    with open ('SVM_results_f_137_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_137_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_137_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMdp_results_f_137_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_137_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_137_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("137 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_174_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_174_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_174_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMdp_results_f_174_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_174_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_174_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("174 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_197_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_197_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_197_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMdp_results_f_197_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_197_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_197_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("197 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_219_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_219_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_219_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMdp_results_f_219_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_219_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_219_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("219 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))

    with open ('SVM_results_f_254_linear', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM F-Score:", np.mean([item[3] for item in itemlistsvm]))
    with open ('SVMp_results_f_254_lin_quad', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVM+ F-Score:", np.mean([item[5] for item in itemlistsvm]))
    with open ('SVMmt_results_f_254_fixed_delta_lin_lin_am', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMmt F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMdp_results_f_254_fixed_delta_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMd+ F-Score:", np.mean([item[6] for item in itemlistsvm]))
    with open ('SVMu_results_f_254_fixed_delta_lin_quad_gaus', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 SVMu F-Score:", np.mean([item[8] for item in itemlistsvm]))
    with open ('KT_results_f_254_lin_lin', 'rb') as fp:
        itemlistsvm = pickle.load(fp)
    print("254 KT F-Score:", np.mean([item[5] for item in itemlistsvm]))


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    #mp.set_start_method('spawn')
    #mp.freeze_support()
    print(mp.get_start_method())
    #mp.context
    main()