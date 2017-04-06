from Container_Classes import *

class Model_Manager():
    def train(self, problem : svm_problem):


    def grid_search(C, Delta, Gamma, dataset):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

        xkerns = [Gaussian(), Polynomial(), Linear()]
        xskerns = [Gaussian(), Polynomial(), Linear()]

        results = []

        for i in range(3):
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

            inner_probs = []
            for j in range(5):
                inner_x = inner_folds_train[j][:,0:train_x.shape[1]]
                inner_xS = inner_folds_train[j][:, train_x.shape[1]:train_x.shape[1]+train_xS.shape[1]]
                inner_y = inner_folds_train[j][:, train_x.shape[1]+train_xS.shape[1]:train_x.shape[1]+train_xS.shape[1]+train_y.shape[1]]
                inner_test_x = inner_folds_test[j][:, :test_x.shape[1]]#test_x.shape[1]:test_x.shape[1]+train_xS.shape[1]]
                inner_test_y = inner_folds_test[j][:, -1:]
                inner_probs.append([inner_test_y, inner_test_x, inner_y, inner_x, inner_xS, i, j])

            inner_probs = np.array(inner_probs)

            svm_prob = [(data[3], data[4], data[2], data[1], data[0], c, xk, data[5], data[6]) for data in inner_probs for c in C for xk in xkerns]
            svmp_prob = [(data[3], data[4], data[2], data[1], data[0], c, g, xk, xsk, data[5], data[6]) for data in inner_probs for c in C for g in Gamma for xk in xkerns for xsk in xskerns]
            svmdps_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, xk, xsk, data[5], data[6], 0) for data in inner_probs for c in C for d in Delta for xk in xkerns for xsk in xskerns]
            svmdp_prob = [(data[3], data[4], data[2], data[1], data[0], c, d, g, xk, xsk, data[5], data[6], 0) for data in inner_probs for c in C for d in Delta for g in Gamma for xk in xkerns for xsk in xskerns]

            print(len(svmp_prob))

            noCores = 4
            try:
                noCores = cpu_count()
                logging.info("No of Cores: "+str(noCores))
            except:
                logging.info("Number of cores couldn't be determined")

            pool = ThreadPool(cpu_count())

            #resultsSVM = pool.map(svmThread, svm_prob)
            resultsSVMp = pool.map(svmpThread, svmp_prob)
            #resultsSVMdpsa = pool.map(svmdpsaThread, svmdps_prob)
            #resultsSVMdp = pool.map(svmdpThread, svmdp_prob)

            pool.close()
            pool.join()
            results = results+resultsSVMp#resultsSVM+resultsSVMp+resultsSVMdpsa+resultsSVMdp

        #with open('SVMPLUSoutfile', 'wb') as fp:
        #    pickle.dump(results, fp)