import libemg
from libemg.utils import make_regex
from utils import prepare_data
import numpy as np
import matplotlib.pyplot as plt

# some parameters/ dataset details:
window_size = 200
window_increment = 100
num_subjects = 12

feature_parameters = {"WAMP_threshold":1e-6}

if __name__ == "__main__":
    dataset_dir = 'dataset/'
    # prepare -> get it into a format that libemg can consume it -- 
    # a collection of .csv files
    prepare_data(dataset_dir)

    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = fe.get_feature_list()
    feature_group_list = fe.get_feature_groups()
    om = libemg.offline_metrics.OfflineMetrics()
    metrics = ["CA"]

    results = np.zeros((num_subjects, len(feature_list)+len(feature_group_list)))
    
    # 12 subjects
    reps_values = [str(r) for r in range(22)]
    classes_values = [str(c) for c in range(9)]
    filename_dic = {
            "reps": reps_values,
            "reps_regex": make_regex(left_bound="R", right_bound=".csv", values=reps_values),
            "classes": classes_values,
            "classes_regex": make_regex(left_bound="/C", right_bound="R", values=classes_values),
        }
    for s in range(num_subjects):
        filename_dic["subjects"] = [str(s+1)]
        filename_dic["subjects_regex"]  = make_regex(left_bound="s", right_bound="/C", values=filename_dic["subjects"])
        odh = libemg.data_handler.OfflineDataHandler()
        odh.get_data(dataset_dir, filename_dic)
        train_odh = odh.isolate_data("reps",list(range(20)))
        test_odh = odh.isolate_data("reps", list(range(20,22)))
        train_windows, train_metadata = train_odh.parse_windows(window_size, window_increment)
        test_windows,  test_metadata  = test_odh.parse_windows(window_size, window_increment)

        # get all features
        for f in range(len(feature_list)+len(feature_group_list)):
            if f < len(feature_list):
                feature = feature_list[f]
                train_features = fe.extract_features([feature], train_windows,feature_parameters)
                test_features = fe.extract_features([feature], test_windows,feature_parameters)
            else:
                feature = list(feature_group_list.keys())[f-len(feature_list)]
                train_features = fe.extract_feature_group(feature, train_windows,feature_parameters)
                test_features = fe.extract_feature_group(feature, test_windows,feature_parameters)
            feature_dictionary = {
                "training_features": train_features,
                "training_labels": train_metadata["classes"]
            }
           
            # train classifier
            clf = libemg.emg_classifier.EMGClassifier()
            clf.fit("LDA", feature_dictionary.copy())

            preds = clf.run(test_features, test_metadata["classes"])
            
            # test classifier
            results[s,f] = om.extract_offline_metrics(metrics, test_metadata["classes"], preds[0])[metrics[0]] * 100
            print("Subject: {}, Feature: {}, Accuracy: {}".format(s+1, feature, results[s,f]))
    
    np.save("results.npy", results)
    mean_feature_accuracy = results.mean(axis=0)
    std_feature_accuracy  = results.std(axis=0)


    plt.bar(feature_list, mean_feature_accuracy, yerr=std_feature_accuracy)
    plt.grid()
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.xticks(feature_list+list(feature_group_list.keys()), rotation=90)

    plt.tight_layout()
    plt.savefig("results.png")
