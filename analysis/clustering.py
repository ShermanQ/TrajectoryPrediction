import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.helpers import get_dir_names,extract_frames,bb_intersection_over_union,extract_trajectories
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler

ROOT = "./../"
CSV = ROOT + "extractors/csv/"

def extract_first_last_point(trajectories):
    features = []
    for trajectory in trajectories:
        first_point = trajectory[0]
        last_point = trajectory[-1]
        features.append([first_point[0],first_point[1],last_point[0],last_point[1]])

    return features

def trajectories_clustering_feature_based(features,n_clusters = 10, random_state = 0):
    
    cl = KMeans(n_clusters= n_clusters, random_state= random_state)
  
    # std = StandardScaler()
    std = MinMaxScaler()

    features_std = std.fit_transform(features)

    clusters = cl.fit_predict(features_std)
    

    return clusters

def multi_step_clustering(file_path,features_extraction_functions,clustering_functions):
    trajectories = extract_trajectories(file_path)
    multistep_clusters = []
    multistep_clusters.append({})
    # clusters = {}
    multistep_clusters[0]["0"] = [key for key in trajectories]


    for feature_extractor,clustering_function in zip(features_extraction_functions,clustering_functions):
        print(feature_extractor)

        # feature_extractor = features_extraction_functions[step]
        # clustering_function = clustering_functions[step]
        

        clusters = multistep_clusters[-1]

        cluster_nb = 0
        new_clusters = {}
        for key in clusters:
            trajectories_coordinates = []
            for id_ in clusters[key]:
                trajectories_coordinates.append(trajectories[id_]["coordinates"])
            features = feature_extractor(trajectories_coordinates)
            trajectories_label = clustering_function(features)

            sub_clusters = {}

            for i,label in enumerate(trajectories_label):
                if label not in sub_clusters:
                    sub_clusters[label] = []
                sub_clusters[label].append(clusters[key][i])
            for key1 in sub_clusters:
                new_clusters[cluster_nb] = sub_clusters[key1]
                cluster_nb += 1
        multistep_clusters.append(new_clusters)
    return multistep_clusters

def main():


    file_path = CSV + "new_rates/deathCircle1_30.0to2.5.csv"
    

    clustering_functions = [trajectories_clustering_feature_based,trajectories_clustering_feature_based]
    features_extraction_functions = [extract_first_last_point,extract_first_last_point]
    multistep_clusters = multi_step_clustering(file_path,features_extraction_functions,clustering_functions)


    print(len(multistep_clusters))
    print(sorted(multistep_clusters[2][0]))



if __name__ == "__main__":
    main()