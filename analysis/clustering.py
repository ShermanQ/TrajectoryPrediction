import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.helpers import get_dir_names,extract_frames,bb_intersection_over_union,extract_trajectories
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = "./../"
CSV = ROOT + "extractors/csv/"

def extract_first_last_point(trajectories):
    features = []
    for trajectory in trajectories:
        first_point = trajectory[0]
        last_point = trajectory[-1]
        features.append([first_point[0],first_point[1],last_point[0],last_point[1]])

    return features

def trajectories_clustering_feature_based(features,n_clusters = 20, random_state = 0):
    cl = KMeans(n_clusters= n_clusters, random_state= random_state)
    std = StandardScaler()
    features_std = std.fit_transform(features)
    clusters = cl.fit_predict(features_std)
    return clusters


def main():


    file_path = CSV + "new_rates/deathCircle1_30.0to2.5.csv"
    trajectories = extract_trajectories(file_path)

    
    multistep_clusters = []
    multistep_clusters.append({})
    # clusters = {}
    multistep_clusters[0]["0"] = [key for key in trajectories]

    clusters = multistep_clusters[-1]
    for key in clusters:
        trajectories_coordinates = []
        for id_ in clusters[key]:
            trajectories_coordinates.append(trajectories[id_]["coordinates"])
        features = extract_first_last_point(trajectories_coordinates)
        trajectories_label = trajectories_clustering_feature_based(features)

        sub_clusters = {}

        for i,label in enumerate(trajectories_label):
            if label not in sub_clusters:
                sub_clusters[label] = []
            sub_clusters[label].append(clusters[key][i])
        print(sorted(sub_clusters.keys()))




if __name__ == "__main__":
    main()