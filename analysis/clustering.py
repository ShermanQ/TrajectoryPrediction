import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extractors.helpers as helpers
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import cv2
import helpers_v as vis


ROOT = "./../"
CSV = ROOT + "extractors/csv/"

class Extractor:
    def __init__(self,selected_extractor = 0):
        self.selected_extractor = selected_extractor
        
    def extract(self,trajectories):
        if self.selected_extractor == 0:
            return self.extract_first_last(trajectories)

    def extract_first_last(self,trajectories):
        features = []
        for trajectory in trajectories:
            first_point = trajectory[0]
            last_point = trajectory[-1]
            features.append([first_point[0],first_point[1],last_point[0],last_point[1]])
        return features

class Clusterer:
    def __init__(self,selected_clusterer = 0,nb_clusters = 20 ,random_state = 0):
        self.nb_clusters = nb_clusters
        self.random_state = random_state
        self.selected_clusterer = selected_clusterer

    def cluster(self,features):
        if self.selected_clusterer == 0:
            return self.cluster_kmeans(features)

    def cluster_kmeans(self,features):
        
        cl = KMeans(n_clusters= self.nb_clusters, random_state= self.random_state)    
        # std = StandardScaler()
        std = MinMaxScaler()
        features_std = std.fit_transform(features)
        clusters = cl.fit_predict(features_std)        

        return clusters

def multi_step_clustering(trajectories,extractors,clusterers):
    # trajectories = extract_trajectories(file_path)
    multistep_clusters = []
    multistep_clusters.append({})
    multistep_clusters[0]["0"] = [key for key in trajectories]


    for extractor,clusterer in zip(extractors,clusterers):
       
        clusters = multistep_clusters[-1]

        cluster_nb = 0
        new_clusters = {}
        for key in clusters:
            trajectories_coordinates = get_coordinates(trajectories,clusters[key])
            
            features = extractor.extract(trajectories_coordinates)
            trajectories_label = clusterer.cluster(features)

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

def display_clusters(trajectories,clusters,img,factor_div, nb_columns = 8, mosaic = True):
    
    nb_clusters = len(clusters.keys())
    print(nb_clusters)
    # get one color for each cluster
    colors = get_random_colors(nb_clusters)
    
    # storages for each cluster image
    lines = []
    line = []
    img1 = img.copy()
    for i,cluster in enumerate(clusters):
        if mosaic:
            img1 = img.copy()
        ids = clusters[cluster]

        # get the coordinates from the trajectories of the cluster
        trajectories_coordinates = get_coordinates(trajectories,ids)
        # scale those coordinates according to factor div, for visualisation purpose
        trajectories_coordinates = scale_coordinates(trajectories_coordinates,factor_div)

        # draw every trajectory
        for points in trajectories_coordinates:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img1,[pts],False,colors[i])

        if mosaic:
            # resize cluster image to one quarter of its original size
            img1 = cv2.resize(img1, (0, 0), None, 0.25, 0.25)

            # if the end of the line is reached, create a new line
            if i % nb_columns == 0 and i != 0:  

                lines.append(np.hstack(tuple(line)))
                line = []
            line.append(img1)
        
    if mosaic:
        # if line is not empty, fill line with empty images to match previous line size
        if len(line)>0:
            while len(line) < nb_columns:
                line.append(cv2.resize(img, (0, 0), None, 0.25, 0.25))

        lines.append(np.hstack(tuple(line)))        
        # stack lines vertically
        mosaic = np.vstack(tuple(lines))
        cv2.imshow('image1',mosaic)
    else:
        cv2.imshow('image1',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale_coordinates(trajectories_coordinates,factor_div):
    return [[[p[0]/factor_div, p[1]/factor_div]  for p in t ] for t in trajectories_coordinates ]

def get_random_colors(nb):
    colors = []
    color = tuple([int(r) for r in np.random.randint(low = 1, high = 255, size = 3 )])
    colors.append(color)

    for _ in range(1,nb):
        new_color = tuple([int(r) for r in np.random.randint(low = 1, high = 255, size = 3 )])
        while new_color in colors:
            new_color = tuple([int(r) for r in np.random.randint(low = 1, high = 255, size = 3 )])
        colors.append(new_color)
    return colors

def get_coordinates(trajectories,ids):
    trajectories_coordinates = []
    for id_ in ids:
        trajectories_coordinates.append(trajectories[id_]["coordinates"])
    return trajectories_coordinates

def main():


    file_path = CSV + "new_rates/deathCircle1_30.0to2.5.csv"
    
    # print(sorted(multistep_clusters[2][0]))

    ##############################
    temp_path = "./temp.txt"
    helpers.extract_frames(file_path,temp_path,save = True)
    min_,max_ = vis.find_bounding_coordinates(temp_path)

    w,h = vis.get_scene_image_size(min_,max_,factor_div = 2.0)
    img = np.zeros((h,w,3), np.uint8)
    os.remove(temp_path)
    ###############################
    trajectories = helpers.extract_trajectories(file_path)
    

    clusterers = [Clusterer(selected_clusterer= 0,nb_clusters= 20),Clusterer(nb_clusters= 5, selected_clusterer= 0 )]
    extractors = [Extractor(0),Extractor(0)]
    multistep_clusters = multi_step_clustering(trajectories,extractors,clusterers)
    display_clusters(trajectories,multistep_clusters[1],img,factor_div=2.0, mosaic= True)



if __name__ == "__main__":
    main()