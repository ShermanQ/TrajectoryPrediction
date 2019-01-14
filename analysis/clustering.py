import numpy as np 

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extractors.helpers as helpers
from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import cv2
import helpers_v as vis
import scipy
import pandas as pd

ROOT = "./../"
CSV = ROOT + "extractors/csv/"

class Extractor:
    def __init__(self,selected_extractor = 0):
        self.selected_extractor = selected_extractor
        
    def extract(self,trajectories):
        if self.selected_extractor == 0:
            return self.extract_first_last(trajectories)
        elif self.selected_extractor == 1:
            return self.extract_first(trajectories)
        elif self.selected_extractor == 2:
            return self.extract_last(trajectories)
        elif self.selected_extractor == 3:
            return self.extract_speed(trajectories)
        elif self.selected_extractor == 4:
            return self.extract_positions(trajectories)

    def extract_first_last(self,trajectories):
        features = []
        for trajectory in trajectories:
            first_point = trajectory[0]
            last_point = trajectory[-1]
            features.append([first_point[0],first_point[1],last_point[0],last_point[1]])
        return features
    def extract_first(self,trajectories):
        features = []
        for trajectory in trajectories:
            first_point = trajectory[0]
            features.append([first_point[0],first_point[1]])
        return features
    def extract_last(self,trajectories):
        features = []
        for trajectory in trajectories:
            last_point = trajectory[-1]
            features.append([last_point[0],last_point[1]])
        return features

    def extract_speed(self,trajectories):
        features = []
        for trajectory in trajectories:
            speed = [scipy.spatial.distance.euclidean(trajectory[i-1],trajectory[i]) for i in range(1,len(trajectory))]
            stats = self.get_stats(speed)
            features.append(stats)
        return features

    def extract_positions(self,trajectories):
        features = []
        for trajectory in trajectories:
            x = [p[0] for p in trajectory]
            y = [p[1] for p in trajectory]
            stats_x = self.get_stats(x)
            stats_y = self.get_stats(y)
            stats = stats_x + stats_y
            features.append(stats)
        return features
    
    def get_stats(self,sequence):
        stats = pd.Series(sequence).describe().values.tolist()
        skewness = scipy.stats.skew(sequence)
        kurtosis = scipy.stats.kurtosis(sequence)
        stats.append(skewness)
        stats.append(kurtosis)
        return stats
            


class Clusterer:
    def __init__(self,eps = 0.5,min_samples = 5, selected_clusterer = 0,nb_clusters = 20 ,random_state = 0):
        self.nb_clusters = nb_clusters
        self.random_state = random_state
        self.selected_clusterer = selected_clusterer
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self,features):
        if self.selected_clusterer == 0:
            return self.cluster_kmeans(features)
        elif self.selected_clusterer == 1:
            return self.cluster_dbscan(features)

    def cluster_kmeans(self,features,std = StandardScaler()):
        
        cl = KMeans(n_clusters= self.nb_clusters, random_state= self.random_state)    

        if std != None:
            features = std.fit_transform(features)
        clusters = cl.fit_predict(features)        

        return clusters

    def cluster_dbscan(self,features,std = StandardScaler()):
        
        cl = DBSCAN(eps= self.eps, min_samples= self.min_samples)  

        if std != None:
            features = std.fit_transform(features)  
        clusters = cl.fit_predict(features)        

        return clusters

def multi_step_clustering(trajectories,extractors,clusterers):
    # trajectories = extract_trajectories(file_path)
    multistep_clusters = []
    multistep_clusters.append({})
    multistep_clusters[0][0] = [key for key in trajectories] ##########################################################################################################

    clusters_hierarchy = []

    for extractor,clusterer in zip(extractors,clusterers):
       
        clusters = multistep_clusters[-1]
        cluster_hierarchy = []

        cluster_nb = 0
        new_clusters = {}
        for key in clusters:

            sub_cluster_hierarchy = []
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
                sub_cluster_hierarchy.append(cluster_nb)
                cluster_nb += 1

            cluster_hierarchy.append(sub_cluster_hierarchy)
        clusters_hierarchy.append(cluster_hierarchy)

        multistep_clusters.append(new_clusters)
    return multistep_clusters,clusters_hierarchy

def display_clusters(trajectories,clusters,img,factor_div, nb_columns = 8, mosaic = True,save = False):
    
    nb_clusters = len(clusters.keys())
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

            cv2.circle(img1,tuple([int(p) for p in points[0]]), 5, (0,255,0), -1)
            cv2.circle(img1,tuple([int(p) for p in points[-1]]), 5, (0,0,255), -1)

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
        if save:
            return mosaic
        cv2.imshow('image1',mosaic)
    else:
        if save:
            return img1
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

def display_multi_step_clustering(multistep_clusters,img,trajectories,factor_div,save = False):
    images = []
    for clusters in multistep_clusters:
        img1 = display_clusters(trajectories,clusters,img,factor_div, mosaic = False,save = True)
        img1 = cv2.resize(img1, (0, 0), None, 0.5, 0.5)
        images.append(img1)
    img = np.hstack(tuple(images))
    if save:
        return img
    cv2.imshow('image1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_parent_children(trajectories,two_step_clusters,last_step_hierarchy,parent_id,img,factor_div,save = False):

    parent_ids = two_step_clusters[0][parent_id]
    children_clusters_ids = last_step_hierarchy[parent_id]

    current_id = 0
    clusters = {}

    clusters[current_id] = parent_ids
    current_id += 1

    for child_cl in children_clusters_ids:
        child_ids = two_step_clusters[1][child_cl]
        clusters[current_id] = child_ids
        current_id += 1

    img = display_clusters(trajectories,clusters,img,factor_div, mosaic = True,save = True)

    if save:
        return img

    cv2.imshow('image1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():


    # file_path = CSV + "new_rates/bad_30.0to2.5.csv"
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
    

    clusterers = [Clusterer(selected_clusterer= 1),Clusterer(selected_clusterer= 1 )]
    extractors = [Extractor(4),Extractor(3)]

    # clusterers = [Clusterer(selected_clusterer= 1)]
    # extractors = [Extractor(4)]

    multistep_clusters,cluster_hierarchy = multi_step_clustering(trajectories,extractors,clusterers)


    display_clusters(trajectories,multistep_clusters[1],img,factor_div=2.0, mosaic= True)

    display_parent_children(trajectories,multistep_clusters[1:],cluster_hierarchy[-1],0,img,2.0)

    # display_multi_step_clustering(multistep_clusters,img,trajectories,2.0)



if __name__ == "__main__":
    main()