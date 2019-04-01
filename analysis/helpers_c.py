import numpy as np 
import cv2

# import fastdtw

def multi_step_clustering(trajectories,extractors,clusterers):
    # trajectories = extract_trajectories(file_path)
    multistep_clusters = []
    multistep_clusters.append({})
    multistep_clusters[0][0] = [key for key in trajectories] 
    clusters_hierarchy = []

    for extractor,clusterer in zip(extractors,clusterers):
       
        clusters = multistep_clusters[-1]
        cluster_hierarchy = []

        cluster_nb = 0
        new_clusters = {}
        for key in clusters:

            sub_cluster_hierarchy = []
            trajectories_coordinates = get_coordinates(trajectories,clusters[key])
            # print(len(trajectories_coordinates))
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

def display_clusters(trajectories,clusters,img,offset,factor_div, nb_columns = 8, mosaic = True,save = False):
    
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
        trajectories_coordinates = scale_coordinates(trajectories_coordinates,offset,factor_div)
        


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

def scale_coordinates(trajectories_coordinates,offset,factor_div):
    scaled_trajectories = []
    for coordinates in trajectories_coordinates:
        scaled_coordinates = [p/factor_div for p in coordinates]
        offset_coordinates = np.add(scaled_coordinates,offset).tolist()

        scaled_trajectories.append(offset_coordinates)
    return scaled_trajectories

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

def display_multi_step_clustering(multistep_clusters,img,trajectories,offset,factor_div,save = False):
    images = []
    for clusters in multistep_clusters:
        img1 = display_clusters(trajectories,clusters,img,offset,factor_div, mosaic = False,save = True)
        img1 = cv2.resize(img1, (0, 0), None, 0.5, 0.5)
        images.append(img1)
    img = np.hstack(tuple(images))
    if save:
        return img
    cv2.imshow('image1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_parent_children(trajectories,two_step_clusters,last_step_hierarchy,parent_id,img,offset,factor_div,save = False):

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

    img = display_clusters(trajectories,clusters,img,offset,factor_div, mosaic = True,save = True)

    if save:
        return img

    cv2.imshow('image1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()