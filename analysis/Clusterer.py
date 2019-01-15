from sklearn.cluster import KMeans,DBSCAN,SpectralClustering
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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
        elif self.selected_clusterer == 2:
            return self.cluster_dbscan_dtw(features)
        elif self.selected_clusterer == 3:
            return self.cluster_spectral_dtw(features)

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

    def cluster_dbscan_dtw(self,distance_matrix):
        
        cl = DBSCAN(eps= self.eps, min_samples= self.min_samples,metric="precomputed")  
        clusters = cl.fit_predict(distance_matrix)        

        return clusters

    def cluster_spectral_dtw(self,distance_matrix):
        
        cl = SpectralClustering(n_clusters = self.nb_clusters,affinity="precomputed")  
        clusters = cl.fit_predict(distance_matrix)        

        return clusters
    