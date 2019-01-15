import scipy
import pandas as pd
import fastdtw
import numpy as np 

class Extractor:
    def __init__(self,selected_extractor = 0,gamma = -1):
        self.selected_extractor = selected_extractor
        self.gamma = gamma
        
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
        elif self.selected_extractor == 5:
            return self.extract_dtw(trajectories)

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

    # if gamma is -1, returns the distance matrix, else compute the affinity matrix
    def extract_dtw(self,trajectories):
        d = np.zeros(shape=(len(trajectories),len(trajectories)))

        for i in range(len(trajectories)):
            for j in range(i ,len(trajectories)):
                trajectory1 = np.array(trajectories[i])
                trajectory2 = np.array(trajectories[j])
                distance, _ = fastdtw.fastdtw(trajectory1,trajectory2, dist = scipy.spatial.distance.euclidean)

                if self.gamma + 1 > 0:
                    distance = np.exp(-self.gamma * distance ** 2)


                d[i,j] = distance
                d[j,i] = distance

        return d
    
    def get_stats(self,sequence):
        stats = pd.Series(sequence).describe().values.tolist()
        skewness = scipy.stats.skew(sequence)
        kurtosis = scipy.stats.kurtosis(sequence)
        stats.append(skewness)
        stats.append(kurtosis)
        return stats
            