import csv 
import cv2
import numpy as np
from skimage.transform import ProjectiveTransform
import matplotlib.pyplot as plt

def main():

    h,w = 1000,1000

    TRAJECTORIES_PATH = "/home/laurent/Documents/master/extractors/datasets/bad/trajectories.csv"
    with open(TRAJECTORIES_PATH) as csv_file:
        img = np.zeros((h,w,3), np.uint8)
        traj_reader = csv.reader(csv_file, delimiter=',')
        points = []
        for i,row in enumerate(traj_reader):
            
            point = [float(row[0]),float(row[1])]
            points.append(point)

            
        plt.scatter([p[0] for p in points],[p[1] for p in points])
        plt.show()
 

if __name__ == "__main__":
    main()




# homography = np.loadtxt("/home/laurent/Documents/master/extractors/datasets/bad/homography/homography.txt")
# transformer = ProjectiveTransform(matrix = homography)

    # print(points)
    # points = [float(row[0])*w,float(row[1])*h]
    # points = transformer.inverse(points)[0]
#     type_ = row[2]
#     if type_ == "car":
#         cv2.circle(img,tuple([int(p) for p in points]), 2, (0,0,255), -1)

        # cv2.imshow('image1',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()