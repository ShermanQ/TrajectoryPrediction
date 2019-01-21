import cv2
from skimage import io
import matplotlib.pyplot as plt

from skimage.viewer import ImageViewer
from skimage.transform import ProjectiveTransform
import numpy as np
import math
# points_img = np.array([
#     [874,716],
#     [1257,556],
#     [1266,44],
#     [1200,328],
#     [1072,303],
#     [441,362],
#     [286,420],
#     [368,535]
# ])

# points_sim = np.array([
#     [200,0],
#     [350,100],
#     [350,200],
#     [300,300],
#     [100,300],
#     [0,200],
#     [0,100],
#     [100,0]
# ])

points_img = np.array([
    [4,645],[0,717],[297,719],
    [878,719],[500,636],[236,582],[950,625],[654,565],
    [468,510],[674,548],[991,602],[1117,567],[760,512],
    [502,474],[496,473],[504,464],[643,427],[568,416],
    [444,446],[530,410],[409,443],[251,416],[377,385],
    [451,368],[659,389],[737,395],[771,388],[827,385],
    [967,404],[990,391],[840,379],[917,353],[1035,365],
    [1200,367],[1155,410],[1156,425],[1177,429],[1194,438],
    [1265,445],[1264,495],[1177,487],[1260,554],[1143,560],
    [1136,566],[878,444],[95,628]
])
 
points_sim = np.array([
    [-6.,-16.5],[-3.,-17.5],[0.,-15.5],
    [6.,-14.],[0.,-14.],[-6.,-14.],[6.,-10.],[0.,-10.],
    [-6.,-9.],[0.,-9.],[6.,-9.],[6.,-9.],[0.,-9.],
    [-6.,-7.],[-6.5,-6.5],[-7.,-6.],[-7.,0.],[-9.,0.],
    [-9.,-6.],[-10.,0.],[-10.,-6.],[-14.,-6.],[-14.,0.],
    [-14.,6.],[-9.,6.],[-7.,6.],[-6.5,6.5],[-6.,7.],
    [0.,7.],[0.,9.],[-6.,9.],[-6.,15.5],[0.,15.5],
    [6.,15.5],[6.,9.],[6.,7.],[6.5,6.5],[7.,6.],
    [9.0,6.],[9.,0.],[7.,0.],[9.,-6.],[7.,-6.],
    [6.5,-6.5],[0.,0.],[-6.,-15.]
])

center = [878,444]
points_img_centered = [ ]
for p in points_img:
    new_p = np.subtract(p,center).tolist()
    new_p[1] *= -1.
    points_img_centered.append(new_p)
points_img_centered = np.array(points_img_centered )
# print(points_img_centered)
# # plt.scatter([p[0] for p in points_sim],[p[1] for p in points_sim])
# plt.scatter([p[0] for p in points_img_centered],[p[1] for p in points_img_centered])
# plt.show()


tr = ProjectiveTransform()

print(tr.estimate(points_sim,points_img_centered))

print(tr.params)

np.savetxt("./extractors/datasets/bad/homography/homography.txt",tr.params)

