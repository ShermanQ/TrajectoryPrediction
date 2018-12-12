import cPickle

with (open("alberta_cam_original_2017-10-27_09-00-10_trajectories.cpkl", "rb")) as openfile:
    objects = cPickle.load(openfile)

   

print(len(objects))
for o in objects[290:300]:
    print(o)