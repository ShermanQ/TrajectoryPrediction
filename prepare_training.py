import csv
from itertools import islice
import extractors.helpers as helpers 
import json

def main():
    parameters = {
        "original_file" : "./data/csv/lankershim_inter2.csv",
        "frames_temp" : "./data/temp/frames.txt",
        "trajectories_temp" : "./data/temp/trajectories.txt"

    }
    
    helpers.extract_frames(parameters["original_file"],parameters["frames_temp"],save = True)
    helpers.extract_trajectories(parameters["original_file"],parameters["trajectories_temp"],save = True)

    with open(parameters["trajectories_temp"]) as trajectories:
        for trajectory in trajectories:
            trajectory = json.loads(trajectory)
            frames = trajectory["frames"]
            current_id = str(trajectory["id"])
            start,stop = frames[0],frames[-1] + 1
            with open(parameters["frames_temp"]) as frames:
                main_traj = []
                for frame in islice(frames,start,stop):
                    frame = json.loads(frame)
                    main_traj.append(frame[current_id]["coordinates"])
                    
if __name__ == "__main__":
    main()