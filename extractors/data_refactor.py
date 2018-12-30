
import helpers

FILE_PATH = "./csv/quad0.csv"
FRAMES = 1


# fsc find type and bbox --> nothing that I can do except set a fixed size bounding box / it seems that x and y are from the closest object point from the robot




def main():
    frames = []
    trajectories = []
    if FRAMES:
        frames = helpers.extract_frames(FILE_PATH)
    else:
        trajectories = helpers.extract_trajectories(FILE_PATH)

    print(frames.keys())
if __name__ == "__main__":
    main()