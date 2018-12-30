
import helpers

FILE_PATH = "./csv/quad0.csv"
FRAMES = 1


# fsc find type and bbox
# ngsim problem still 0 0 0 0 starting bounding box


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