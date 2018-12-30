
import helpers

FILE_PATH = "./csv/koper.csv"
FRAMES = 1


# problem with sdd
# fsc find type and bbox
# ngsim problem still 0 0 0 0 starting bounding box


def main():
    frames = []
    trajectories = []
    if FRAMES:
        frames = helpers.extract_frames(FILE_PATH)
    else:
        trajectories = helpers.extract_trajectories(FILE_PATH)


if __name__ == "__main__":
    main()