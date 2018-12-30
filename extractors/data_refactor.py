

CSV_PATH = "./csv/"
FILE = "koper.csv"


# problem with sdd
# fsc find type and bbox
# ngsim problem still 0 0 0 0 starting bounding box


def extract_trajectories(file_name):

    trajectories = {}

    with open(file_name) as file_:
        for line in file_:
            # print(line)

            line = line.split(",")
            # print(line)

            # for i,l in enumerate(line[:-1]):
            #     print(i,l)

            id_ = line[3]
            # print(id_)
            coordinates = [line[4],line[5]]
            bbox = [line[6],line[7],line[8],line[9]]
            frame = line[2]

            if id_ not in trajectories:

                trajectories[id_] = {
                    "coordinates" : [],
                    "bboxes" : [],
                    "frames" : [],
                    "scene" : line[1],
                    "user_type" : line[10]
                }
            trajectories[id_]["coordinates"].append(coordinates)
            trajectories[id_]["bboxes"].append(bbox)
            trajectories[id_]["frames"].append(frame)
    return trajectories

def main():

    file_path = CSV_PATH + FILE


    # extract trajectories
    trajectories = extract_trajectories(file_path)


    

    # print(trajectories)
    # print("------")
    # print(trajectories["2"])
    # print(trajectories.keys())
    # print(len(trajectories.keys()))


    return

if __name__ == "__main__":
    main()