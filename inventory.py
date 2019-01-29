import extractors.helpers as helpers
import json

def main():
    scenes_selected_path = "./parameters/scenes.json"
    files_path = "./data/csv/"
    save_path = "./data/datasets/inventory.json"
    save_path1 = "./data/datasets/inventory1.json"
    with open(scenes_selected_path) as json_file:
        scenes = json.load(json_file)
        
        datas = {}
        datasets = {}

        for dataset in scenes:
            datasets[dataset] = {"total" : 0}
            for scene in scenes[dataset]:
                datas[scene] = {"total" : 0}
                path = files_path + scene + ".csv"
                trajectories = helpers.extract_trajectories(path)
                for id_ in trajectories:
                    type_ = trajectories[id_]["user_type"]
                    datas[scene]["total"] += 1
                    datasets[dataset]["total"] += 1

                    if type_ in datas[scene]:
                        datas[scene][type_] += 1
                    else:
                        datas[scene][type_] = 0

                    if type_ in datasets[dataset]:
                        datasets[dataset][type_] += 1
                    else:
                        datasets[dataset][type_] = 0

    with open(save_path,"w") as save_path:
        json.dump(datas,save_path,indent=2)
    with open(save_path1,"w") as save_path1:
        json.dump(datasets,save_path1,indent=2)


if __name__ == "__main__":
    main()

          
       