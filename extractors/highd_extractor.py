import csv
import helpers
import sys
import json




# TODO bounding box
class HighdExtractor():
    def __init__(self, data_path,param_path):
        
        'Initializing parameters'
        data = json.load(open(data_path))
        param = json.load(open(param_path))

        self.dataset = param["dataset"]
        self.original_dataset_tracks = data["original_datasets"] + param["tracks_path"]
        self.original_dataset_meta = data["original_datasets"] + param["meta_path"]
        self.original_scene_tracks = data["original_datasets"] + param["tracks_path"] +"{}"
        self.original_scene_meta = data["original_datasets"] + param["meta_path"] +"{}"


        self.destination_file = data["extracted_datasets"] + "{}"
    
    def extract(self):
        csv_files = helpers.get_dir_names(self.original_dataset_tracks)
        csv_meta = helpers.get_dir_names(self.original_dataset_meta,lower = False)
            
        for csv_,meta in zip(csv_files,csv_meta):
            print("processing scene: " + csv_)
            with open(self.original_scene_tracks.format(csv_) ) as csv_reader:
                csv_reader = csv.reader(csv_reader, delimiter=',')

                with open(self.original_scene_meta.format(meta)) as meta_reader:
                    meta_reader = csv.reader(meta_reader, delimiter=',')
                
                    scene_file = self.destination_file.format( csv_)
                    helpers.remove_file(scene_file)


                    with open(scene_file,"a") as scene_csv:
                        subscene_writer = csv.writer(scene_csv)

                        row_meta = next(meta_reader)
                        last_id = "-2"
                        new_scene_name = csv_.split(".")[0]
                        for i,row in enumerate(csv_reader):
                            if i != 0:
                                new_id = row[1]
                                if new_id != last_id:
                                    try:
                                        row_meta = next(meta_reader)
                                    except:
                                        "no row available"
                                
                                new_row = self.__parse_row(row,row_meta,new_scene_name)

                                subscene_writer.writerow(new_row)
                                last_id = new_id
    def __parse_row(self,row,row_meta,new_scene_name):
        
        unknown = -10000
        new_row = []
        new_row.append(self.dataset) #dataset
        new_row.append(new_scene_name) #scene
        new_row.append(int(row[0])) # frame
        new_row.append(int(row[1])) # id
        new_row.append(float(row[2])) #x
        new_row.append(float(row[3]))  #y

        new_row.append(unknown) #xl
        new_row.append(unknown) #yl
        new_row.append(unknown) #xb
        new_row.append(unknown) #yb

        new_row.append(row_meta[6].lower()) # type
        return new_row
        
# python highd_extractor.py parameters/data.json parameters/highd_extractor.json

def main():
    args = sys.argv
    highd_extractor = HighdExtractor(args[1],args[2])
    highd_extractor.extract()

if __name__ == "__main__":
    main()