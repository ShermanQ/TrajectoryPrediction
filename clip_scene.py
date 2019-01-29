import csv
import os 

def clip_scene(x_low,x_up,y_low,y_up,scene_path,new_path = "./data/temp/temp.csv"):
    
    with open(scene_path) as scene_csv:
        csv_reader = csv.reader(scene_csv)

        with open(new_path,"a") as new_csv:
            csv_writer = csv.writer(new_csv)

            for line in csv_reader:
                x = float(line[4])
                y = float(line[5])

                if x > x_low and x < x_up and y > y_low and y < y_up:
                    csv_writer.writerow(line)

    if os.path.exists(scene_path):
        os.remove(scene_path)

    os.rename(new_path,scene_path)

    if os.path.exists(new_path):
        os.remove(new_path)

    
def main():
    scene_path = "./data/csv/bad.csv"
    
    x_low = -15
    x_up = 20
    y_up = 10
    y_low = -15
    clip_scene(x_low,x_up,y_low,y_up,scene_path)
    

if __name__ == "__main__":
    main()