# aiot-beehive
AIoT Beehive Monitoring System

# TUTORIAL FOR improt_beedata.ipynb:
    Step 1: download and unzip: https://1drv.ms/u/s!Amb90n5v6JFuoY82J1QPxtzMkqs3KQ?e=9dxgCg
    Step 2: but the unzipped location to `dataset_path` variable in .env
    Step 3: run pip3 install -r requirements.txt
    Step 3: at the moment all programed function in import_beedata.ipynb shoud work, countinue to develope it!!!

# Augmentation.py tutorial:
    step 1: follow the note in .env.tutorial
        + `dataset_path` is the original images's folder path
            - recomend using .jpeg
            - any size of image allowed
        + `labels_file` is the path of .json which store label of data
            - syntax of .json: 
                ```json
                {
                    "<imgname>":{
                        "cooling": <Bool>,
                        "pollen": <Bool>,
                        "varroa": <Bool>,
                        "wasps": <Bool>
                    },
                    [...]
                }
                ```
        + `export_path` is the exportation location for new image
    step 2 run pip3 install -r requirements.txt
    step 3: run the augmentation.py file by your python compiler and wait
    
# tutorial to get Number of Bee in image:
    Call function detect(save_img, weights, source, save_dir) in ./yolo/yolov7/Object_Detection.py
        + save_img (boolean): decide to output image with boundingbox or not
        + weights (Path - String): Path of the trained model
        + source (Path - String): Path of the input image
        + save_dir (Path - String): Path for output image (only require if save_img = True )