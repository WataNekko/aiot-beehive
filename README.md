# aiot-beehive
AIoT Beehive Monitoring System

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
    step 2: run the augmentation.py file by your python compiler and wait
