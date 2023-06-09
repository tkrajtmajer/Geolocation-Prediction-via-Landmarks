# Invention Assignment 
## Geolocation Prediction via Landmarks

### setup
install the dependencies with
```
pip install -r requirements.txt
```

### run
to run the application first run
```
python construct_database.py
```
to create the database with all the entries located at 'resources/images'
After the script is executed, you can run 
```
python runWeb.py
```
to access the local server at url 'http://127.0.0.1:5000/'. Here you can upload the video of your choice. 

### directory setup
the directory structure is set up as follows:
```
project_root/
├── resources/
│   ├── db/
│   │   ├── ...
│   ├── images/
│   │   ├── aaf
│   │   |   ├── ...
│   │   ├── agi
│   │   |   ├── ...
│   │   ├── ...
│   │   └── zen
|   ├── videos/
|   |   ├── server/
│   │   |   ├── ...
│   │   ├── VIDEO0191.avi
│   │   ├── ...
│   └── ...
├── main.py
├── construct_database.py
├── ...
├── requirements.txt
└── README.md
```

#### List of files that contain new code:
- construct_database.py
- database_reader.py
- evaluate.py
- evaluate_main.py
- geolocation.py
- main.py
- matching.py
- processing.py
- runWeb.py
- SIFT_transform.py
- templates/display_res.html
- templates/index.html
