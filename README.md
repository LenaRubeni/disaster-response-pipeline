## Udacity Disaster Response Pipeline

This repository aims at bulding a machine learning pipeline with natural language processing for data from the company *Figure Eight*. 
The dataset contains labeled tweet messages from real disaster events. The goal of this project is to categorize new tweets for disaster response. 

The repository is split into three sections

- the preprocessing of the data in the *data* section
- the building of the machine learning model in the *models* section
- building of web app in the *app* section

## Installation and Dependencies

# Dependencies

- Python
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

# Installation

To run the code clone the repository. 
To start the ETL process please run:


```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
```

To start the machine learning building run:

```
python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
```

To run the web app execute:

```
python run.py
```
