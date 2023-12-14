# Disaster Response Pipeline Project
The repositorcy can be reached at https://github.com/jockyeving/udacity-disaster_response
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

- The main page displays information about the input messages, such as categorization by genre, or wether they are weather or aid-request related. The text box can be used to test the trained model, so that it displays each output categorization for the given message.