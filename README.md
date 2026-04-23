# GWU-DS-Capstone
Curate - Personal Style Application

## Problem
Online Shopping takes a lot of time and effort sifting through different websites.
Finding exactly what you are looking for is based on what websites your search engine shows you.
For more specific searches, it is harder to find exactly what you are looking for.
Online shopping doesn't offer the same personalized touch as in person.

## Solution
An AI based shopping assistant that learns from the user.
Help the user determine intention and make it easier to find exactly what they are looking for.
Different then standard search engines as they only search with input text, but this gets personalized with a person over time.
Standard search engine also only show popular brands or sponsored brands while this allows for smaller reputable brands to be shown if it matches.
Helps users by find clothes exactly to their wishes and retailers by lowering return rate of clothes.
Offers a ‘For you’ section that recommends clothes based on specific taste of the user.

## Tech Stack
User Profile-
Allows the user to select their personal preferences to better allow the algorithm to understand.
Like/Dislike Feedback loop allows the user to like or dislike the recommendations, so the model can learn their taste. 
Photo Upload using PILLOW package - Allows the user to upload a picture, and the algorithm gains insights.

Search Function - 
NLP Model - Implements a Semantic BERT model to understand the meaning of the user.
Manual scoring mechanism that shows results based on the user's preferences. 
Cosine Similarity to show the results of the search. 

For You Section-
XGBoost Machine Learning model- Uses a Machine Learning model based on all of the user feedback and user preferences to show clothes the user might like.
Falls back to the manual scoring methodology if the XGBoost model is not fully trained. 

Backend -
SQLite - Used to store user data to make sure each person’s profile data is stored and they can come back to the application to pick up where they left off. 
FastAPI- Handles all the models and connects the frontend and the backend.

Frontend-
React - Allows for a visually appealing and reactive interface for the backend.

## Steps to Launch 
Step 1 - Download all of the source files and make sure to check all of the packages are the correct version using the requirements.txt file.

Step 2 - Navigate to the file in a terminal and then do the following commands to run the backend of the project first 

cd backend

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload

Step 3 - Open a seperate terminal and navigate to the main project folder in order to then run the frontend using the following commands.

cd frontend

npm install

npm run dev

This should provide you with a link that looks like localhost. Click it to start 

Step 4 - In the application input a unique username to get personalized results. 

Step 5 - To ensure the XGBoost is working properly, do the following:

Use the app and like/dislike items

Go to http://127.0.0.1:8000/docs

Run POST /train-xgb


