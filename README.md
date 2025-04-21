# Wine Quality Prediction App

This Streamlit app predicts wine quality using a K-Nearest Neighbors (KNN) algorithm. The app allows users to input various wine characteristics and receive a quality prediction.

## Features
- Predicts wine quality as Low, Medium, or High
- Uses KNN algorithm with distance-weighted neighbors
- Interactive input interface for wine characteristics
- Displays prediction results with confidence scores
- Beautiful, modern UI with styled components

## Local Setup
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud
1. Create a GitHub repository and push your code
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app" and select your repository
5. Choose the main branch and app.py as the entry point
6. Click "Deploy"

## Dataset
The model is trained on the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UCI Machine Learning Repository.

## Model Details
- Algorithm: K-Nearest Neighbors (KNN)
- Configuration: k=5 neighbors with distance weighting
- Features: 11 physicochemical properties of wine
- Target: Wine quality (Low, Medium, High)
- Evaluation: Train-test split with 80-20 ratio 