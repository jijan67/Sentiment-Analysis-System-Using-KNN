# 🔍 Twitter Sentiment Analysis Using KNN

This documentation provides a detailed explanation of the Twitter Sentiment Analysis project implemented using K-Nearest Neighbors (KNN) classifier in Python. The goal of this system is to analyze and classify the sentiment of tweets based on their textual content.

---

## Overview

The project uses a dataset containing Twitter data with sentiment labels. The model employs K-Nearest Neighbors (KNN) algorithm for classification and achieves an accuracy of 94.19%.

## Libraries Used

The following Python libraries were used to implement the model:

- **NumPy:** For numerical operations
- **Pandas:** For data manipulation and analysis
- **Matplotlib & Seaborn:** For data visualization
- **re:** For regular expression operations in text cleaning
- **Scikit-learn:** For data preprocessing, feature extraction, model building, and evaluation metrics

## Project Structure

**1. Data Loading and Preprocessing:**
- The dataset is loaded from CSV files for both training and testing
- Unnecessary columns are dropped and remaining columns are renamed
- Text data is cleaned using regex to remove URLs, mentions, special characters, and numbers
- Empty texts and duplicates are removed

**2. Text Feature Extraction:**
- **TF-IDF Vectorizer** is used to convert text data into numerical features
- A maximum of 5000 features is selected for the vectorization

**3. Model Building:**
- A **KNeighborsClassifier** with n_neighbors=5 is created and trained on the vectorized text data
- Labels are encoded using **LabelEncoder**

**4. Model Evaluation:**
- The model is evaluated on the test dataset
- Accuracy score, classification report, and confusion matrix are generated to assess model performance

**5. Sentiment Prediction Function:**
- A function `predict_text_knn` is implemented to take new text input and predict its sentiment
- The function cleans the input text, transforms it using the trained TF-IDF vectorizer, and applies the KNN model for prediction

## Evaluation Results

- **Test Accuracy:** 94.19%
- The confusion matrix shows that the model performs well with minimal misclassifications
- The classification report provides detailed performance metrics for each sentiment class

## Conclusion

The Twitter sentiment analysis system built with KNN demonstrates high accuracy in classifying tweet sentiments. This model can help analyze public opinion, customer feedback, and social media trends by automatically determining the sentiment expressed in textual content.

## Future Improvements

- Experiment with different n_neighbors values to optimize the KNN model
- Try more advanced text preprocessing techniques like lemmatization
- Compare performance with other classification algorithms
- Implement cross-validation for more robust evaluation
- Explore dimensionality reduction techniques to improve efficiency
