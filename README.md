# Phishing Website Detector
https://phishingdetector.streamlit.app/
![](https://ideas.ted.com/wp-content/uploads/sites/3/2020/01/final_featured_art_phishing_istock.jpg)

## Overview
This Python script detects phishing websites using machine learning techniques. It employs logistic regression and multinomial naive Bayes algorithms to classify URLs as either safe or potentially phishing.


## Features
- Utilizes machine learning algorithms for classification.
- Performs text preprocessing on URLs, including tokenization, stemming, and vectorization.
- Compares the performance of logistic regression and multinomial naive Bayes models.
- Provides accuracy scores, classification reports, and confusion matrices for model evaluation.
- Includes sample URLs for testing and example usage.

## Requirements
- Python 3.x
- Required Python libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - nltk
  - pillow
  - wordcloud
  - bs4 (BeautifulSoup)
  - selenium
  - networkx

## Installation
1. Clone or download the repository containing the Python script.
2. Install the required Python libraries using pip:

## Usage
1. Ensure that the script (`phishing_detector.py`) and any required datasets are in the same directory.
2. Run the script using Python:
3. Follow the instructions provided by the script to enter URLs for phishing detection.
4. After entering a URL, click the "Check for Phishing" button to get the prediction.
5. The script will display whether the URL is likely safe or potentially phishing, along with the probability of the prediction.

![Phishing Website Detector](https://github.com/abh2050/phishing-detector/assets/44420081/271c9471-27da-4655-9917-f8aa0594f50e)

## Model Comparison and Results
### Models Compared:
1. **Logistic Regression (LR):**
    - **Algorithm Type:** Supervised Learning (Classification)
    - **Purpose:** Used for binary classification tasks.
    - **Performance:** Achieved an accuracy score of approximately 96% on the test data.
    - **Evaluation Metrics:** Utilized classification report and confusion matrix to evaluate precision, recall, and F1-score for both classes (bad and good).
    - **Pros:** Simple, efficient for small datasets with linear decision boundaries.
    - **Cons:** Sensitive to irrelevant features, may underperform when features are correlated.
  
2. **Multinomial Naive Bayes (NB):**
    - **Algorithm Type:** Supervised Learning (Classification)
    - **Purpose:** Widely used for text classification tasks, including spam detection and sentiment analysis.
    - **Performance:** Achieved an accuracy score of approximately 96% on the test data.
    - **Evaluation Metrics:** Utilized classification report and confusion matrix to evaluate precision, recall, and F1-score for both classes (bad and good).
    - **Pros:** Efficient and fast for large datasets, performs well with high-dimensional data.
    - **Cons:** Assumes independence between features, which may not hold true in real-world scenarios.

### Performance Comparison:
- Both Logistic Regression and Multinomial Naive Bayes models performed comparably well on the test data, achieving an accuracy score of around 96%.
- Evaluation metrics such as precision, recall, and F1-score were also similar for both models, indicating balanced performance in classifying phishing websites.

### Conclusion:
- In this particular scenario, both Logistic Regression and Multinomial Naive Bayes models demonstrated robust performance in detecting phishing websites.
- The choice between these models may depend on factors such as computational efficiency, interpretability, and the nature of the dataset.


![Model Comparison and Results](https://github.com/abh2050/phishing-detector/assets/44420081/796d8adb-b8ba-43c6-9105-ea97579a0db9)

## File Descriptions
- `phishing_detector.py`: Main Python script for detecting phishing websites.
- `phishing_model.pkl`: Pickled machine learning model for phishing detection.
- `cv.pkl`: Pickled CountVectorizer for text preprocessing.

## Credits
This script utilizes various Python libraries and machine learning algorithms. Credits to the developers and contributors of these libraries.
