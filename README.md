# Fake-news-prediction

This repository contains a Python project for predicting fake news using machine learning techniques. The project utilizes various libraries and tools for text processing and classification.

### Libraries Used
- **NumPy (np)**: NumPy is used for numerical computing and array operations.
- **Pandas (pd)**: Pandas is used for data manipulation and analysis.
- **re**: The re module is used for searching text within a document.
- **NLTK (Natural Language Toolkit)**: NLTK is used for natural language processing tasks such as stopwords removal and stemming.
- **scikit-learn**: Scikit-learn is used for machine learning tasks, including feature extraction and classification.
  - **TfidfVectorizer**: Used to convert text into feature vectors using TF-IDF.
  - **train_test_split**: Used for splitting the dataset into training and testing sets.
  - **LogisticRegression**: Used as the classification algorithm.
  - **accuracy_score**: Used to evaluate the accuracy of the model.

### Project Overview
This project aims to predict whether a given piece of news is fake or genuine. It utilizes a dataset containing labeled news articles, where each article is labeled as either fake or genuine.

### How to Use
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your_username/fake-news-prediction.git

**Install the required dependencies. You can use the following command to install them:**
pip install numpy pandas nltk scikit-learn

**Additionally, you might need to download NLTK data using:**
import nltk
nltk.download('stopwords')

### How to Use
Run the Python script `fake_news_prediction.py` to train the model and make predictions.

### Files
- **fake_news_prediction.py**: Python script containing the implementation of the fake news prediction model.
- **data/**: Directory containing the dataset used for training and testing.

### Dataset
The dataset contains labeled news articles, with each article labeled as either fake or genuine. It is located in the `data/` directory.

### Model Training
The model is trained using logistic regression on the TF-IDF features extracted from the text data. After training, the model's accuracy is evaluated using the test set.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Author
Khalfaoui Hassani Marwane
Feel free to contribute or provide feedback to improve this project!

