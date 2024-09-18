# Spam Email Detection Using NLP and Multinomial Naive Bayes

This project implements a **Spam Email Detection** system using **Natural Language Processing (NLP)** techniques and a **Multinomial Naive Bayes** classifier. The goal of the project is to classify emails as either **spam** or **ham** (not spam) by analyzing the email's text content.

## Features
- **Text Preprocessing**: Emails are tokenized, stop words and punctuation are removed using **spaCy**.
- **TF-IDF Vectorization**: Text features are converted into numerical format using **TF-IDF**.
- **Multinomial Naive Bayes**: A Naive Bayes classifier is used to detect spam emails.
- **Performance Metrics**: The system provides evaluation metrics like accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: Visual representation of model performance through a confusion matrix.
- **ROC Curve & AUC**: Displays the model's ability to distinguish between classes using ROC and AUC.

## Technologies Used
- **Python 3.7+**
- **spaCy**: For NLP tokenization and stopword removal.
- **Scikit-learn**: For machine learning and model evaluation.
- **Matplotlib**: For visualizing results (confusion matrix, ROC curve).
- **pandas**: For data manipulation.

## Getting Started

### Prerequisites
Before you begin, ensure you have the following installed on your machine:
- Python 3.7+ 
- `pip` (Python package manager)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/spam-email-detection.git
   cd spam-email-detection
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the **spaCy** English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Dataset
The dataset used for this project is `spam_ham_dataset.csv`. This file should be located in the root directory of the project. If not, you can download or place your dataset there with the following columns:
- **text**: The email content
- **label_num**: The label for classification (1 = spam, 0 = ham)

### Running the Project

To run the project, use the following command:
```bash
python spam_email_classifier.py
```

This will process the dataset, train the **Multinomial Naive Bayes** model, and output performance metrics for both the validation and test sets.

### Outputs
The following results will be generated:
- **Accuracy**, **Precision**, **Recall**, and **F1-score** for validation and test sets.
- A **confusion matrix** plot for the test set.
- An **ROC curve** plot showing the model's performance in distinguishing between spam and ham.

### Directory Structure
```
.
├── spam_email_classifier.py     # Main script for the spam email detection system
├── spam_ham_dataset.csv         # Dataset file
├── README.md                    # Project readme file
├── requirements.txt             # Required Python packages
└── ...
```

## Results
In this project, the model achieved the following performance:
- **Validation Accuracy**: X%
- **Test Accuracy**: X%
- **AUC Score**: X%

The confusion matrix and ROC curve provide further insights into model performance.

## Contributing
If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Make sure to update tests as appropriate.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### requirements.txt
For your **requirements.txt** file, you can include the following packages:
```
pandas
scikit-learn
spacy
matplotlib
```