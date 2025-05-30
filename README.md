# AI Learn

This repository serves as a collection of AI and machine learning projects, models, and experiments. Each project is organized into its own directory and includes all necessary files for dataset preparation, model training, evaluation, and deployment.

## Table of Contents

- [Projects](#projects)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Contributing](#contributing)
---

## Projects

### 1. **First Model: Sentiment Analysis**
- **Description**: A sentiment analysis model fine-tuned on a Hugging Face transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- **Key Features**:
  - Dataset preparation using `creatingdataset.ipynb`.
  - Model training and evaluation using `training.ipynb`.
  - Deployment via Hugging Face pipelines.
- **Directory**: `first-model-sentiment-analysis/`
- **Output**: Fine-tuned model saved in `my_sentiment_model/`.

### 2. **Second Model: Suicide Detection**
- **Description**: A suicide detection model fine-tuned on the `distilbert-base-uncased-finetuned-sst-2-english` transformer model.
- **Key Features**:
  - Dataset preparation and filtering using `json2jsonl.py` and `trainingModel.ipynb`.
  - Fine-tuning the model for binary classification (`non-suicide` and `suicide`).
  - Deployment via Hugging Face pipelines with custom label mapping.
- **Directory**: `suicide-detection/`
- **Output**: Fine-tuned model saved in `my_suicide_buddy/`.
- **Model Link**: "amor3x/my_suicide_buddy"

### 3. **Third Model: Mental Health Sentiment Analysis**
- **Description**: Mental Sentiment model finetuned on the `distilbert/distilbert-base-uncased` transformer model.
- **Key Features**:
  - Dataset preparation and filtering using `json2jsonl.py` and `trainingModel.ipynb`.
  - Fine-tuning the model for classification (`anxiety` , `normal` , `bipolar` , `depression` and `suicidal`).
  - Deployment via Hugging Face pipelines with custom label mapping.
- **Directory**: `mental-health/`
- **Output**: Fine-tuned model saved in `mental_health_bud/`.
- **Model Link**: "amor3x/mental_health_bud"

---

## Getting Started

To get started with any project in this repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-learn.git
   cd ai-learn
2. Navigate to project directory of your choice
3. Run the notebook Cells in appropriate order :
    - Creating Dataset
    - Training Model

## Requirements
Each project may have its own dependencies. Below are the general requirements:

- Python 3.x
- Jupyter Notebook
- Hugging Face Transformers
- Kagglehub 
- Datasets library
- Additional libraries as specified in individual project directories.

## Tools
- https://codebeautify.org/csv-to-json-converter FOR csv to JSON
- Hugging face for models and datasets
- Kaggle for Datasets 

## Contributing
Contributions are welcome! If you'd like to add a new project or improve an existing one:

## Command to Upload Model to HuggingFace
$ huggingface-cli repo create my_suicide_buddy
$ huggingface-cli upload amor3x/mental_health_bud
