# AWS Certified AI Practitioner - September 2025 Cohort

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![AWS](https://img.shields.io/badge/AWS-AI%20Practitioner-FF9900.svg)](https://aws.amazon.com/certification/certified-ai-practitioner/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A comprehensive learning repository for AWS Certified AI Practitioner certification preparation, focusing on AI/ML fundamentals, hands-on implementation, and AWS AI services.**

## 📋 Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Learning Path](#learning-path)
- [DevOps Best Practices](#devops-best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Resources](#resources)

## 🎯 Overview

This repository contains hands-on learning materials for the AWS Certified AI Practitioner certification. The content is structured as a progressive learning journey covering fundamental AI/ML concepts, practical implementations using Python and scikit-learn, and AWS AI service integration.

### Key Focus Areas
- **Artificial Intelligence Fundamentals**: Understanding AI, ML, DL, and GenAI
- **Machine Learning Lifecycle**: End-to-end ML model development
- **AWS AI Services**: Hands-on experience with AWS AI/ML tools
- **DevOps for ML**: Best practices for ML model deployment and maintenance

## 🎓 Learning Objectives

### Completed Learning Outcomes ✅

After completing the first three sessions, you will be able to:

- **Understand AI/ML Fundamentals**: Differentiate between AI, ML, Deep Learning, and Generative AI
- **Classification Concepts**: Distinguish between supervised vs unsupervised learning, and classification vs regression
- **Model Training**: Execute the complete ML lifecycle from data preparation to model evaluation
- **Evaluation Metrics**: Explain and compute key metrics including accuracy, precision, recall, and F1 score
- **Confusion Matrix**: Interpret confusion matrices and understand their practical implications
- **Real-world Applications**: Relate evaluation metrics to business decision-making scenarios
- **Hands-on Skills**: Build, train, and evaluate ML models using scikit-learn
- **Data Analysis**: Work with real datasets (Iris, Titanic) and perform exploratory data analysis
- **Model Interpretation**: Understand overfitting, underfitting, and feature importance
- **Cloud ML Training**: Deploy and manage ML training jobs using Amazon SageMaker
- **AWS Infrastructure**: Set up SageMaker Studio domains using CloudFormation templates
- **S3 Integration**: Upload datasets and retrieve model artifacts from S3 buckets
- **Managed Training**: Use SKLearn Estimator for scalable, reproducible model training
- **Cloud Monitoring**: Monitor training jobs through CloudWatch logs and metrics
- **Secure ML Workflows**: Understand IAM roles and permissions for ML operations

### Future Learning Goals 🎯

Additional objectives will be added as the course progresses, including:
- Advanced ML techniques and algorithms
- Additional AWS AI services integration
- MLOps and deployment practices
- Deep learning and neural networks
- Model deployment and inference endpoints

## 📚 Prerequisites

### Technical Requirements
- **Python**: 3.11 or higher
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 5GB free space

### Knowledge Prerequisites
- Basic programming experience (preferably Python)
- Understanding of basic statistics and mathematics
- Familiarity with command line/terminal operations
- Basic understanding of cloud computing concepts

## 🛠️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/luit-sept-2025-black-aws-ai.git
cd luit-sept-2025-black-aws-ai
```

### 2. Python Virtual Environment Setup

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n aws-ai-practitioner python=3.11
conda activate aws-ai-practitioner
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install Jupyter extensions (optional)
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

### 5. Verify Installation
Run the first cell in any notebook to verify all dependencies are correctly installed.

## 📁 Project Structure

```
luit-sept-2025-black-aws-ai/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── .github/                          # GitHub templates and workflows
│   └── pull_request_template.md     # PR template
├── caip_01_01/                       # Week 1, Call 1 materials
│   ├── caip_week1_call1.ipynb       # Introduction to AI/ML fundamentals
│   ├── *.png                        # Supporting images and diagrams
│   └── caip_week1_call1_intro_to_ai_ml.pdf  # Reference materials
├── caip_01_02/                       # Week 1, Call 2 materials
│   ├── caip_week1_call2.ipynb       # Model evaluation and metrics
│   ├── caip01_week1_call2.pdf       # Reference materials
│   ├── train.csv                    # Titanic training dataset
│   └── test.csv                     # Titanic test dataset
└── caip_02_01/                       # Week 2, Call 1 materials
    ├── caip_week2_call1.ipynb       # Train and evaluate in the cloud
    ├── caip_week2_call1.pdf         # Reference materials
    ├── cf_templates/                # CloudFormation templates
    │   └── sagemaker_infra.yaml     # SageMaker infrastructure setup
    ├── train_model.py               # Training script for SageMaker
    ├── requirements.in              # Training script dependencies
    ├── cleaned_titanic.csv          # Preprocessed dataset
    ├── train.csv                    # Original training data
    ├── test.csv                     # Original test data
    └── *.png                        # Supporting images and diagrams
```

## 🗺️ Learning Path

### Week 1: AI/ML Fundamentals

#### Call 1: Introduction to AI, ML, DL, and GenAI ✅ *Completed*
- 📓 [`caip_01_01/caip_week1_call1.ipynb`](caip_01_01/caip_week1_call1.ipynb)
- **Topics Covered:**
  - Differences between AI, ML, Deep Learning, and Generative AI
  - Supervised vs Unsupervised Learning (with practical examples)
  - Classification vs Regression problems
  - Overfitting vs Underfitting concepts
  - Feature importance analysis
  - Complete ML lifecycle: Prepare → Train → Predict & Evaluate
- **Hands-on Experience:**
  - Built classification models using scikit-learn
  - Worked with the Iris dataset
  - Created decision trees and evaluated model performance
  - Visualized data patterns and model decision boundaries

#### Call 2: Model Evaluation - "How Good Is Your Model?" ✅ *Completed*
- 📓 [`caip_01_02/caip_week1_call2.ipynb`](caip_01_02/caip_week1_call2.ipynb)
- **Topics Covered:**
  - Introduction to Kaggle and the Titanic dataset
  - Key evaluation metrics: Accuracy, Precision, Recall, F1 Score
  - Understanding and interpreting confusion matrices
  - Why accuracy alone can be misleading
  - Real-world application of metrics (Facebook Marketplace gun detection example)
  - Trade-offs between different types of errors (false positives vs false negatives)
- **Hands-on Experience:**
  - Loaded and preprocessed the Titanic dataset
  - Built and evaluated classification models
  - Computed and visualized evaluation metrics
  - Interpreted classification reports and confusion matrices

### Week 2: Cloud-Based ML with AWS SageMaker

#### Call 1: Train and Evaluate in the Cloud ✅ *Completed*
- 📓 [`caip_02_01/caip_week2_call1.ipynb`](caip_02_01/caip_week2_call1.ipynb)
- **Topics Covered:**
  - Benefits of cloud-based ML training (scalability, reproducibility, separation of concerns)
  - Infrastructure setup using CloudFormation templates
  - SageMaker Studio domain and IAM role configuration
  - S3 integration for data storage and model artifacts
  - SKLearn Estimator for managed training jobs
  - CloudWatch logging and monitoring
  - Secure ML workflows with proper permissions
- **Hands-on Experience:**
  - Deployed SageMaker infrastructure programmatically
  - Uploaded datasets to S3 buckets
  - Created and executed training scripts for SageMaker
  - Launched managed training jobs using SKLearn Estimator
  - Monitored training progress through CloudWatch logs
  - Retrieved and analyzed model artifacts from S3

### Future Sessions
*Content will be added as sessions are completed*

## 🔧 DevOps Best Practices

### Version Control
- **Git Workflow**: Feature branches, pull requests, and code reviews
- **Notebook Management**: Use `nbstripout` to remove outputs before committing
- **Model Versioning**: Track model artifacts using DVC or MLflow

### Environment Management
```bash
# Always use virtual environments
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Pin dependency versions
pip freeze > requirements.txt

# Use environment variables for configuration
cp .env.example .env
# Edit .env with your specific settings
```

### Code Quality
```bash
# Install development dependencies
pip install black flake8 pytest jupyter-nbextensions-configurator

# Format code
black scripts/ tests/

# Lint code
flake8 scripts/ tests/

# Run tests
pytest tests/
```

### Jupyter Notebook Best Practices
- Clear outputs before committing: `jupyter nbconvert --clear-output --inplace *.ipynb`
- Use meaningful cell markdown for documentation
- Keep notebooks focused on single topics
- Extract reusable code into Python modules

### ML Model Lifecycle
1. **Experimentation**: Use notebooks for exploration and prototyping
2. **Development**: Convert notebook code to Python modules
3. **Testing**: Unit tests for data processing and model functions
4. **Deployment**: Containerize models for consistent environments
5. **Monitoring**: Track model performance and data drift

## 🚨 Troubleshooting

### Common Issues

#### Jupyter Kernel Issues
```bash
# Reinstall kernel
python -m ipykernel install --user --name=aws-ai-practitioner
```

#### Package Installation Errors
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Clear pip cache
pip cache purge

# Install with verbose output
pip install -v package-name
```

#### Memory Issues
- Reduce dataset size for initial learning
- Use `del` to free memory in notebooks
- Restart kernel regularly during development

#### AWS Credentials
```bash
# Configure AWS CLI
aws configure

# Or use environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

## 📖 Resources

### Official AWS Resources
- [AWS Certified AI Practitioner Exam Guide](https://aws.amazon.com/certification/certified-ai-practitioner/)
- [AWS Machine Learning Training](https://aws.amazon.com/training/learning-paths/machine-learning/)
- [AWS AI Services Documentation](https://docs.aws.amazon.com/ai-services/)

### Additional Learning Materials
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)

### Community Resources
- [AWS Machine Learning Community](https://aws.amazon.com/developer/community/machine-learning/)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Papers with Code](https://paperswithcode.com/)

---

## 🏷️ Tags

`aws` `ai` `machine-learning` `certification` `python` `jupyter` `devops` `mlops` `education` `hands-on-learning`

---

**Happy Learning! 🚀**

*For questions or support, please reach out to the course instructors or use the course discussion forums.*