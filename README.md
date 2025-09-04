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

By completing this course, you will:

- ✅ Differentiate between AI, ML, Deep Learning, and Generative AI
- ✅ Implement supervised and unsupervised learning algorithms
- ✅ Identify and address overfitting and underfitting in ML models
- ✅ Execute the complete ML lifecycle from data preparation to model evaluation
- ✅ Build, train, and evaluate ML models using industry-standard tools
- ✅ Apply DevOps practices to ML model development and deployment
- ✅ Integrate AWS AI services into real-world applications

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
├── .env.example                      # Environment variables template
├── caip_01_01/                       # Week 1, Call 1 materials
│   ├── caip_week1_call1.ipynb       # Main learning notebook
│   ├── *.png                        # Supporting images and diagrams
│   └── caip_week1_call1_intro_to_ai_ml.pdf  # Reference materials
├── notebooks/                        # Additional practice notebooks
├── data/                            # Sample datasets
├── models/                          # Trained model artifacts
├── scripts/                         # Utility scripts
├── tests/                           # Unit tests
└── docs/                           # Additional documentation
```

## 🗺️ Learning Path

### Week 1: AI/ML Fundamentals
- **Call 1**: Introduction to AI, ML, DL, and GenAI
  - 📓 [`caip_01_01/caip_week1_call1.ipynb`](caip_01_01/caip_week1_call1.ipynb)
  - Topics: AI concepts, supervised vs unsupervised learning, overfitting/underfitting
  - Hands-on: Build and evaluate classification models

### Week 2: Advanced ML Concepts
- **Call 2**: Deep Learning and Neural Networks
- **Call 3**: Model Evaluation and Selection

### Week 3: AWS AI Services
- **Call 4**: Introduction to AWS AI/ML Services
- **Call 5**: Hands-on with Amazon SageMaker

### Week 4: Generative AI and LLMs
- **Call 6**: Understanding Large Language Models
- **Call 7**: AWS Bedrock and Foundation Models

### Week 5: Practical Applications
- **Call 8**: Computer Vision with AWS Rekognition
- **Call 9**: Natural Language Processing with AWS Comprehend

### Week 6: Deployment and Operations
- **Call 10**: MLOps and Model Deployment
- **Call 11**: Monitoring and Maintenance
- **Call 12**: Final Project and Review

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

### Getting Help
- 📧 **Course Support**: [support@example.com](mailto:support@example.com)
- 💬 **Discord Community**: [Join our Discord](https://discord.gg/example)
- 📚 **AWS Documentation**: [AWS AI/ML Docs](https://docs.aws.amazon.com/machine-learning/)

## 🤝 Contributing

We welcome contributions to improve the learning experience!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Include tests for new functionality
- Update documentation as needed

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Tags

`aws` `ai` `machine-learning` `certification` `python` `jupyter` `devops` `mlops` `education` `hands-on-learning`

---

**Happy Learning! 🚀**

*For questions or support, please reach out to the course instructors or use the course discussion forums.*