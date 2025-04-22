# League of Legends Match Outcome Predictor (PyTorch)

This project demonstrates how to build a simple logistic regression model using PyTorch to predict match outcomes (win/loss) in League of Legends based on in-game statistics.

> 🔗 Full blog post: [Read the tutorial on DataSkillBlog]([https://dataskillblog.com/your-post-link-here](https://dataskillblog.com/league-of-legends-match-prediction-pytorch))  
> 🌐 Visit: [https://dataskillblog.com](https://dataskillblog.com)

---

## 🚀 Features

- Beginner-friendly code structure
- Data loading and preprocessing
- Logistic regression using PyTorch
- Model training and optimization
- Evaluation using confusion matrix, classification report, and ROC curve
- Feature importance visualization

---

## 📁 Files Included

- `league_match_predictor.py`: Full code with comments and visualizations
- `league_of_legends_data.csv`: Dataset used to train and test the model

---

## 📊 Dataset

The dataset contains match statistics from League of Legends games with the target column `win`:
- `1` → Win
- `0` → Loss

Each row represents a single game instance with features like kills, deaths, gold earned, damage dealt, etc.

---

## 📈 Example Visualizations

- Confusion Matrix
- ROC Curve
- Feature Importance Bar Chart

These are automatically generated after training the model.

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- matplotlib
- seaborn

Install required libraries:
```bash
pip install torch pandas scikit-learn matplotlib seaborn
