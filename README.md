# 🎓 A Machine Learning Approach for Tracking and Predicting Student Performance in Degree Programs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

Every year, only **50% of university students** successfully complete their graduation courses. Early identification of at-risk students can enable timely intervention and improve outcomes significantly.

This project implements an end-to-end **Machine Learning pipeline** to:
- **Predict** whether a student's future course GPA will be **HIGH or LOW** based on their past performance
- **Recommend** the most suitable **future IT career path** from 34 options based on their academic profile
- **Compare** multiple ML algorithms (SVM, Logistic Regression, Random Forest) against a proposed **Ensemble-based Progressive Prediction (EPP)** algorithm

The system is deployed as an interactive **Tkinter desktop application** with a clean GUI, allowing academic advisors to upload student datasets and get real-time predictions.

---

## 🚀 Key Features

- 📂 **Dataset Upload** — Load the UCLA student performance dataset directly via the GUI
- 🔢 **Matrix Factorization** — Converts raw course marks into structured feature vectors; students who haven't taken a course get a `0` in the matrix
- 🤖 **4 ML Algorithms** — Train and evaluate SVM, Logistic Regression, Random Forest, and the proposed EPP algorithm
- 📊 **MSE Comparison Graph** — Visual bar chart comparing Mean Square Error across all algorithms
- 🎯 **Real-time Prediction** — Upload new student records and instantly get GPA prediction (HIGH/LOW) + recommended future course
- 📈 **Extension Work** — Enhanced version predicts not just GPA grade but also suggests one of **34 specific IT career paths**

---

## 🧠 How It Works

### Problem Statement

Three core challenges are addressed:

| # | Challenge | Solution |
|---|-----------|----------|
| 1 | Students take different courses — how to build a unified ML model? | **Matrix Factorization** — clusters related courses; marks `1` if taken, `0` if not |
| 2 | Courses are not equally informative for predictions | Feature vector construction filters relevant course clusters |
| 3 | Existing algorithms only use past data, ignoring ongoing performance | **EPP algorithm** uses both past and ongoing course data for progressive prediction |

### Algorithm Pipeline

```
Raw Student Data (CSV)
        │
        ▼
Label Encoding (10 categorical features)
        │
        ▼
Matrix Factorization → Feature Vectors (21 features)
        │
        ▼
Normalizer Preprocessing
        │
        ▼
Train / Test Split (80% / 20%)
        │
        ├──► SVM Classifier
        ├──► Logistic Regression
        ├──► Random Forest (n=200)
        │
        ▼
Base Classifier Results → BaggingClassifier (EPP)
        │
        ▼
Predict GPA: HIGH or LOW + Recommend Future Course
```

### Proposed EPP Algorithm

```python
base = RandomForestClassifier()
epp  = BaggingClassifier(base_estimator=base)
epp.fit(X_train, y_train)
prediction = epp.predict(X_test)
```

EPP (Ensemble-based Progressive Prediction) wraps a **Random Forest base classifier** inside a **BaggingClassifier**, combining multiple estimators to reduce variance and improve accuracy — achieving the **lowest MSE (37%)** among all tested algorithms.

---

## 📊 Results

| Algorithm | Accuracy | MSE |
|-----------|----------|-----|
| SVM | ~44% | 56% |
| Random Forest | ~57% | 43% |
| Logistic Regression | ~57% | 43% |
| **EPP (Proposed)** | **~63%** | **37%** ✅ |

> **Extension dataset results:** EPP achieved **95% accuracy** on the extended dataset with career path prediction.

---

## 🗂️ Project Structure

```
📦 student-performance-predictor
├── 📄 StudentPerformance.py       # Main application (Tkinter GUI + ML pipeline)
├── 📄 run.bat                     # Windows launcher — double-click to run
├── 📁 dataset/
│   ├── dataset.txt                # UCLA student dataset (77 records, 13 features)
│   └── extension_dataset.csv      # Extended dataset with career labels
├── 📁 test_data/
│   ├── new_test_records.csv       # Sample test records for GPA prediction
│   └── extension_new_test_record.txt  # Test records for career path prediction
├── 📄 README.md
```

---

## 🗃️ Dataset

**Source:** UCLA University student performance dataset

**Base Dataset Features (13 columns):**

| Feature | Description |
|---------|-------------|
| Math33A, Math33B, Math31B ... | Subject-wise marks (0 if not taken) |
| GPA | Target label — `0` = Low GPA, `1` = High GPA |

**Extended Dataset Features (21 columns):**

| Feature | Description |
|---------|-------------|
| `self-learning_capability` | Student's self-learning ability (categorical) |
| `Extra-courses_did` | Whether student took extra courses (categorical) |
| `certifications` | Certifications obtained |
| `workshops` | Workshop participation |
| `talenttests_taken` | Talent tests attempted |
| `reading_and_writing_skills` | Language skills score |
| `memory_capability_score` | Memory assessment score |
| `Interested_subjects` | Subject preferences |
| `interested_career_area` | Career area of interest |
| `Job_Higher_Studies` | Preference: Job or Higher Studies |
| + 11 more academic features | ... |
| **Target** | Career path label (34 classes) |

**Dataset Size:** 77 student records | **Train:** 61 records | **Test:** 16 records

---

## 🎯 Predicted Career Paths (34 Classes)

The extension model recommends one of these IT career paths:

<details>
<summary>Click to expand all 34 career paths</summary>

- Database Developer
- Portal Administrator
- Systems Security Administrator
- Business Systems Analyst
- Software Systems Engineer
- Business Intelligence Analyst
- CRM Technical Developer
- Mobile Applications Developer
- UX Designer
- Quality Assurance Associate
- Web Developer
- Information Security Analyst
- CRM Business Analyst
- Technical Support
- Project Manager
- Information Technology Manager
- Programmer Analyst
- Design & UX
- Solutions Architect
- Systems Analyst
- Network Security Administrator
- Data Architect
- Software Developer
- E-Commerce Analyst
- Technical Services / Help Desk / Tech Support
- Information Technology Auditor
- Database Manager
- Applications Developer
- Database Administrator
- Network Engineer
- Software Engineer
- Technical Engineer
- Network Security Engineer
- Software Quality Assurance (QA) / Testing

</details>

---

## ⚙️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
```

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

> **Note:** `tkinter` is included with standard Python installations on Windows. On Linux, install with:
> ```bash
> sudo apt-get install python3-tk
> ```

### Run the Application

**Option 1 — Windows (recommended):**
```
Double-click run.bat
```

**Option 2 — Command Line:**
```bash
python StudentPerformance.py
```

---

## 🖥️ Usage Guide

Follow these steps in the GUI after launching the application:

| Step | Button | Action |
|------|--------|--------|
| 1 | `Upload UCLA Students Dataset` | Load `dataset/dataset.txt` |
| 2 | `Matrix Factorization` | Build feature vectors from dataset |
| 3 | `Run SVM Algorithm` | Train SVM and view Accuracy + MSE |
| 4 | `Run Random Forest Algorithm` | Train Random Forest and view results |
| 5 | `Run Logistic Regression Algorithm` | Train Logistic Regression and view results |
| 6 | `Propose EPP Algorithm` | Train the proposed EPP model |
| 7 | `Predict Performance` | Upload test records → get GPA prediction + career recommendation |
| 8 | `Mean Square Error Graph` | View bar chart comparing all algorithm MSEs |

> **For Extension (Career Path Prediction):** In Step 1, upload `extension_dataset.csv`. In Step 7, upload `extension_new_test_record.txt`.

---

## 📷 Sample Output

### GPA Prediction Output
```
[0.27, 1.45, 1.03, 1.69, 1.04, 0.08, 399.67, 441.83, 704.345, 0, 0, 0]
====> Predicted New Course GPA Score will be: HIGH

[1.31, 0.38, -1.78, 1.13, -0.64, -1.23, 0, 477.21, 0, 630.17, 3.78, 0]
====> Predicted New Course GPA Score will be: LOW
```

### Career Path Prediction Output (Extension)
```
[student marks array]
====> Predicted GPA: HIGH | Suggested Future Course: Data Architect

[student marks array]
====> Predicted GPA: LOW  | Suggested Future Course: Technical Support
```

### MSE Comparison Graph
```
Y-axis: Mean Square Error (%)
X-axis: SVM | Random Forest | Logistic Regression | EPP (Proposed)

EPP achieves the lowest MSE → highest prediction accuracy
```

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| GUI Framework | Tkinter |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Algorithms | SVM, Logistic Regression, Random Forest, BaggingClassifier (EPP) |
| Preprocessing | LabelEncoder, Normalizer, Matrix Factorization |

---

## 🔮 Future Enhancements

- [ ] Migrate GUI to a **Streamlit web application** for browser-based access
- [ ] Integrate **deep learning models** (LSTM) for temporal progression tracking
- [ ] Add **real-time database connectivity** for live student data ingestion
- [ ] Deploy as a **REST API** for integration with university portals
- [ ] Expand dataset to include **more universities and diverse course structures**
- [ ] Add **explainability layer** (SHAP values) to show feature importance per prediction

---

## 👩‍💻 Author

**Shaik Salma**
Data Analytics & Data Science Professional

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shaik-salma-b982a6349/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/salmawithai)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:salmashaik55301@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If you found this project helpful, please give it a star!
