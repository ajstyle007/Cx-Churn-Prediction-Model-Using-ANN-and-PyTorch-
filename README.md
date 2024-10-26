# 💳👨🏻‍💼💁🏻‍♀️ Customer Churn Prediction Using ANN and PyTorch
Live Web App Link: [Cx Churn Predictor](https://musk12-cx-churn-prediction-with-pytorch.hf.space)

This project predicts customer churn in credit card services using an Artificial Neural Network (ANN) built with PyTorch. The model was trained on the Credit Card Customer Churn Prediction dataset from Kaggle, achieving an accuracy of 86%.

### Project Overview
Customer churn prediction helps companies identify customers likely to stop using their services, allowing targeted retention strategies. In this project, an ANN model was designed and trained to predict churn based on various customer metrics.

![app_image](https://github.com/user-attachments/assets/b088e200-9cf8-4657-b270-859f70a4584b)

### Table of Contents
- 📁 Dataset
- 🧩 Workflow
- 🏗️ Model Architecture
- 📈 Results
- 🚀 Deployment
- 📝 Conclusion
- 🛠️ Tools Used

### 📁 Dataset
The dataset was sourced from Kaggle: Credit Card Customer Churn Prediction. It contains customer demographics, account information, and usage metrics, which were processed and fed into the neural network model.

### 🧩 Workflow
1. Data Collection
Downloaded the dataset from Kaggle and loaded it into a pandas DataFrame.

2. Data Cleaning
Used pandas and numpy to clean and preprocess the data, handling any missing values or outliers.

3. Exploratory Data Analysis (EDA)
Explored data patterns and distributions using Matplotlib, Seaborn, and Plotly to gain insights and identify key features affecting churn.

4. Feature Engineering
Converted categorical columns to numerical values using pd.get_dummies() to prepare for model training.

5. Data Splitting and Scaling
Split the data into training and testing sets using train_test_split and applied standard scaling using sklearn's StandardScaler.

6. PyTorch Setup

Imported necessary modules from PyTorch, including torch and torch.nn.
Wrote device-agnostic code to allow training on GPU (if available) or CPU.
Data Conversion to PyTorch Tensors
Converted the processed data into PyTorch tensors for efficient handling during training.

7. Model Definition 
Defined an ANN architecture using a custom class CxChurn(nn.Module) with the following structure:
The model architecture consists of four fully connected layers with ReLU activation and dropout for regularization:

8. Training and Testing
Defined training and testing loops, using BCEWithLogitsLoss as the loss function and Adam optimizer with a learning rate of 0.001. Achieved 86% accuracy after 2900 epochs.

9. Model Saving
Saved the trained PyTorch model for future use, achieving a final performance with:
```
Epoch: 2900 | Train_Loss:  0.34532 | Train_Acc:  86% | Test_loss:  0.33879 | Test_acc:  86%
```

### 🏗️ Model Architecture
The model architecture consists of four fully connected layers with ReLU activation and dropout for regularization:
```
class CxChurn(nn.Module):

  def __init__(self):
    super().__init__()

    self.layer1 = nn.Linear(in_features=11, out_features=16)
    self.layer2 = nn.Linear(in_features=16, out_features=16)
    self.layer3 = nn.Linear(in_features=16, out_features=8)
    self.layer4 = nn.Linear(in_features=8, out_features=1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = self.layer1(self.relu(x))
    x = self.dropout(x)
    x = self.layer2(self.relu(x))
    x = self.dropout(x)
    x = self.layer3(self.relu(x))
    x = self.dropout(x)
    x = self.layer4(x)

    return x

model = CxChurn().to(device)
model
```

### 📈 Results
The model was trained to an accuracy of 86% with training and testing losses of 0.34532 and 0.33879, respectively.

### 🚀 Deployment
Live Web App Link: [Cx Churn Predictor](https://musk12-cx-churn-prediction-with-pytorch.hf.space)
The Streamlit application provides an interactive platform for making predictions on new customer data. This app was deployed on Hugging Face Spaces for easy access to stakeholders and users.

### 📝 Conclusion
The ANN model accurately predicted customer churn with a high degree of precision. This model can assist businesses in proactively managing customer retention, leading to a better understanding of customer behavior and more targeted marketing strategies.

### 🛠️ Tools Used
🐍 Python: Core programming language
📊 Pandas: Data manipulation and analysis
🧮 NumPy: Numerical computing
📉 Matplotlib, Seaborn, Plotly: Data visualization
🔄 PyTorch: Building the neural network
📈 sklearn: Data preprocessing (train-test split, scaling)
🌐 Streamlit: Model deployment interface
🤗 Hugging Face Spaces: Model hosting and deployment platform
