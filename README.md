# Banking Customer Segmentation

This is a **machine learning-powered web application** that segments banking customers based on their behavioral and demographic characteristics. The app utilizes a trained model to analyze customer data and categorize them into distinct segments for better targeting and personalized services.

## 🚀 Features

- **User-friendly Interface:** Built using Streamlit for seamless user interaction.
- **Machine Learning Model:** Uses clustering algorithms for accurate customer segmentation.
- **Real-Time Predictions:** Instantly classifies customers into segments based on user input.
- **Preprocessing Pipeline:** Encodes categorical variables, scales numerical features and dimensionality reduction before making predictions.

## 📌 How It Works

1. Users input customer details such as **account balance,education level, jobtype, housing loan etc.**
2. The data is preprocessed and passed through a trained clustering model.
3. The model assigns the customer to a specific segment (e.g., "Stable Savers with Loans", "High-Balance Professionals").
4. The application displays insights on the segment characteristics.

## 🧐 Installation

To run the app locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Vipina7/BankingCustomerSegmentation.git

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 🏰 Input Parameters

| Parameter           | Description                                                 |
|----------------------|-------------------------------------------------------------|
| job                | Type of job (e.g., admin, technician, services, etc.)        |
| marital            | Marital status (single, married, divorced)                   |
| education          | Education level (primary, secondary, tertiary, unknown)      |
| balance           | Account balance in euros                                     |
| default           | Credit default status (`yes` or `no`)                        |
| housing           | Has a housing loan (`yes` or `no`)                           |
| loan              | Has a personal loan (`yes` or `no`)                          |
| contact           | Contact communication type (e.g., cellular, telephone, unknown) |
| month             | Last contact month                                           |
| duration         | Call duration in seconds                                     |
| campaign         | Number of contacts during this campaign                      |
| previous         | Number of times contacted before                             |
| poutcome         | Outcome of the previous marketing campaign (success, failure, unknown) |

## 📊 Dataset Description

The dataset used for training contains the following attributes:

- **Numerical Features:** Age, Annual Income, Account Balance, Monthly Transactions, Credit Score
- **Categorical Features:** Employment Type, Marital Status
- **Target Variable:** Customer Segments (e.g., "Stable Savers with Loans", "High-Balance Professionals")

## 🔍 Model and Preprocessing

- **Preprocessor:**
  - Handled missing values and outliers.
  - Encoded categorical variables using **One-Hot Encoding**.
  - Standardized numerical features using **StandardScaler**.
- **Model Training:**
  - Evaluated multiple clustering algorithms, including **K-Means, DBSCAN, Agglomerative Clustering**.
  - Selected the best-performing model based on **Silhouette Score**.
  - **Final Model:** K-Means with optimized cluster count using **Silhouette Analysis**.

## 🌟 Example Output

- **Segment Prediction:** "This customer belongs to the **Stable Savers with Loans** segment."
- **Segment Insights:** "Use email or SMS reminders for follow-ups, Customers in this cluster have **lower engagement** and a **lower likelihood of subscribing** (~45%)."

## 📎 Folder Structure

```
BankingSegmentation/
|---artifacts/
|   |--model.pkl
|   |--preprocessor.pkl
|   |--pca.pkl
|   |--train.csv
|   |--test.csv
|   |--kmeans_scores.csv
|   |--agglo_scores.csv
|   |--db_scores.csv
|   |--labeled_train.csv
|   |--labeled_test.csv
|___ Notebook/
|    |--Data/
|      |--ban_marketing.csv
|    |--EDA.py
│── src/
|   |---components/
|   |    |---data_ingestion.py
|   |    |---data_transformation.py
|   |    |---dimentionality_reduction.py
|   |    |---cluster.py
|   |    |---model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   ├── exception.py
|   |__ logger.py
|   |__ utils.py
│── app.py
│── requirements.txt
│── README.md
│── setup.py
```

## 🤝 Contributing

Feel free to submit issues and pull requests to improve the project!

---

🔥 **Vipina Manjunatha** 🔥
Mail me at vipina1394@gmail.com

