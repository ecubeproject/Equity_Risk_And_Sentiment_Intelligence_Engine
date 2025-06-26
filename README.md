# Equity_Risk_And_Sentiment_Intelligence_Engine
Here we demonstrate how using NLP, PySpark, Azure NER, FastAPI and Docker we can deploy an app in azure cognitive services to (A) Do sentiment analysis of equity analyst's commentary (B) Do risk classification based on sentiment (C) Use Named Entity Recognition (NER) to Extract key words from announcements to give explainability to annuncement
# Risk_and_Sentiment_Intelligence_Engine with Named Entity Recognition (NER)
Risk and Sentiment Intelligence Engine for Analyst Ratings Using PySpark and Azure Cognitive Services
# Risk and Sentiment Intelligence Engine for Analyst Ratings

This repository contains the code, notebooks, and deployment artifacts for the **Risk and Sentiment Intelligence Engine**, developed as part of a final project using PySpark, Machine Learning, and Azure services.

## 🚀 Project Overview

The project automates the classification of financial analyst rating headlines by:

- **Classifying sentiment** into Positive, Neutral, or Negative.
- **Flagging risk** in headlines using a binary classifier.
- **Extracting named entities** (e.g., company names) using Azure Cognitive Services (NER).
- **Deploying** the solution via Docker and Azure App Service.

---

## 📂 Repository Structure
![Repo Structure](https://raw.githubusercontent.com/aimldstejas/Risk_and_Sentiment_Intelligence_Engine/main/image.png)

---

## 📊 Problem Statement

Financial institutions rely on quick and accurate interpretation of market news and analyst ratings. However, these are often high-volume and unstructured, requiring automation to extract:

- Sentiment signal (Positive/Neutral/Negative)
- Risk indicator (High Risk or Not)
- Named entities (e.g., company names)

Our goal was to design a scalable, deployable system to automate this workflow.

---

## 📈 Technical Stack

| Layer           | Technologies Used                                   |
|----------------|------------------------------------------------------|
| Data Processing | PySpark, Pandas                                     |
| Modeling        | XGBoost, Scikit-learn                               |
| API Framework   | FastAPI                                             |
| Deployment      | Docker, Azure App Service, Azure Container Registry |
| NER             | Azure Cognitive Services - Language API             |

---

## 🔍 How It Works

### Stage A: Sentiment Classification  
Multiclass classification using CountVectorizer + XGBoost  
→ Classes: Positive, Neutral, Negative

### Stage B: Risk Flag Classification  
Binary classification model (XGBoost) predicts whether the headline is **High Risk (1)** or **Not Risky (0)**.

### Stage C: NER + Deployment  
Azure Language Service extracts company names and financial terms.  
The entire app is containerized and deployed to Azure App Service.

---

## 📦 API Endpoints

Available via FastAPI Swagger UI:
[https://analystappf2-g8cphqa7bddsg7hc.centralindia-01.azurewebsites.net/docs](https://analystappf2-g8cphqa7bddsg7hc.centralindia-01.azurewebsites.net/docs)

| Endpoint                | Description                              |
|-------------------------|------------------------------------------|
| `/predict`              | Predicts sentiment of a given headline   |
| `/predict_risk`         | Flags if a headline indicates high risk  |
| `/predict_with_entities`| Performs NER and returns enriched output |

---

## 📉 Model Performance

| Model Type         | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Sentiment Model    | 98.8%    | ~0.99     | ~0.99  | ~0.99    |
| Risk Classifier    | 98.1%    | 0.94–1.00 | 0.82–0.84 | 0.89–0.90 |

(Weighted averages; see notebook for full metrics)

---

## 🧪 Run Locally

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/risk-sentiment-engine.git
   cd risk-sentiment-engine
````

2. Build Docker image

   ```bash
   docker build -t analystapi -f api/Dockerfile .
   ```

3. Run with Uvicorn

   ```bash
   uvicorn api.app.main:app --reload
   ```

4. Access: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ☁️ Azure Deployment (Final Configuration)

* App Name: `analystappf2`
* Container Image: `analystapi2025.azurecr.io/analystapi:latest`
* Hosted via: Azure App Service (Linux)
* Public URL:
  [https://analystappf2-g8cphqa7bddsg7hc.centralindia-01.azurewebsites.net](https://analystappf2-g8cphqa7bddsg7hc.centralindia-01.azurewebsites.net)

---

## 📄 License

Data is sourced from [Kaggle Stock News Dataset](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests), licensed under **CC0 (Public Domain)**.

This project is released under the [MIT License](LICENSE).

---

## 👥 Team

* **Member 1**: Role (e.g., Modeling, EDA)
* **Member 2**: Role (e.g., NER, Deployment)
* **Member 3**: Role (e.g., Integration, Documentation)

---

## 📬 Contact

For queries or contributions, please contact: [your.email@domain.com](mailto:your.email@domain.com)

```

---

Here’s a **Future Scope** section you can include in your report, README, or presentation — outlining possible real-world enhancements to your project:

---

## 🔮 Future Scope

While the current system effectively processes analyst rating headlines for sentiment, risk, and named entities, several enhancements can improve its usability, accuracy, and business impact:

### 1. **Interactive Front-End User Interface**

* Replace the current JSON-based Swagger UI with a **user-friendly web dashboard** using frameworks like **Streamlit**, **Dash**, or **React + FastAPI**.
* Allow business users to paste or upload multiple headlines and instantly visualize sentiment/risk results with entity highlights and charts.

### 2. **Topic Modeling Integration**

* Use **LDA or BERTopic** to extract emerging themes and topics from large-scale analyst news datasets.
* Help portfolio managers understand market narratives (e.g., M\&A, layoffs, ESG trends) beyond just sentiment scores.

### 3. **Real-Time Financial News Ingestion**

* Integrate APIs from financial news providers (e.g., **NewsAPI**, **Alpha Vantage**, **Refinitiv**) to fetch and process headlines in real-time.
* Enable **streaming classification** to detect risks as they emerge in the market.

### 4. **Multilingual & Cross-Market Capability**

* Extend model capabilities to support **non-English headlines** (e.g., Japanese, German markets).
* Use **Azure Translation API** or **mBERT-based multilingual classification models**.

### 5. **Improved Explainability and Transparency**

* Integrate **SHAP or LIME** explanations for each prediction (especially for risk flagging) to support compliance and analyst trust.

### 6. **Feedback Loop for Human-in-the-Loop Learning**

* Add options for users to **validate or correct predictions** (thumbs up/down), and use this data to fine-tune models periodically.
* Supports continuous learning in production environments.

### 7. **Deployment to Azure Kubernetes Service (AKS)**

* Scale the app for enterprise use via **Azure Kubernetes Service (AKS)** instead of App Service.
* Enable load balancing, autoscaling, and secure CI/CD integration using GitHub Actions or Azure DevOps.

### 8. **Custom Entity Recognition**

* Extend NER using **custom-trained Azure Language models** or **SpaCy custom pipelines** to identify domain-specific entities like:

  * `Analyst Firm`
  * `Recommendation Type` (e.g., Downgrade, Buy Initiation)
  * `Financial Events` (e.g., bankruptcy, merger)

### 9. **Analytics Dashboard for Decision Makers**

* Build a live dashboard to **aggregate classified headlines**, show risk/sentiment heatmaps over time, and drill down by stock, sector, or publisher.
* Tools: **Power BI**, **Plotly Dash**, or **Grafana**.

### 10. **Integration with Internal Enterprise Systems**

* Provide APIs or connectors to feed predictions into internal systems like:

  * **CRM tools**
  * **Portfolio management platforms**
  * **Investment research dashboards**

---

