# Stock Recommendation Project
Our project is to design a stock recommender system that recommends stock portfolio holding decisions and similar stock inside portfolio to user, basing on stock features and user past behaviors. 
Our project utilizes lightgbm as baseline model and d6tflow to manage tasks with features-included stock data. 
- Lightgbm: A efficient gradient boosting framework that uses tree based learning algorithms. 
- D6tflow: A python library which makes it easier to build data workflows specialized for data scientists and data engineers.
- Data: We have features data of the stocks within a certain amount of funds during a certain amount of time.

## Motivation for our project
- Recommending stocks to PMs and analysts
  - With thoroughly eda analysis on the whole dataset
  - With specific information about each fund/date
- Solve problems with recommender systems
  - Recommend actions of buy or sell
  - Recommend similar stocks in a fund
- Improving our recommendation based on the baseline model
  - Parameter research
  - Comparison between other models

## Main deliveries
- Analytic results (EDAs)
  - The EDA files generates thorough analysis on the entire dataset
- Product
  - Takes input of fund, stock, and date from user
  - Recommendation of buy-sell decisions
  - Recommendation of similar stocks within the portfolio

## EDA results
- Model eda
  - Imbalanced data
  - Accuracy score
  - Feature importance
  - Predicted probability 
  - Shap values (top 5 of all, enter recommend, exit recommend)
- Holdings eda
  - Number of holdings 
  - Number of positions new/exited
  - Features boxplot over time (all, enter recommend, exit recommend)
  - Enter and exited stock median



