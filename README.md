# Air Fare Analysis & Price Prediction

What drives domestic economy air fares in India, and how well can a ticket's price be predicted?

## Data
Kaggle flight price dataset (`Clean_Dataset.csv`): 206,666 economy-class flights across 6 airlines (Vistara, Air India, IndiGo, GO FIRST, AirAsia, SpiceJet) and 6 cities, with booking lead time (`days_left`), duration, stops, departure/arrival time and price. Fares range from ₹1,105 to ₹42,349 (median ₹5,772). The original dataset also includes Business class, which is filtered out.

## What I did
1. **Cleaning:** removed missing rows, kept economy flights only, and dropped the CSV row-index column so it isn't used as a feature.
2. **EDA:** price distribution and outliers; fares by booking lead time, duration, airline, route and arrival time.
3. **Modelling:** label-encoded categorical columns, split into train/test (80/20), scaled features using the training data only, and compared 10 regression models (Linear, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, Bagging, Gradient Boosting, XGBoost, KNN) on MAE, RMSE, R² and MAPE.

## Key findings
- **Booking lead time is the biggest driver:** price is strongly negatively correlated with days left before departure (r = −0.56), so fares climb sharply in the final days.
- **Duration matters much less** (r = 0.29).
- **Airline matters:** Vistara and Air India average around ₹7,300–7,800, while AirAsia averages about ₹4,100.
- **Tree-based models perform best:** Random Forest explained ~89% of price variation on the test set (R² = 0.89, MAE ≈ ₹503, MAPE ≈ 7%), compared with 42% for linear regression, showing that fare pricing is highly non-linear.

## Run it
Open `air_fare_analysis.ipynb` in Google Colab, upload `Clean_Dataset.csv`, and run all cells.

**Tools:** Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost
