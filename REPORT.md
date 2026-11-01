# Can We Flag a Late Order Before It Ships?
### A leakage-free study of delivery risk using the DataCo Smart Supply Chain dataset

**Muhammad Imran Syed** · M.S. Data Science, University of St. Thomas

---

## 1. Business problem
Late deliveries make customers unhappy and cost companies money. If a company could tell **at the moment an order is placed** that it will be late, it could warn the customer, ship faster, or change the promise.
I asked three questions, one for each technique in my proposal:

1. **Classification:** will this order be late? (target: `Late_delivery_risk`)
2. **Regression:** how many days will it take? (target: `Days for shipping (real)`)
3. **Clustering:** which groups of customers exist, and which ones matter most?

## 2. Data
The DataCo Smart Supply Chain dataset (Instituto Politécnico de Leiria; Kaggle / Mendeley Data) has **180,519 rows and 53 columns** covering January 2015 to January 2018.
One row is one product inside an order. There are **65,752 orders** and **20,652 customers**, and all items in an order share the same late/on-time outcome.

**Cleaning:** I removed personal information (emails, passwords, names, street), 6 pairs of duplicate columns, empty or constant columns, and canceled orders (7,754 rows, which include all 4,062 suspected-fraud rows), because canceled orders never ship.

## 3. My approach: three rules
1. **Only information known at order time.** I never use actual shipping days, delivery status or order status as inputs.
2. **Test on the future.** I train on January 2015 – June 2017 (49,826 orders) and test on July 2017 – January 2018 (13,071 orders). Hyperparameters are tuned with time-ordered cross-validation on the training set only.
3. **One row per order**, so the same order never appears in both training and testing.

**Why this matters:** I found that `Late_delivery_risk` is exactly *actual days > promised days* for 100% of shipped orders. Public notebooks that report about 99% accuracy include actual shipping days as an input, so their models are reading the answer.

## 4. Exploratory analysis: what drives late delivery?
- **57% of shipped orders are late**, mostly by 1–2 days. The rate is flat at about 57% every quarter for three years.
- **Shipping mode is the key factor** (Cramér's V = 0.48, strong):
  - First Class always takes 2 days but promises 1, so it is **100% late**.
  - Second and Standard Class take the **same** 2–6 days, spread evenly. Second Class just promises 2 days instead of 4, so it's 80% late versus 40%.
- **A hidden noon cut-off:** Same Day orders placed before 12:00 are always on time; after 12:00 they are always late.
- **Region, market, customer segment, product category, payment type, weekday and month have no meaningful effect** (all Cramér's V ≤ 0.03, mostly not significant), even when tested within a single shipping mode.
- **Profit:** 21% of orders lose money, and those losses cancel out about 49% of the profit. But losses are not linked to late delivery, discount, product or market (about 19% of items lose money in every group).

From these patterns I calculated the **best accuracy any honest model can reach: about 72%**.

## 5. Classification results
Models: always-late baseline, Logistic Regression (with and without one engineered feature), Random Forest, and XGBoost tuned with `RandomizedSearchCV`. All use pipelines with one-hot encoding and scaling, and class weights for the mild imbalance.

| Model | Accuracy | Recall (late) | ROC-AUC |
|---|---|---|---|
| Always "late" | 57.6% | 100% | – |
| Logistic Regression | 69.6% | 53.9% | 0.753 |
| Logistic Regression + "Same Day after noon" feature | **72.6%** | 59.2% | **0.781** |
| Random Forest | **72.6%** | 59.2% | 0.780 |
| XGBoost (tuned) | **72.6%** | 59.2% | 0.779 |
| XGBoost, cost-based threshold (0.23) | 60.1% | **99.9%** | 0.779 |
| *Leaky Random Forest (uses actual shipping days)* | *100%* | *100%* | *1.000* |

- The best models reach the 72% ceiling exactly and match a simple business rule. **Feature engineering beat model complexity.**
- Permutation importance confirms that only shipping mode and order hour matter.
- **Threshold tuning:** assuming a missed late order costs 5 and a false alarm costs 1, the best threshold (chosen on validation data) is 0.23. Recall rises from 59% to 99.9%, and the total cost of mistakes falls by about two-thirds (15,872 → 5,233).
- The predicted probabilities are well calibrated.

## 6. Regression results
| Model | RMSE (days) | MAE (days) | R² |
|---|---|---|---|
| Overall average | 1.62 | 1.42 | 0.00 |
| Linear / Ridge / Random Forest / XGBoost | **1.26** | **0.97** | **0.39** |
| Calculated best possible | 1.26 | – | 0.39 |
| *Leaky model (uses late flag)* | *0.71* | *0.52* | *0.81* |

All honest models reach the calculated limit. First Class and Same Day are predicted to within 0.05 days. Second and Standard Class are about 1.2 days off, because their 2–6 day spread is random in this data. A second target, the *delay gap* (actual − promised days), gives the same error.

## 7. Clustering results
Country-level differences in late rate were no bigger than random chance (spread 0.029 versus 0.027 expected), so I clustered **customers**. Customer IDs reset from October 2017 (99% of later orders come from brand-new, one-time IDs), so I used orders before that date: **12,296 customers**.
Features: number of orders, total spend, profit margin, late share, premium-shipping share and days since the last order. I used **K-Means with k = 4** (chosen with the elbow method and the silhouette score) and confirmed the result with **Ward hierarchical clustering** (ARI = 0.60).

| Cluster | Customers | Share of profit | Key trait |
|---|---|---|---|
| Loyal high-value | 53% | **81%** | About 6 orders, about $3,200 spend, recently active |
| Premium shippers, often late | 21% | 18% | 61% premium shipping, so **78% late**; inactive |
| Occasional standard shippers | 16% | 13% | Only 24% late, but few orders |
| Loss-making customers | 10% | **−13%** | Average margin −29% |

The company's own segment labels (Consumer / Corporate / Home Office) are spread identically across the clusters, so the clusters add new information.

## 8. Recommendations
1. **Change the First Class promise to 2 days** (or truly ship it in 1). This removes about 27% of all late orders.
2. **Fix or re-price Second Class.** It ships no faster than Standard.
3. **Show the 12:00 cut-off for Same Day shipping** at checkout.
4. **Quote a range ("2–6 days")** for Second and Standard Class instead of a single date.
5. **Win back the "premium, often late" customers.**
6. **Review pricing and discounts for loss-making customers.**
7. **Don't spend on region- or product-specific delivery fixes.** These factors show no effect.

## 9. Limitations
- The dataset appears **partly simulated**: shipping days are evenly random within 2–6 days, the loss rate is about 19% in every group, and 96% of customers return whether or not their first order was late.
- Customer IDs reset from October 2017.
- There is no carrier, warehouse or distance information.
- The cost weights used for the threshold (5 : 1) are assumptions.

## 10. Conclusion
When only information available at order time is used, the best possible accuracy for predicting late delivery is about 72%, and Logistic Regression (with one engineered feature), Random Forest and XGBoost all reach it. The 99% results reported elsewhere come from data leakage. The data shows that delays are driven by **unrealistic shipping promises**, not by region, customer or product. The most valuable action for the company is to fix those promises, not to build a more complex model.

---
**Project files:** notebooks `01`–`06` in `notebooks/` (all run on Google Colab), charts in `figures/`, and presenter notes `STEP1`–`STEP5_presenter_notes.md`.
