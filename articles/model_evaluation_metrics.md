# Understanding Model Evaluation Metrics in Fraud Detection: Beyond Accuracy
Let's be real - if you're reading this, you've probably had that moment where you had to explain to stakeholders why your model's 99% accuracy is actually terrible. We've all been there. Let's dive into why traditional metrics fail us in fraud detection and what actually works in production.

## The Challenge: When 99% Accuracy is Actually Bad

In fraud detection, achieving 99% accuracy is not just insufficient - it's a red flag. Let's examine why:

### The Numbers Game
In our dataset spanning 2019-2020, we observed months with only 258 fraudulent transactions among 139,538 total transactions (0.18%). The European Bank Authority's 2024 report shows even more extreme imbalance: fraud represents only 0.015% of total card payments in Q1 2023. This severe class imbalance fundamentally changes how we must evaluate model performance.

### Why Traditional Metrics Lie
Here's a fun thought experiment: A model that simply predicts "not fraud" for all transactions would achieve:
- Accuracy: 99.985%
- Precision: Undefined (0/0)
- Recall: 0%
- F1 Score: 0%

While technically achieving high accuracy, such a model would be completely ineffective for fraud detection. It's like having a security guard who just waves everyone through - technically very efficient, but not exactly doing their job.

## The Real Metrics That Matter
Lets have a look at the confusion matrix:
<img src="../images/metrics/enhanced_confusion_matrix.png" width="50%" alt="Enhanced Confusion Matrix">

The confusion matrix above illustrates the fundamental components of model evaluation:
- **True Positives (TP)**: Correctly identified fraud cases
- **False Positives (FP)**: Legitimate transactions incorrectly flagged as fraud
- **True Negatives (TN)**: Correctly identified legitimate transactions
- **False Negatives (FN)**: Missed fraud cases

### Precision, Recall, and F1 Score: Core Evaluation Metrics

Let's examine these metrics in the context of fraud detection:

**Precision**: \[ Precision = \frac{TP}{TP + FP} \]
- High precision indicates few false positives
- Critical for operational efficiency
- Impacts investigation resource allocation
- Translation: How many of your fraud alerts are actually fraud?

**Recall**: \[ Recall = \frac{TP}{TP + FN} \]
- High recall indicates comprehensive fraud detection
- Directly impacts financial risk
- Essential for regulatory compliance
- Translation: How many actual fraud cases did you catch?

**F1 Score**: \[ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} \]
- Harmonic mean of precision and recall
- Balances detection capability with operational efficiency
- Provides a single metric for model comparison
- Translation: The sweet spot between catching fraud and not driving your team crazy with false alarms

### Real-World Example: Model Performance Analysis

Our model's performance across different datasets:

| Metric    | Train    | Holdout  | OOT      |
|-----------|----------|----------|----------|
| Accuracy  | 98.97%   | 98.98%   | 99.15%   |
| Precision | 35.85%   | 35.62%   | 29.74%   |
| Recall    | 97.19%   | 96.61%   | 90.36%   |
| F1 Score  | 52.38%   | 52.05%   | 44.76%   |

This data perfectly illustrates why accuracy alone is misleading:
1. **High Accuracy (98-99%)**: Looks impressive but doesn't tell the whole story
2. **Moderate Precision (29-36%)**: Shows that about 1/3 of our fraud alerts are actual fraud
3. **High Recall (90-97%)**: Indicates we're catching most of the actual fraud cases
4. **Moderate F1 Score (44-52%)**: Reflects the trade-off between precision and recall

## ROC-AUC vs PR-AUC: Technical Analysis

### Why ROC-AUC Can Be Deceptive

![ROC Curve](../images/metrics/roc_curve.png)

ROC curves plot True Positive Rate (TPR) against False Positive Rate (FPR) across different threshold values:
\[ TPR = \frac{TP}{TP + FN} \]
\[ FPR = \frac{FP}{FP + TN} \]

While widely used, ROC curves can be misleading in highly imbalanced datasets.:
- ROC-AUC can remain high even with poor minority class performance, since the focus is on majority class
- alse Positive Rate (FPR) becomes less meaningful due to the large number of true negatives
- The curve's shape may not reflect practical model utility
- Translation: ROC-AUC is like that friend who's always too optimistic

### Why PR-AUC is Your Friend

![PR Curve](../images/metrics/PR_curve.png)

PR curves plot Precision against Recall:
\[ Precision = \frac{TP}{TP + FP} \]
\[ Recall = \frac{TP}{TP + FN} \]
The PR-AUC plots provide a more realistic picture of model performance in fraud detection scenarios, as they better capture the 
challenges of detecting rare fraudulent transactions while maintaining reasonable precision. This is particularly important in fraud 
detection where both false positives (blocking legitimate transactions) and false negatives (missing fraud) have significant business 
implications.


Advantages in fraud detection:
- Directly focuses on the minority class
- More sensitive to changes in false positive rate
- Better reflects operational impact
- Provides clearer insight into model's practical utility
- Translation: PR-AUC is like that brutally honest friend who tells you exactly what you need to hear

### Performance Analysis Across Datasets

Our model's performance reveals:
- PR-AUC: Shows significant overfitting (0.4 drop from Training to OOT)
- ROC-AUC: Remains stable (around 0.99) but masks underlying issues
- Key indicators of overfitting:
  - Poor generalization to new data
  - Precision degradation
  - Recall instability

## Best Practices: Technical Implementation

1. **Metric Selection**
   - Primary: PR-AUC for overall performance
   - Secondary: Precision and Recall for specific aspects
   - Business metrics for operational impact

2. **Threshold Optimization**
   - Consider cost matrix for false positives vs false negatives
   - Implement multiple thresholds for different risk levels
   - Regular recalibration based on performance monitoring

3. **Validation Strategy**
   - Stratified cross-validation to maintain class distribution
   - Out-of-time validation for temporal stability
   - Multiple evaluation periods for robustness

## The Bottom Line
Remember: In fraud detection, the perfect model doesn't exist. The goal is to build something that:
- Catches enough fraud to protect your business
- Doesn't drive your customers crazy with false alarms
- Keeps working as fraud patterns change
- Makes business sense

Ready to dive deeper into any of these topics? Check out our other articles on model stability, sampling strategies, and time series analysis! 