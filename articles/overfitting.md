# Debiasing Fraud Detection Models: A Deep Dive into Sampling Strategies

Fraud detection is one of the most challenging problems in machine learning, primarily due to its inherently imbalanced nature. In this article, we'll explore how to handle bias in fraud models and present some effective strategies for building more robust fraud detection systems.

Disclamer: this article presents only the standard downsampling method ad debiasing technique, but many others can be valid depending on the use case.

## The Challenge: Imbalanced Data in Fraud Detection

Fraud detection presents a unique challenge in machine learning due to its inherently imbalanced nature and the inherent uncertainty in class labels. In real-world scenarios, fraudulent transactions typically represent less than 1% of all transactions, creating a severe class imbalance. For instance, in our dataset spanning 2019-2020, we observe months where only 258 fraudulent transactions occur among 139,538 total transactions (0.18%). In real word scenrios this ration is even less: the European Bank Authority reported in '2024 REPORT ON PAYMENT FRAUD' that in Q1 2023 fraud where 0.015% of total card payments in volume.
This extreme imbalance is further complicated by the "dirty" nature of the majority class - transactions labeled as legitimate aren't guaranteed to be truly non-fraudulent. Some fraudulent activities may go undetected due to sophisticated fraud schemes, delayed discovery, or limitations in fraud detection systems. This means our "non-fraud" class potentially contains hidden fraudulent transactions, introducing noise into our training data. The combination of class imbalance and uncertain labeling creates several critical problems: models naturally bias towards the majority class (legitimate transactions) since predicting "not fraud" for everything would still achieve 99%+ accuracy, making traditional accuracy metrics misleading. Additionally, the scarcity of confirmed fraud examples, coupled with the uncertainty in non-fraud labels, makes it difficult for models to learn meaningful patterns in fraudulent behavior. This imbalance and label uncertainty can lead to models that perform well on paper but fail to detect actual fraud in production, potentially costing businesses millions in fraudulent transactions while also risking false positives that could harm legitimate customer experiences.

Technically, the main challenges are:
* Models tend to be biased toward the majority class (non-fraudulent transactions)
* Traditional accuracy metrics become misleading
* The model might fail to learn patterns in the minority class

### Why is unbalance a problem

## Three Approaches to Model Building

Class imbalance introduces a strong bias in the training process. When left unaddressed, this bias leads models to favor the majority class (non-fraudulent transactions), essentially learning to be overly optimistic and potentially missing critical fraud patterns. However, simply addressing the imbalance through basic resampling techniques can introduce its own set of problems, including overfitting and unstable model performance.
To demonstrate the evolution of fraud detection modeling and its best practices, I'll present three different approaches, each building upon the limitations of the previous one:
1. **The Baseline Approach**: A standard classification model without any sampling techniques, illustrating why conventional methods fall short in fraud detection.
2. **The Basic Debiasing Solution**: Introduction of simple downsampling techniques to address class imbalance, showing both improvements and limitations of this approach.
3. **The Advanced Cross-Validation Framework**: A sophisticated solution that combines strategic sampling with robust cross-validation, representing the current best practice for fraud detection modeling.

By examining these approaches in sequence, we'll understand not just how to build better fraud detection models, but why certain techniques work better than others. This progression will highlight the critical importance of maintaining model stability while addressing class imbalance, ultimately leading to more reliable fraud detection systems.
Let's dive into each approach, examining their implementation, performance, and real-world implications.

### 1. Standard Model (No Debiasing)
Let's examine our first approach - training a model without any sampling techniques. This serves as our baseline and helps illustrate why conventional methods fall short in fraud detection.

```python
# Data preparation
X = train.drop(columns=['is_fraud'])
X.drop(metadata_columns, axis=1, inplace=True)
y = train['is_fraud']

# Basic train-test split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Model configuration
hyperparameters = {
    "objective": "binary",
    "n_estimators": 500,
    "early_stopping_round": 10,
    "first_metric_only": True,
    "num_leaves": 10,
    "min_data_in_leaf": 20
}

# Train the model
lgbm_model = LGBMClassifier(**hyperparameters)
model = lgbm_model.fit(X=train_X,
                      y=train_y,
                      eval_set=[(test_X, test_y)],
                      eval_metric='average_precision')
```

#### Model Performance:
Lets look at performances across the different datasets:

**Metrics Comparison**
| Metric |Train|Holdout|Out of time|Δ Train-OOT |
|:-----------|----------|----------|----------|-------------|
|Accuracy|0.997707|	0.997034|0.995655|-0.002%|
|Precision|0.833782|0.743860|0.452146|-38,16%|
|Recall|0.755238|0.715250|0.638668|11,65%|
|F1|0.792569|0.729274|0.529460|26,31%|

This reveals a classic case of overfitting in fraud detection:
1. Deceptive Accuracy:
While accuracy remains consistently high across all sets (only 0.002% decay), this is misleading due to the class imbalance. The model is essentially "mastering" the prediction of non-fraudulent cases, which represent the vast majority of transactions.
2. Deteriorating Precision:
The sharp decline in precision from training to OOT (a drop of nearly 40 percentage points) is a clear red flag. The model's ability to correctly identify frauds becomes significantly worse on unseen data.
3. Unstable Recall:
The declining recall indicates the model is missing an increasing proportion of actual fraud cases as we move from training to out-of-time data.

**Precision Recall Curves**

<img title="PR Auc performance of base model" alt="Alt text" src="/images/baselinemodel_pr.png">

The Precision-Recall Area Under Curve (PR-AUC) plot provides a clear visualization of the model's performance degradation across different datasets. PR-AUC curve drops significantly when moving from training to out-of-time data, with the area under the curve decreasing from 0.8 to 0.4. This substantial drop in PR-AUC (a decrease of approximately 40%) further confirms the model's poor generalization and its struggle to maintain consistent performance on new data. The steep decline in the curve for the OOT dataset indicates that the model cannot maintain high precision without severely sacrificing recall, making it impractical for real-world fraud detection where both metrics are crucial.

#### Considerations

This model is not reliable as the substantial performance gap between training and OOT metrics suggests the model is learning patterns that don't generalize well to new fraud patterns, a critical flaw in fraud detection where patterns evolve rapidly.
Moreover, the declining recall (down to 63.87% in OOT) means the model is missing about 36% of actual fraud cases, exposing the business to significant financial risk.

This analysis clearly demonstrates why a standard modeling approach is insufficient for fraud detection. The combination of class imbalance and the temporal nature of fraud patterns requires more sophisticated techniques, which we'll explore in our next approaches.

NOTE [TODO]: not big fan of scale_pos_weight param as it that the usage of all these parameters will result in poor estimates of the individual class probabilities (https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.n_iter_) and it applyis on the whole dataset.

### 2. Simple Downsampling
The second approach involves downsampling the majority class (non-fraudulent transactions) in the training set. This is implemented using scikit-learn's RandomUnderSampler:

```python
# Import necessary libraries
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Initialize the undersampler with a sampling ratio of 0.2 (1:5 fraud to non-fraud ratio)
undersample_func = RandomUnderSampler(sampling_strategy=0.2, random_state=42)

# Apply undersampling to the training data only
X_train_resampled, y_train_resampled = undersample_func.fit_resample(X_train, y_train)

# Train the model with the same hyperparameters as baseline
# Train the model on resampled data
lgbm_model = LGBMClassifier(**hyperparameters)
model = lgbm_model.fit(X=X_train_resampled,
                      y=y_train_resampled,
                      eval_set=[(X_test, y_test)],
                      eval_metric='average_precision')
```

This approach:
+ Improves fraud detection capability
+ Reduces bias toward the majority class
- But may lose important information from the majority class

The sampling ratio of 0.2 means we keep all fraud cases and randomly sample non-fraud cases to achieve a 1:5 ratio, which significantly reduces class imbalance while still maintaining some of the natural class distribution's characteristics.

#### Model Performance:

Lets look at performances across the different datasets:

**Metrics Comparison**
| Metric | Train | Holdout | OOT | Δ Train-OOT |
|-----------|----------|----------|----------|-------------|
| Accuracy | 0.989747 | 0.989768 | 0.991462 | +0.17% |
| Precision | 0.358474 | 0.346832 | 0.297448 | -6.10% |
| Recall | 0.971902 | 0.941970 | 0.903579 | -6.83% |
| F1 | 0.523764 | 0.506991 | 0.447563 | -7.62% |


The performance metrics reveal significant improvements in handling overfitting compared to our baseline approach:

1.More Stable Precision Degradation:
* Precision starts at a more modest 0.36 (vs. baseline's 0.83)
* The lower initial precision is actually more realistic, avoiding the baseline's overconfident predictions
* Performance degradation is more gradual: 6% drop (vs. baseline's 38% drop)

2. Improved Fraud Detection Stability:
* A high recall is maintained across all sets (only -6% drop between sets)
* Significantly better than baseline's OOT recall of 0.64
* Model retains ability to detect frauds even as patterns evolve

3. More Consistent Performance Decay:
>Precision: 0.35 → 0.34 → 0.30 (gradual decline)\
>Recall: 0,97 → 0.50 → 0.45 (controlled degradation)
Compared to baseline's sharp drops, these smoother transitions indicate better generalization

**Precision Recall Curves**
The PR curve further supports these improvements:

<img title="PR Auc performance of base model" alt="Alt text" src="/images/debiasmodel_pr.png">

* Smaller gap between training and holdout curves
* More gradual performance decline in OOT
* Better maintained relationship between precision and recall across datasets

#### Considerations

While the lower precision might seem concerning, the model's improved stability and robust fraud detection capabilities suggest it's learning more generalizable patterns than the baseline approach. However, the trade-off between precision and recall indicates room for further improvement through more sophisticated techniques.

This analysis demonstrates that while simple downsampling isn't perfect (note the perfect training recall), it significantly reduces overfitting compared to the baseline.

### 3. Cross-Validation with Dynamic Downsampling
The dynamic downsampling is implemented using scikit-learn's Pipeline and GridSearchCV, ensuring that downsampling is performed properly within each cross-validation fold:

```python
from imblearn.pipeline import Pipeline

hyperparameters = {
        #'encode__columns': [categorical_columns],
        'class__n_estimators': [20, 50, 200, 500],
        "class__objective": ["binary"],
        "class__early_stopping_round": [10],
        "class__num_leaves": [5, 10, 80, 100],
        "class__min_data_in_leaf": [10, 50, 100, 200],
    }

undersample_pipe = Pipeline([('sampling', RandomUnderSampler(sampling_strategy=0.1, random_state=42)) 
                             , ('class', LGBMClassifier(objective='binary'))])

score_balanced_parameter_model = GridSearchCV(undersample_pipe, param_grid=hyperparameters, cv=3, scoring='average_precision')

score_balanced_parameter_model.fit(train_X, train_y, class__eval_set=(holdout_X, holdout_y))
```



#### Model Performance:
Lets look at performances across the different datasets:

**Metrics Comparison**
| Metric | Train | Holdout | Out of Time | Δ Train-OOT |
|:-----------|----------|----------|----------|-------------|
| Accuracy | 0.995310 | 0.994724 | 0.994951 |-0.04% |
| Precision | 0.552941 | 0.515483 | 0.420662 |-13.23% |
| Recall | 1.000000 | 0.921053 | 0.845924 |-15.41% |
| F1 | 0.712121 | 0.661017 | 0.561902 |-15.02% |


1. Balanced Precision-Recall Trade-off:
* Precision starts at 0.55, more balanced than baseline (0.83) and simple downsampling (0.36)
* Recall maintains high performance (1.00 → 0.92 → 0.85)
* Better balance between false positives and missed frauds

2. Improved Stability Across Time:
* Accuracy remains remarkably stable (only 0.04% drop)
* Precision decay is more controlled: 0.55 → 0.52 → 0.42
* Recall maintains strong performance even in OOT (0.85)

3. More Robust Learning:
* F1 score shows gradual decline (0.71 → 0.66 → 0.56)
* Performance drops are more predictable
* Better generalization to unseen patterns


**Precision Recall Curves**
The PR curve demonstrates these improvements:

<img title="PR Auc performance of base model" alt="Alt text" src="/images/balanced_parameter_model_pr.png">

* Smoother transitions between training and holdout curves
* More consistent area under the curve across datasets
* Better maintained precision-recall relationship in OOT

#### Considerations

Key Advantages Over Previous Approaches:
vs. Baseline: More stable metrics, better fraud detection, less severe overfitting
vs. Simple Downsampling: Better precision, more balanced metrics, improved generalization
The dynamic CV approach shows superior handling of overfitting through:
* More realistic training performance
* Better retention of model capabilities in OOT
* More balanced precision-recall trade-off
* Consistent performance across different time periods

### 4. Bonus Method
BalancedBaggingClassifier

## Best Practices for Implementation
1. Only Downsample Training Data

```python
# Correct approach
train_df, y_train = self.undersample_func.fit_resample(train_df, y_train)
# Keep validation data as is
test_df, y_test = dataframe.iloc[test_index], y[test_index]
```

2. Use Stratified Cross-Validation:

```pyhton
skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
```

3. Use sampling in Stratified KFold

4. Monitor Multiple Metrics:
```python
eval_metric=["average_precision", "auc"]
```

## Conclusion
While class imbalance in fraud detection presents significant challenges, proper sampling techniques can substantially improve model performance. The cross-validation with dynamic downsampling approach offers the best balance between maintaining information richness and reducing bias, leading to more robust and reliable fraud detection models.
Remember:
* Never downsample validation or test sets
* Implement sampling within cross-validation folds
* Use appropriate metrics for imbalanced classification
* Consider the business impact of false positives vs. false negatives

By following these principles, you can build more effective fraud detection systems that better serve real-world applications.