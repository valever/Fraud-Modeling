# Debiasing Fraud Detection Models: A Deep Dive into Sampling Strategies

Picture this: You've spent weeks building your fraud detection model. The training metrics look fantastic - 99% accuracy! Your stakeholders are impressed, and you're ready to deploy. But then reality hits: your model starts flagging legitimate transactions left and right, missing actual fraud cases, and generally making a mess of things. Sound familiar? You're not alone. This is the classic tale of overfitting in fraud detection, where our models become too "clever" for their own good.

The problem isn't just about model complexity - it's about the unique nature of fraud data. We're dealing with extremely imbalanced datasets where fraud cases are like needles in a haystack (and the haystack is on fire). In our dataset spanning 2019-2020, we observed months where only 258 fraudulent transactions occurred among 139,538 total transactions (0.18%). In real word scenrios this ration is even less: the European Bank Authority reported in their '2024 REPORT ON PAYMENT FRAUD' that in Q1 2023, fraud was only 0.015% of total card payments. That's not just imbalanced - that's severely imbalanced!

But wait, it gets worse. Our "non-fraud" class isn't even clean - some fraudulent activities might be hiding in there, undetected due to sophisticated schemes or delayed discovery or limitations in fraud detection systems. This means we're training our models on potentially noisy data, which is like trying to learn chess while some pieces keep changing their moves.

The combination of class imbalance and uncertain labeling creates several critical problems: models naturally bias towards the majority class (legitimate transactions) since predicting "not fraud" for everything would still achieve 99%+ accuracy, making traditional accuracy metrics misleading. Additionally, the scarcity of confirmed fraud examples, coupled with the uncertainty in non-fraud labels, makes it difficult for models to learn meaningful patterns in fraudulent behavior. This imbalance and label uncertainty can lead to models that perform well on paper but fail to detect actual fraud in production, potentially costing businesses millions in fraudulent transactions while also risking false positives that could harm legitimate customer experiences.

Technically, this creates three major headaches:
* Models get lazy and just predict "not fraud" for everything
* Traditional accuracy metrics become completely useless
* The model might miss important fraud patterns entirely

### Why is unbalance a problem?

Quick heads up before we dive in: While we're focusing on the most widely-used sampling technique here, there are plenty of other ways to tackle class imbalance. Don't feel boxed in - you might find that a different approach (or even a combination of methods) works better for your specific fraud detection challenge. The key is to experiment and find what fits your use case best!

## Three Approaches to Model Building

Class imbalance is like having a biased referee in a game - it makes the whole system unfair. When we don't address it, our models become overly optimistic and miss critical fraud patterns. But here's the catch: fixing it with basic resampling can create its own problems, like overfitting and unstable performance.

To demonstrate the evolution of fraud detection modeling and its best practices, let's look at three different approaches, each 
building on what we learned from the previous one:
1. **The Baseline Approach**: A standard classification model without any sampling techniques (spoiler: it's not great)
2. **The Basic Debiasing Solution**: Simple downsampling to address class imbalance (better, but still not perfect)
3. **The Advanced Cross-Validation Framework**: A sophisticated solution that combines strategic sampling with robust cross-validation (this is where it gets interesting)

As we walk through each approach, we'll not only learn how to build better fraud detection models, but also get a feel for why some 
techniques work better than others. Think of it as a journey from "meh" to "wow" - we'll see how each step helps us build more stable 
and reliable models that actually work in the real world.

Let's dive into each approach and see what works, what doesn't, and why.

### 1. Standard Model (No Debiasing)

First up, let's look at what happens when we don't do anything about the imbalance or the parameter tuning. This will serve as our baseline and show us exactly why we need better approaches.

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
Let's see how this basic approach performs across different datasets:

**Metrics Comparison**
| Metric |Train|Holdout|Out of time|Δ Train-OOT |
|:-----------|----------|----------|----------|-------------|
|Accuracy|0.997707|	0.997034|0.995655|-0.002%|
|Precision|0.833782|0.743860|0.452146|-38,16%|
|Recall|0.755238|0.715250|0.638668|11,65%|
|F1|0.792569|0.729274|0.529460|26,31%|

This is a classic case of overfitting in fraud detection:
1. **That Deceptive Accuracy**: Looks great, right? But remember our earlier warning about accuracy in imbalanced datasets...The model is essentially "mastering" the prediction of non-fraudulent cases,
2. **The Precision Plunge**: Watch that precision drop from 83% to 45% - that's our model's ability to correctly identify fraud going down the drain as he sees unseen data,
3. **Recall Reality Check**: The declining recall means we're missing more and more actual fraud cases as we move from training to 
out-of-time data.

**Precision Recall Curves**

<img title="PR Auc performance of base model" alt="Alt text" src="/images/baselinemodel_pr.png">

The Precision-Recall Area Under Curve (PR-AUC) plot tells a clear story: our model's performance is falling apart when it hits real-world data. The area under the curve drops from 0.8 to 0.4 from training to out-of-time data - that's like going from an A to an F in model performance. The steep decline in the curve for the OOT dataset indicates that the model cannot maintain high precision without severely sacrificing recall, making it impractical for real-world fraud detection where both metrics are crucial.

#### Considerations

This model is about as reliable as a chocolate teapot. The huge gap between training and OOT metrics means it's learning patterns that don't generalize well to new fraud patterns, a critical flaw in fraud detection where patterns evolve rapidly. And that 63.87% recall in OOT? That means we're missing about 36% of actual fraud cases - exposing the business to significant financial risk.

This analysis clearly demonstrates why a standard modeling approach is insufficient for fraud detection. The combination of class imbalance and the temporal nature of fraud patterns requires more sophisticated techniques, which we'll explore in our next approaches.

NOTE [TODO]: not big fan of scale_pos_weight param as it that the usage of all these parameters will result in poor estimates of the individual class probabilities (https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.n_iter_) and it applyis on the whole dataset.

### 2. Simple Downsampling

Now let's try something smarter - downsampling the majority class. This is like leveling the playing field by reducing the number of non-fraud cases we train on:

```python
# Import necessary libraries
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Initialize the undersampler with a sampling ratio of 0.2 (1:5 fraud to non-fraud ratio)
undersample_func = RandomUnderSampler(sampling_strategy=0.2, random_state=42)

# Apply undersampling to the training data only
X_train_resampled, y_train_resampled = undersample_func.fit_resample(X_train, y_train)

# Train the model on resampled data, with the same hyperparameters as baseline
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

Let's see how this improved approach performs:

**Metrics Comparison**
| Metric | Train | Holdout | OOT | Δ Train-OOT |
|-----------|----------|----------|----------|-------------|
| Accuracy | 0.989747 | 0.989768 | 0.991462 | +0.17% |
| Precision | 0.358474 | 0.346832 | 0.297448 | -6.10% |
| Recall | 0.971902 | 0.941970 | 0.903579 | -6.83% |
| F1 | 0.523764 | 0.506991 | 0.447563 | -7.62% |

This is much better! Let's break down what we're seeing:

1. **More Stable Precision**:
   - Starts at a more realistic 0.36 (vs. baseline's 0.83)
   - Performance degradation is more gradual: only 6% drop (vs. baseline's 38% drop)

2. **Better Fraud Detection**:
   - Maintains high recall across all sets (only -6% drop)
   - Much better than baseline's OOT recall of 0.64
   - Model keeps detecting frauds even as patterns evolve

3. **Smoother Performance Decay**:
   >Precision: 0.35 → 0.34 → 0.30 (gradual decline)\
   >Recall: 0.97 → 0.50 → 0.45 (controlled degradation)
   Compared to baseline's sharp drops, these smoother transitions indicate better generalization

**Precision Recall Curves**

<img title="PR Auc performance of base model" alt="Alt text" src="/images/debiasmodel_pr.png">

The PR curve shows these improvements:
* Smaller gap between training and holdout curves
* More gradual performance decline in OOT
* Better maintained relationship between precision and recall across datasets

#### Considerations

While the lower precision might seem concerning, this model is actually more reliable than our baseline. It's learning more generalizable patterns and maintaining better stability across different datasets. Sure, it's not perfect (that perfect training recall is a bit suspicious), but it's a huge improvement over our first attempt.

This analysis demonstrates that while simple downsampling isn't perfect (note the perfect training recall), it significantly reduces 
overfitting compared to the baseline.

### 3. Cross-Validation with Dynamic Downsampling: The Ensemble Resampling Technique

This is where it gets interesting - the state-of-the-art approach for fraud modeling. The problem with doing downsampling once for all CV is that we lose variability and information on the negative class. The solution? Sample in each iteration of cross-validation. A nice paper discovering the ensable resampling is ["Exploratory Undersampling for Class-Imbalance Learning"](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tsmcb09.pdf).

The dynamic downsampling is implemented using scikit-learn's Pipeline and GridSearchCV, ensuring that downsampling is performed properly within each cross-validation fold:

```python
from imblearn.pipeline import Pipeline

hyperparameters = {
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

Let's see how this advanced approach performs:

**Metrics Comparison**
| Metric | Train | Holdout | Out of Time | Δ Train-OOT |
|:-----------|----------|----------|----------|-------------|
| Accuracy | 0.995310 | 0.994724 | 0.994951 |-0.04% |
| Precision | 0.552941 | 0.515483 | 0.420662 |-13.23% |
| Recall | 1.000000 | 0.921053 | 0.845924 |-15.41% |
| F1 | 0.712121 | 0.661017 | 0.561902 |-15.02% |

This is where we start seeing real improvements:

1. **Balanced Precision-Recall Trade-off**:
   - Precision starts at 0.55, more balanced than our previous attempts
   - Recall maintains high performance (1.00 → 0.92 → 0.85)
   - Better balance between false positives and missed frauds

2. **Rock-Solid Stability**:
   - Accuracy remains remarkably stable (only 0.04% drop)
   - Precision decay is more controlled: 0.55 → 0.52 → 0.42
   - Recall maintains strong performance even in OOT (0.85)

3. **Robust Learning**:
   - F1 score shows gradual decline (0.71 → 0.66 → 0.56)
   - Performance drops are more predictable
   - Better generalization to unseen patterns

**Precision Recall Curves**

<img title="PR Auc performance of base model" alt="Alt text" src="/images/balanced_parameter_model_pr.png">

The PR curve shows these improvements:
* Smoother transitions between training and holdout curves
* More consistent area under the curve across datasets
* Better maintained precision-recall relationship in OOT

#### Considerations

This approach is the real deal. Compared to our previous attempts:
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

Here's what you need to remember when implementing these approaches:

1. **Only Downsample Training Data**
```python
# Correct approach
train_df, y_train = self.undersample_func.fit_resample(train_df, y_train)
# Keep validation data as is
test_df, y_test = dataframe.iloc[test_index], y[test_index]
```

2. **Use Stratified Cross-Validation**
```python
skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
```

3. **Use sampling in Stratified KFold**

4. **Monitor Multiple Metrics**
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
Do you want to know more on how to split your dataset in the best way? Read my article [beyondtimeseries.md]