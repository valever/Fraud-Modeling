# Understanding Overfitting in Fraud Detection: The Basics

Picture this: You've spent weeks building your fraud detection model. The training metrics look fantastic - 99% accuracy! Your stakeholders are impressed, and you're ready to deploy. But then reality hits: your model starts flagging legitimate transactions left and right, missing actual fraud cases, and generally making a mess of things. Sound familiar? You're not alone. This is the classic tale of overfitting in fraud detection, where our models become too "clever" for their own good.

## The Challenge: Imbalanced Data in Fraud Detection

The problem isn't just about model complexity - it's about the unique nature of fraud data. We're dealing with extremely imbalanced datasets where fraud cases are like needles in a haystack (and the haystack is on fire). In our dataset spanning 2019-2020, we observed months where only 258 fraudulent transactions occurred among 139,538 total transactions (0.18%). And get this - the European Bank Authority reported in their 2024 report that in Q1 2023, fraud was only 0.015% of total card payments. That's not just imbalanced - that's severely imbalanced!

But wait, it gets worse. Our "non-fraud" class isn't even clean - some fraudulent activities might be hiding in there, undetected due to sophisticated schemes or delayed discovery. This means we're training our models on potentially noisy data, which is like trying to learn chess while some pieces keep changing their moves.

Technically, this creates three major headaches:
* Models get lazy and just predict "not fraud" for everything
* Traditional accuracy metrics become completely useless
* The model might miss important fraud patterns entirely

Quick heads up before we dive in: While we're focusing on the most widely-used sampling technique here, there are plenty of other ways to tackle class imbalance. Don't feel boxed in - you might find that a different approach (or even a combination of methods) works better for your specific fraud detection challenge. The key is to experiment and find what fits your use case best!

## Two Basic Approaches to Model Building

Class imbalance is like having a biased referee in a game - it makes the whole system unfair. When we don't address it, our models become overly optimistic and miss critical fraud patterns. But here's the catch: fixing it with basic resampling can create its own problems, like overfitting and unstable performance.

Let's look at two approaches, each building on what we learned from the previous one:
1. **The Baseline Approach**: A standard classification model without any sampling techniques (spoiler: it's not great)
2. **The Basic Debiasing Solution**: Simple downsampling to address class imbalance (better, but still not perfect)

As we walk through each approach, we'll not only learn how to build better fraud detection models, but also get a feel for why some techniques work better than others. Think of it as a journey from "meh" to "better" - we'll see how each step helps us build more stable and reliable models.

### 1. Standard Model (No Debiasing)

First up, let's look at what happens when we don't do anything about the imbalance. This will serve as our baseline and show us exactly why we need better approaches.

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
1. **That Deceptive Accuracy**: Looks great, right? But remember our earlier warning about accuracy in imbalanced datasets...
2. **The Precision Plunge**: Watch that precision drop from 83% to 45% - that's our model's ability to correctly identify fraud going down the drain
3. **Recall Reality Check**: The declining recall means we're missing more and more actual fraud cases

**Precision Recall Curves**

<img title="PR Auc performance of base model" alt="Alt text" src="/images/baselinemodel_pr.png">

The PR-AUC plot tells a clear story: our model's performance is falling apart when it hits real-world data. The area under the curve drops from 0.8 to 0.4 - that's like going from an A to an F in model performance.

#### Considerations

This model is about as reliable as a chocolate teapot. The huge gap between training and OOT metrics means it's learning patterns that don't generalize well to new fraud patterns. And that 63.87% recall in OOT? That means we're missing about 36% of actual fraud cases - not exactly what you want in a fraud detection system.

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

# Train the model with the same hyperparameters as baseline
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

**Precision Recall Curves**

<img title="PR Auc performance of base model" alt="Alt text" src="/images/debiasmodel_pr.png">

The PR curve shows these improvements:
* Smaller gap between training and holdout curves
* More gradual performance decline in OOT
* Better maintained relationship between precision and recall

#### Considerations

While the lower precision might seem concerning, this model is actually more reliable than our baseline. It's learning more generalizable patterns and maintaining better stability across different datasets. Sure, it's not perfect (that perfect training recall is a bit suspicious), but it's a huge improvement over our first attempt.

## Basic Best Practices

Here's what you need to remember when implementing these basic approaches:

1. **Only Downsample Training Data**
```python
# Correct approach
train_df, y_train = self.undersample_func.fit_resample(train_df, y_train)
# Keep validation data as is
test_df, y_test = dataframe.iloc[test_index], y[test_index]
```

2. **Monitor Multiple Metrics**
```python
eval_metric=["average_precision", "auc"]
```

## Next Steps

While these basic approaches can help, there's a more sophisticated solution that combines strategic sampling with robust cross-validation. Check out our next article on advanced techniques for handling overfitting in fraud detection models. 