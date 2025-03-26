# Feature Engineering for Fraud Detection: The Time Factor

"Your model looks amazing in testing! But why isn't it catching any fraud in production?" 

If you've ever heard this (or said it yourself), you're not alone. After a decade of building fraud detection systems, I can tell you that this is one of the most common problems I see. And surprisingly, it's not because of complex algorithms or fancy models - most of the time it's because of how we handle time.

In our last article, we explored how time affects data splitting. Now, let's dig into something even trickier: how to build features that actually work in the real world, where you can't peek into the future. This article is part of a series where we explore different aspects of fraud detection. If you haven't already, check out our articles on [model_evaluation_metrics.md], [model_stability_metrics.md], and [beyondtimeseries.md] - they'll give you a solid foundation for what we're about to discuss.

## The Time Factor in Fraud Detection

Here's a story that might sound familiar: A team builds a fraud detection model that looks incredible during testing - we're talking near-perfect scores. Everyone's excited. Then they deploy it... and suddenly their amazing model can barely catch any fraud. What happened?

It might be overfitting or the answer can lie in how we built our features. In development, it's surprisingly easy to accidentally use information that wouldn't be available in the real world. Think about it - when a transaction happens in production, you only have information up to that exact moment. No peeking at fraud labels that get confirmed weeks later, no using patterns that haven't emerged yet.

But here's the tricky part - this mistake is so common that I've seen it happen to everyone from junior developers to seasoned ML engineers. It's subtle, it's sneaky, and it can make your model look amazing in testing while setting it up for failure in the real world.

Ready to learn how to spot and fix this? Let's dive in.

## The Label Maturity Challenge

Let me tell you about one of the biggest gotchas in fraud detection - label maturity. Here's what you need to know:

- Fraud labels can take 30, 60, or even more days to mature after a transaction
- Your dataset should have a column showing when each fraud label was updated
- When building past behavioral features, always check the label's timestamp

Let's look at a real example:

| transaction_id | transaction_date | is_fraud | is_fraud_date |
|---------------|------------------|----------|---------------------|
| a12345        | 01/02/2024      | 0        |                     |
| a12367        | 20/02/2024      | 1        | 10/03/2024         |
| b14891        | 05/03/2024      | 1        | 01/05/2024         |
| c1511         | 20/04/2024      | 1        | 15/06/2024         |

See how those fraud labels take their sweet time to show up? That's why we need to be super careful with our feature engineering.

## The Data Leakage Trap

Here's where it gets interesting. Let's say you want to know how many fraudulent transactions a customer had in the past. Sounds simple, right? But watch out - there's a trap!

Let's look at our example data. If we want to predict for transaction `c1511` (20/04/2024), here's what NOT to do:

```sql
select sum(is_fraud) 
from transactions
where transaction_date < '20/04/2024'
```

This would give you 2 fraudulent transactions. But wait! The fraud label for transaction `b14891` wasn't even confirmed until 01/05/2024 - that's after our prediction date! We're using information from the future! 🚫

Here's the right way to do it:

```sql
select sum(is_fraud) 
from transactions
where is_fraud_date < '20/04/2024'
```

Now we correctly get 1 fraudulent transaction. The image below shows this concept in action - we only use information from the green area (the past) when making our prediction.

<img title="Feature engineering period" alt="Alt text" src="/images/feature_eng_flow.jpg">

## The Golden Rule of Feature Engineering

Here's a simple trick to make sure you're getting this right: always ask yourself, "Would this data be available at prediction time?" This small check will help you avoid data leakage every time.

Think about it like this: if you're building a real-time fraud detection system, you can only use information that would have been available at the moment the transaction happened. It's like trying to predict the weather - you can use yesterday's data, but you can't use tomorrow's!

## Key Takeaways

1. Always check your label timestamps when building features
2. Never use information that wouldn't be available at prediction time
3. Think about what information would actually be available in production
4. Test your features with real-world timing scenarios

Remember: In fraud detection, timing isn't just important - it's everything. Get it wrong, and you might as well be using a crystal ball for predictions!

Want to learn more about how to handle time in your fraud detection system? Check out our other articles on model evaluation metrics and data splitting strategies.




