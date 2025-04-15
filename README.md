I built a model that predicts if users will buy something while shopping online.

### 1. Data Loading

I loaded data from CSV and fixed the types:
- Turned text months into numbers (0-11)
- Made returning visitors = 1, others = 0
- Made weekend = 1, weekday = 0
- Made purchase = 1, no purchase = 0

### 2. Model Training

I used a 1-nearest neighbor classifier:
- It finds the most similar user from training data
- It predicts the same action (buy or not buy)

### 3. Evaluation

I measured two things:
- True Positive Rate: How well it finds buyers
- True Negative Rate: How well it finds non-buyers

This works better than just measuring accuracy.

## Key Points

- Simple KNN works well for this problem
- Measuring both rate types gives better results
- No extra features needed - just the basic data

