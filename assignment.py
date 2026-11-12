
# Titanic Survival Classification
# Business Analytics Assignment

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ------------------------------------
# 1. DESCRIPTIVE ANALYTICS
# ------------------------------------

# Read Titanic dataset
df = pd.read_csv("titanic.csv")

print("Dataset shape:")
print(df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Describe three columns
print("\nDescriptive Statistics:")
print(df[["Age", "Fare", "Pclass"]].describe())


# ------------------------------------
# 2. TRAINING AND TESTING DATA
# ------------------------------------

# Split dataset into training and testing
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=1,
    stratify=df["Survived"]
)

print("\nTraining dataset:")
print(train_df.shape)

print("\nTesting dataset:")
print(test_df.shape)


# ------------------------------------
# 3. ANOVA ANALYSIS
# ------------------------------------

# Compare passenger ages between datasets
train_age = train_df["Age"].dropna()
test_age = test_df["Age"].dropna()

f_stat, p_value = f_oneway(train_age, test_age)

print("\nANOVA Results:")
print("F Statistic:", round(f_stat, 4))
print("P Value:", round(p_value, 4))

if p_value < 0.05:
    print("Significant difference in passenger ages")
else:
    print("No significant difference in passenger ages")


# ------------------------------------
# 4. FILL MISSING VALUES
# ------------------------------------

# Use the median age from training data
age_median = train_df["Age"].median()
fare_median = train_df["Fare"].median()

train_df = train_df.copy()
test_df = test_df.copy()

for data in [train_df, test_df]:

    # Fill missing values
    data["Age"] = data["Age"].fillna(age_median)
    data["Fare"] = data["Fare"].fillna(fare_median)

    # Convert Sex into numeric values
    data["Sex"] = data["Sex"].map({
        "male": 0,
        "female": 1
    })


# ------------------------------------
# 5. SELECT INDEPENDENT VARIABLES
# ------------------------------------

independent_vars = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

# Set up X and Y for the algorithms
x_train = train_df[independent_vars]
x_test = test_df[independent_vars]

y_train = train_df["Survived"]
y_test = test_df["Survived"]


# ------------------------------------
# 6. DECISION TREE
# ------------------------------------

# Run Decision Tree algorithm
clf_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=1
)

clf_tree.fit(x_train, y_train)

yhat_tree = clf_tree.predict(x_test)

# Check model results
print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, yhat_tree))
print("Precision:", precision_score(y_test, yhat_tree))
print("F1 Score:", f1_score(y_test, yhat_tree))


# ------------------------------------
# 7. RANDOM FOREST
# ------------------------------------

# Run Random Forest algorithm
clf_forest = RandomForestClassifier(
    random_state=0
)

clf_forest.fit(x_train, y_train)

yhat_forest = clf_forest.predict(x_test)

# Check model results
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, yhat_forest))
print("Precision:", precision_score(y_test, yhat_forest))
print("F1 Score:", f1_score(y_test, yhat_forest))


# ------------------------------------
# 8. LOGISTIC REGRESSION
# ------------------------------------

# Run Logistic Regression algorithm
regr = LogisticRegression(
    solver="liblinear",
    random_state=1
)

regr.fit(x_train, y_train)

yhat_log = regr.predict(x_test)

# Check model results
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, yhat_log))
print("Precision:", precision_score(y_test, yhat_log))
print("F1 Score:", f1_score(y_test, yhat_log))


# ------------------------------------
# 9. DESCRIPTIVE VISUALIZATIONS
# ------------------------------------

# Graph 1: Titanic Survival Counts
plt.figure(figsize=(7, 5))

df["Survived"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Titanic Survival Counts")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("survival_counts.png")
plt.show()
plt.close()


# Graph 2: Passenger Age Distribution
plt.figure(figsize=(7, 5))

df["Age"].dropna().hist(bins=20)

plt.title("Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()
plt.close()


# Graph 3: Survival Rate by Passenger Class
plt.figure(figsize=(7, 5))

df.groupby("Pclass")["Survived"].mean().plot(
    kind="bar"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()
plt.close()


# ------------------------------------
# 10. MODEL COMPARISON
# ------------------------------------

# Create results table
results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression"
    ],

    "Accuracy": [
        accuracy_score(y_test, yhat_tree),
        accuracy_score(y_test, yhat_forest),
        accuracy_score(y_test, yhat_log)
    ],

    "Precision": [
        precision_score(y_test, yhat_tree),
        precision_score(y_test, yhat_forest),
        precision_score(y_test, yhat_log)
    ],

    "F1 Score": [
        f1_score(y_test, yhat_tree),
        f1_score(y_test, yhat_forest),
        f1_score(y_test, yhat_log)
    ]
})

print("\nModel Comparison:")
print(results)

# Save results
results.to_csv("titanic_model_results.csv", index=False)


# Graph 4: Classification Model Comparison
plt.figure(figsize=(10, 6))

results.set_index("Model")[
    ["Accuracy", "Precision", "F1 Score"]
].plot(kind="bar", ax=plt.gca())

plt.title("Classification Model Comparison")
plt.xlabel("Classification Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
plt.close()


# ------------------------------------
# 11. FINAL CONCLUSION
# ------------------------------------

best_model = results.loc[results["F1 Score"].idxmax()]

print("\nFinal Conclusion:")
print("Best classification model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("F1 Score:", round(best_model["F1 Score"], 4))
print("Titanic survival prediction analysis completed.")
