
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

# Read Titanic data
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

# Use training data statistics
age_median = train_df["Age"].median()
fare_median = train_df["Fare"].median()

train_df = train_df.copy()
test_df = test_df.copy()

for data in [train_df, test_df]:
    data["Age"] = data["Age"].fillna(age_median)
    data["Fare"] = data["Fare"].fillna(fare_median)

    # Convert Sex to numeric values
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

# Set up X and Y
x_train = train_df[independent_vars]
x_test = test_df[independent_vars]

y_train = train_df["Survived"]
y_test = test_df["Survived"]


# ------------------------------------
# 6. DECISION TREE
# ------------------------------------

clf_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=1
)

clf_tree.fit(x_train, y_train)

yhat_tree = clf_tree.predict(x_test)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, yhat_tree))
print("Precision:", precision_score(y_test, yhat_tree))
print("F1 Score:", f1_score(y_test, yhat_tree))


# ------------------------------------
# 7. RANDOM FOREST
# ------------------------------------

clf_forest = RandomForestClassifier(
    random_state=0
)

clf_forest.fit(x_train, y_train)

yhat_forest = clf_forest.predict(x_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, yhat_forest))
print("Precision:", precision_score(y_test, yhat_forest))
print("F1 Score:", f1_score(y_test, yhat_forest))


# ------------------------------------
# 8. LOGISTIC REGRESSION
# ------------------------------------

regr = LogisticRegression(
    solver="liblinear",
    random_state=1
)

regr.fit(x_train, y_train)

yhat_log = regr.predict(x_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, yhat_log))
print("Precision:", precision_score(y_test, yhat_log))
print("F1 Score:", f1_score(y_test, yhat_log))


# ------------------------------------
# 9. DESCRIPTIVE VISUALIZATIONS
# ------------------------------------

# Survival counts
df["Survived"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Titanic Survival Counts")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("survival_counts.png")
plt.show()


# Passenger age distribution
df["Age"].dropna().hist(bins=20)

plt.title("Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()


# Survival rate by passenger class
df.groupby("Pclass")["Survived"].mean().plot(
    kind="bar"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()


# ------------------------------------
# 10. MODEL COMPARISON
# ------------------------------------

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

# Comparison visualization
results.set_index("Model").plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Classification Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

