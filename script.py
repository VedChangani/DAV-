import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df = pd.read_csv("df.csv")

df["Grade Trend"] = (
    df["Curricular units 2nd sem (grade)"]
    - df["Curricular units 1st sem (grade)"]
)

total_enrolled = (
    df["Curricular units 1st sem (enrolled)"]
    + df["Curricular units 2nd sem (enrolled)"]
)
total_approved = (
    df["Curricular units 1st sem (approved)"]
    + df["Curricular units 2nd sem (approved)"]
)
df["Approval Rate"] = (total_approved / total_enrolled.replace(0, pd.NA)).fillna(0)
df["Average Semester Grade"] = df[
    ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]
].mean(axis=1)
df["Grade Improvement"] = df["Grade Trend"]
df["Failed Units"] = (total_enrolled - total_approved).clip(lower=0)
df["Academic Risk"] = (
    (df["Approval Rate"] < 0.5)
    | (df["Average Semester Grade"] < 10)
    | (df["Failed Units"] >= 3)
)
df["Financial Risk"] = (
    (df["Debtor"] == 1) | (df["Tuition fees up to date"] == 0)
)

df["Age at enrollment"] = pd.to_numeric(df["Age at enrollment"], errors="coerce")
df["Age Category"] = pd.cut(
    df["Age at enrollment"],
    bins=[18, 22, 26, 30, float("inf")],
    labels=["18-21", "22-25", "26-29", ">=30"],
    right=False,
    include_lowest=True,
)

# Interactive flow visualization across qualification, debt status, and outcome.
df["Father's qualification"] = pd.to_numeric(df["Father's qualification"], errors="coerce")
df["Father's qualification (binned)"] = pd.cut(
    df["Father's qualification"],
    bins=[-float("inf"), 3, float("inf")],
    labels=["Basic or unknown", "Secondary or higher"],
    include_lowest=True,
)
df["Debtor status"] = df["Debtor"].map({0: "Not debtor", 1: "Debtor"}).fillna("Unknown")
fig = px.parallel_categories(
    df,
    dimensions=["Father's qualification (binned)", "Debtor status", "Target"],
    title="Student Pathways by Father's Qualification, Debtor Status, and Outcome",
    labels={"Father's qualification (binned)": "Father's qualification"},
)
fig.show()

print(df.head())
print("Shape:", df.shape)
df.info()
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Donut chart of student outcomes with the total displayed in the center.
target_counts = df["Target"].value_counts().reindex(
    ["Dropout", "Graduate"], fill_value=0
)
fig, ax = plt.subplots()
ax.pie(
    target_counts,
    labels=target_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"width": 0.4},
)
ax.text(0, 0, f"{len(df):,}\nstudents", ha="center", va="center")
ax.set_title("Student Outcomes")
ax.axis("equal")
plt.tight_layout()
plt.show()
plt.figure()
sns.histplot(data=df, x="Age at enrollment", bins=20, kde=True)
plt.title("Age at Enrollment")
plt.tight_layout()
plt.show()
course_names = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service ",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management ",
}
course_labels = df["Course"].map(course_names).fillna(df["Course"].astype(str))
plt.figure(figsize=(10, 7))
sns.countplot(y=course_labels, order=course_labels.value_counts().index)
plt.title("Students by Course")
plt.xlabel("Number of students")
plt.ylabel("Course")
plt.tight_layout()
plt.show()

# Compare target proportions for students with fees up to date vs. behind.
fee_status = df["Tuition fees up to date"].map({"Yes": "Up to date", "No": "Behind on fees"})
fee_target_counts = pd.crosstab(fee_status, df["Target"]).reindex(
    index=["Up to date", "Behind on fees"],
    columns=["Dropout", "Graduate"],
    fill_value=0,
)
fee_target_proportions = fee_target_counts.div(
    fee_target_counts.sum(axis=1).replace(0, pd.NA), axis=0
).fillna(0)
fee_target_proportions.plot(kind="bar", stacked=True)
plt.title("Student Outcomes by Tuition Fee Status")
plt.xlabel("Tuition fees")
plt.ylabel("Proportion of students")
plt.xticks(rotation=0)
plt.legend(title="Target")
plt.tight_layout()
plt.show()

plt.figure()
sns.violinplot(data=df, x="Target", y="Curricular units 1st sem (grade)")
plt.title("First Semester Grades by Student Outcome")
plt.tight_layout()
plt.show()

# Grouped counts by gender and target.
gender_target_counts = pd.crosstab(df["Gender"], df["Target"]).reindex(
    columns=["Dropout", "Graduate"], fill_value=0
)
gender_target_counts.plot(kind="bar")
plt.title("Student Outcomes by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of students")
plt.xticks(rotation=0)
plt.legend(title="Target")
plt.tight_layout()
plt.show()