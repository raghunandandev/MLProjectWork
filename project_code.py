import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# --- Load the core CSV files ---
# (Assuming they are in a 'data/' folder)
try:
    student_info = pd.read_csv('data/studentInfo.csv')
    student_reg = pd.read_csv('data/studentRegistration.csv')
    student_vle = pd.read_csv('data/studentVle.csv', engine='python', on_bad_lines='skip')
    student_assess = pd.read_csv('data/studentAssessment.csv')
    assessments = pd.read_csv('data/assessments.csv')
    courses = pd.read_csv('data/courses.csv')
except FileNotFoundError:
    print("Ensure your OULAD CSV files are in a folder named 'data/'.")
    # Handle the error as needed

# --- 1. Define the Target Variable ---
# Create our base DataFrame from student_info
# This table has our primary key and our target
base_df = student_info.copy()

# Map the target variable 'final_result'
# 1 = At-Risk (Dropout/Fail)
# 0 = Not-at-Risk (Pass/Distinction)
base_df['is_at_risk'] = base_df['final_result'].map({
    'Withdrawn': 1,
    'Fail': 1,
    'Pass': 0,
    'Distinction': 0
})

# Drop rows where the student is still 'InProgress'
base_df = base_df.dropna(subset=['is_at_risk'])
base_df['is_at_risk'] = base_df['is_at_risk'].astype(int)

# This is our master table, we will merge all features onto this.
# Key columns: ['id_student', 'code_module', 'code_presentation']
print(f"Base DataFrame shape: {base_df.shape}")
print(base_df['is_at_risk'].value_counts(normalize=True))

# --- 3.1: Static Features (Day 0 Data) ---
# These are features known at the start of the course

# Start with base_df which already contains 'num_of_prev_attempts'
final_df = base_df.copy()

# One-hot encode categorical demographic features
categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
# Use pandas.get_dummies for simplicity
final_df = pd.get_dummies(final_df, columns=categorical_cols, drop_first=True)

# --- 3.2: Dynamic (Aggregated) Features ---
# We simulate an "early prediction" model by only using data from the first 90 days.
SNAPSHOT_DATE = 90

# --- VLE Clicks (student_vle) ---
# Filter VLE data to only include clicks before the snapshot date
vle_early = student_vle[student_vle['date'] < SNAPSHOT_DATE]

# Group by student-course to get aggregated click features
vle_features = vle_early.groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_clicks_early=('sum_click', 'sum'),
    distinct_clicks_early=('id_site', 'nunique'),
    days_active_early=('date', 'nunique')
).reset_index()

# Merge VLE features into our final DataFrame
final_df = pd.merge(final_df, vle_features, on=['id_student', 'code_module', 'code_presentation'], how='left')

# --- Assessment Scores (student_assess) ---
# First, get assessment deadlines and module/presentation info from the 'assessments' table
assess_info = assessments[['id_assessment', 'date', 'code_module', 'code_presentation']]

# Merge info into student_assess
student_assess_with_dates = pd.merge(student_assess, assess_info, on='id_assessment', how='left')

# Convert 'date' and 'score' columns to numeric, coercing errors
student_assess_with_dates['date'] = pd.to_numeric(student_assess_with_dates['date'], errors='coerce')
student_assess_with_dates['score'] = pd.to_numeric(student_assess_with_dates['score'], errors='coerce')

# Drop rows where 'date' is NaN after conversion
student_assess_with_dates.dropna(subset=['date'], inplace=True)

# Filter for assessments submitted AND due before our snapshot
assess_early = student_assess_with_dates[
    (student_assess_with_dates['date_submitted'] < SNAPSHOT_DATE) &
    (student_assess_with_dates['date'] < SNAPSHOT_DATE)
]

# Group by student-course to get aggregated score features
assess_features = assess_early.groupby(['id_student', 'code_module', 'code_presentation']).agg(
    avg_score_early=('score', 'mean'),
    num_assess_submitted_early=('id_assessment', 'count')
).reset_index()

# Merge assessment features into our final DataFrame
final_df = pd.merge(final_df, assess_features, on=['id_student', 'code_module', 'code_presentation'], how='left')

# --- 3.3: Clean Up Final DataFrame ---
# Drop columns we don't need for modeling
final_df = final_df.drop(columns=['final_result', 'id_student'])

# Handle missing values (NaNs)
# For example, a student with 0 clicks or 0 assessments will have NaN after the merge.
# We must fill these with 0.
fill_zero_cols = ['total_clicks_early', 'distinct_clicks_early', 'days_active_early', 'avg_score_early', 'num_assess_submitted_early']
for col in fill_zero_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0)

# 'num_of_prev_attempts' is already in final_df from base_df, fill NaNs with 0.
final_df['num_of_prev_attempts'] = final_df['num_of_prev_attempts'].fillna(0)


# Drop any remaining rows with NaNs (if any)
final_df = final_df.dropna()

print(f"Final features DataFrame shape: {final_df.shape}")

# --- 4.1: Define Features (X) and Target (y) ---
# Our target 'is_at_risk' is already in final_df
# We also need 'code_presentation' for the temporal split
y = final_df['is_at_risk']
X = final_df.drop(columns=['is_at_risk', 'code_module', 'code_presentation'])

# --- 4.2: Temporal Train-Test Split ---
# This is the correct way to validate for this problem.
train_indices = final_df['code_presentation'].str.contains('2013')
test_indices = final_df['code_presentation'].str.contains('2014')

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# --- 4.3: Scale Data ---
# It's good practice, especially for models other than trees.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4.4: Train a Baseline Model ---
# Random Forest is a powerful and robust baseline.
# class_weight='balanced' helps with the imbalanced dataset.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train_scaled, y_train)

# --- 4.5: Make Predictions ---
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the 'at-risk' class (1)

# --- 4.6: Calculate All ML Measures ---
print("--- Baseline Model Evaluation Metrics ---")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Precision
# Of all students we PREDICTED as 'at-risk', what % actually were?
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

# Recall
# Of all students who ACTUALLY were 'at-risk', what % did we CATCH?
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

# F1-Score
# The harmonic mean of Precision and Recall. Great for imbalanced data.
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.4f}")

# AUC-ROC
# Measures the model's ability to distinguish between the two classes.
# 0.5 is random, 1.0 is perfect.
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc:.4f}")

# --- 5.1: Plot Confusion Matrix ---
# Shows us what kind of errors we're making.
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(
    model,
    X_test_scaled,
    y_test,
    display_labels=['Not-at-Risk', 'At-Risk'],
    cmap='Blues',
    ax=ax
)
ax.set_title('Confusion Matrix (Test Set)')
plt.show()

# --- 5.2: Plot ROC Curve ---
# Shows the trade-off between True Positive Rate (Recall) and False Positive Rate.
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    model,
    X_test_scaled,
    y_test,
    ax=ax,
    name='Baseline Random Forest'
)
ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill (AUC = 0.50)') # Random guess line
ax.set_title('ROC Curve (Test Set)')
ax.legend()
plt.show()


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Load Clean Data (from our previous step) ---
# We need this to get our student list, static features, and target variable.
# Let's assume you have 'final_df' loaded. For clarity, let's re-run the key parts.

# 1. Load raw data
student_info = pd.read_csv('data/studentInfo.csv')
student_reg = pd.read_csv('data/studentRegistration.csv')
student_vle = pd.read_csv('data/studentVle.csv', engine='python', on_bad_lines='skip')
student_assess = pd.read_csv('data/studentAssessment.csv')
assessments = pd.read_csv('data/assessments.csv') # Also load assessments here

# 2. Define target
base_df = student_info.copy()
base_df['is_at_risk'] = base_df['final_result'].map({'Withdrawn': 1, 'Fail': 1, 'Pass': 0, 'Distinction': 0})
base_df = base_df.dropna(subset=['is_at_risk'])
base_df['is_at_risk'] = base_df['is_at_risk'].astype(int)

# 3. Get static features
# Get static features, including 'num_of_prev_attempts' from base_df
static_features_df = base_df.copy()
categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
static_features_df = pd.get_dummies(static_features_df, columns=categorical_cols, drop_first=True)
static_features_df['num_of_prev_attempts'] = static_features_df['num_of_prev_attempts'].fillna(0)
static_features_df = static_features_df.drop(columns=['final_result'])

# --- 1.1: Create the Temporal Feature Set (X_temporal) ---
SNAPSHOT_DATE = 90 # We'll look at the first 90 days

# --- VLE Clicks (student_vle) ---
vle_early = student_vle[student_vle['date'] <= SNAPSHOT_DATE]
# Group by student AND day
vle_daily = vle_early.groupby(['id_student', 'code_module', 'code_presentation', 'date']).agg(
    daily_clicks=('sum_click', 'sum')
).reset_index()

# --- Assessment Scores (student_assess) ---
# Merge assessment info (including module/presentation) into student_assess
student_assess_with_info = pd.merge(student_assess, assessments[['id_assessment', 'code_module', 'code_presentation', 'date']], on='id_assessment', how='left')

# We'll create a feature for "score submitted on this day"
# Convert 'date_submitted' and 'date' (assessment deadline) to numeric, coercing errors
student_assess_with_info['date_submitted'] = pd.to_numeric(student_assess_with_info['date_submitted'], errors='coerce')
student_assess_with_info['date'] = pd.to_numeric(student_assess_with_info['date'], errors='coerce')
student_assess_with_info['score'] = pd.to_numeric(student_assess_with_info['score'], errors='coerce')


# Filter for assessments submitted BEFORE our snapshot date, and where deadline is available
assess_early = student_assess_with_info[
    (student_assess_with_info['date_submitted'] <= SNAPSHOT_DATE) &
    (student_assess_with_info['date'].notna()) # Ensure assessment deadline is available
]

# Group by student-course-submitted_date to get average score on that day
assess_daily = assess_early.groupby(['id_student', 'code_module', 'code_presentation', 'date_submitted']).agg(
    daily_score_avg=('score', 'mean')
).reset_index().rename(columns={'date_submitted': 'date'})


# --- Combine Temporal Features ---
# Create a full (student x day) grid for 0 to 90 days
student_keys = base_df[['id_student', 'code_module', 'code_presentation']].drop_duplicates()
day_range = pd.DataFrame({'date': range(SNAPSHOT_DATE + 1)})
# Cross-join students with all possible days
temporal_grid = student_keys.merge(day_range, how='cross')

# Merge daily clicks and scores onto this grid
temporal_data = pd.merge(temporal_grid, vle_daily, on=['id_student', 'code_module', 'code_presentation', 'date'], how='left')
temporal_data = pd.merge(temporal_data, assess_daily, on=['id_student', 'code_module', 'code_presentation', 'date'], how='left')

# Fill NaNs with 0 (e.g., 0 clicks or 0 assessments on that day)
temporal_data = temporal_data.fillna(0)

# --- 1.2: Create the Static Feature Set (X_static) ---
# We need to make sure our static_features_df and temporal_data align perfectly.
# We'll merge the static features onto the temporal data to get our final, aligned dataset.
final_data = pd.merge(temporal_data, static_features_df, on=['id_student', 'code_module', 'code_presentation'], how='inner')

# --- 1.3: Create Train/Test Split (Temporal Validation) ---
# Same as before. Train on 2013, test on 2014.
train_mask = final_data['code_presentation'].str.contains('2013')
test_mask = final_data['code_presentation'].str.contains('2014')

train_df = final_data[train_mask]
test_df = final_data[test_mask]

# --- 1.4: Reshape Data for the Model ---
# This is the final, crucial step.

# List our feature groups
temporal_feature_cols = ['daily_clicks', 'daily_score_avg']
# All columns that are NOT keys, targets, or temporal
static_feature_cols = [col for col in static_features_df.columns if col not in
                       ['id_student', 'code_module', 'code_presentation', 'is_at_risk']]

# Function to reshape data for one student at a time
def reshape_to_3d(df, temporal_cols):
    # (num_students, timesteps, features)
    # The 'date' column ensures the data is sorted correctly
    return df.sort_values(by='date')[temporal_cols].values

# Function to get static data
def get_static_data(df, static_cols):
    # Get static data (it's the same for all 90 days, so just take the first row)
    return df.drop_duplicates(subset=['id_student', 'code_module', 'code_presentation'])[static_cols].values

# Group by student
train_groups = train_df.groupby(['id_student', 'code_module', 'code_presentation'])
test_groups = test_df.groupby(['id_student', 'code_module', 'code_presentation'])

# Create 3D temporal arrays
X_train_temporal = np.array([reshape_to_3d(group, temporal_feature_cols) for _, group in train_groups])
X_test_temporal = np.array([reshape_to_3d(group, temporal_feature_cols) for _, group in test_groups])

# Create 2D static arrays
X_train_static = np.array([get_static_data(group, static_feature_cols)[0] for _, group in train_groups])
X_test_static = np.array([get_static_data(group, static_feature_cols)[0] for _, group in test_groups])

# Create target arrays (y)
y_train = np.array([group['is_at_risk'].iloc[0] for _, group in train_groups])
y_test = np.array([group['is_at_risk'].iloc[0] for _, group in test_groups])

# --- 1.5: Scale the Data ---
# Temporal data and static data must be scaled separately

# Scale temporal data (fit on train, transform train/test)
# We must reshape to 2D to fit the scaler, then reshape back to 3D
scaler_temporal = StandardScaler()
X_train_temporal_2d = X_train_temporal.reshape(-1, len(temporal_feature_cols))
X_train_temporal_scaled = scaler_temporal.fit_transform(X_train_temporal_2d).reshape(X_train_temporal.shape)
X_test_temporal_2d = X_test_temporal.reshape(-1, len(temporal_feature_cols))
X_test_temporal_scaled = scaler_temporal.transform(X_test_temporal_2d).reshape(X_test_temporal.shape)

# Scale static data
scaler_static = StandardScaler()
X_train_static_scaled = scaler_static.fit_transform(X_train_static)
X_test_static_scaled = scaler_static.transform(X_test_static)

print(f"X_train_temporal shape: {X_train_temporal_scaled.shape}")
print(f"X_train_static shape: {X_train_static_scaled.shape}")
print(f"y_train shape: {y_train.shape}")

# --- 2.1: Define Model Inputs ---
# Temporal branch
temporal_input = Input(shape=(SNAPSHOT_DATE + 1, len(temporal_feature_cols)), name='temporal_input')
# Static branch
static_input = Input(shape=(len(static_feature_cols),), name='static_input')

# --- 2.2: Define Temporal Branch (GRU Brain) ---
# GRU layer reads the sequence. return_sequences=False means it just outputs the final hidden state.
gru_out = GRU(units=32, name='gru_layer')(temporal_input)
gru_out = Dropout(0.3, name='gru_dropout')(gru_out) # Dropout for regularization

# --- 2.3: Define Static Branch (MLP Brain) ---
dense_out = Dense(units=32, activation='relu', name='static_dense_1')(static_input)
dense_out = Dropout(0.3, name='static_dropout')(dense_out)

# --- 2.4: Concatenate Branches ---
# This is where the two "brains" merge their findings
combined = concatenate([gru_out, dense_out], name='concatenate')

# --- 2.5: Final Classifier Head ---
# A final set of layers to make the prediction
output = Dense(units=16, activation='relu', name='final_dense')(combined)
output = Dense(units=1, activation='sigmoid', name='output_layer')(output) # Sigmoid for binary 0/1 probability

# --- 2.6: Build and Compile the Model ---
model = Model(inputs=[temporal_input, static_input], outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Standard loss for binary classification
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# --- 3.1: Handle Class Imbalance ---
# Deep learning models also suffer from imbalance. We'll use class weights.
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = {0: weights[0], 1: weights[1]}

print(f"Class weights: {class_weight}")

# --- 3.2: Train the Model ---
# We feed the inputs as a dictionary
history = model.fit(
    {'temporal_input': X_train_temporal_scaled, 'static_input': X_train_static_scaled},
    y_train,
    validation_data=(
        {'temporal_input': X_test_temporal_scaled, 'static_input': X_test_static_scaled},
        y_test
    ),
    epochs=10, # Start with 10, you may need more (or fewer with early stopping)
    batch_size=64,
    class_weight=class_weight
)

# --- 3.3: Plot Training History ---
# This plot is ESSENTIAL for your paper. It shows the model is learning.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.legend()
plt.show()

# --- 3.4: Calculate Final Metrics on Test Set ---
# Get probabilities
y_pred_proba_dl = model.predict(
    {'temporal_input': X_test_temporal_scaled, 'static_input': X_test_static_scaled}
).ravel() # .ravel() flattens (n, 1) to (n,)

# Convert probabilities to binary predictions (0 or 1)
y_pred_dl = (y_pred_proba_dl > 0.5).astype(int)

print("\n--- Deep Learning Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dl):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dl):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dl):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dl):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba_dl):.4f}")

pip install shap

import shap
import pandas as pd
import matplotlib.pyplot as plt

# --- We assume you have these variables from our previous steps ---
# model: Your trained RandomForestClassifier
# X_train: Your training features (unscaled, for easier plot labels)
# X_test: Your testing features (unscaled)
# X_train_scaled: Your scaled training features
# X_test_scaled: Your scaled testing features

# (If you don't have them, re-run the relevant code from the baseline step)
# Let's ensure X_test is a pandas DataFrame with correct column names
# If X_test_scaled is a numpy array, we need the feature names from X
feature_names = X.columns.tolist()
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# --- 2.1: Create the Explainer ---
# We pass our trained model to the explainer
explainer = shap.TreeExplainer(model)

# --- 2.2: Calculate SHAP Values ---
# We calculate SHAP values for all students in our test set.
# This can take a moment.
shap_values = explainer.shap_values(X_test_scaled)

# shap_values will be a list [class_0_values, class_1_values]
# We only care about the values for the "at-risk" class (Class 1)
shap_values_at_risk = shap_values[1]

print(f"SHAP values shape: {shap_values_at_risk.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# --- 3.1: Global Feature Importance (Summary Bar Plot) ---
# This plot shows the average impact of each feature on the model's output.
print("--- Plotting Global Feature Importance ---")

shap.summary_plot(
    shap_values_at_risk, 
    X_test_df, 
    plot_type="bar",
    title="Global Feature Importance (Risk of Dropout)"
)

# We need to find the indices of these students
# (y_test is our true labels, y_pred is our model's predictions)

# Find a True Positive (Correctly flagged as 'at-risk')
# True label == 1 AND Predicted label == 1
true_positives = np.where((y_test == 1) & (y_pred == 1))[0]
if len(true_positives) > 0:
    tp_index = true_positives[0]
else:
    print("No True Positives found to plot.")
    tp_index = 0 # Fallback

# Find a True Negative (Correctly flagged as 'not-at-risk')
# True label == 0 AND Predicted label == 0
true_negatives = np.where((y_test == 0) & (y_pred == 0))[0]
if len(true_negatives) > 0:
    tn_index = true_negatives[0]
else:
    print("No True Negatives found to plot.")
    tn_index = 0 # Fallback

# --- 4.1: Plot for the AT-RISK Student (True Positive) ---
# shap.initjs() # Run this in a notebook to enable interactive plots

print(f"\n--- Plotting Local Explanation for AT-RISK student (index {tp_index}) ---")
# This force plot shows the "push" and "pull" of each feature
shap.force_plot(
    explainer.expected_value[1], # The baseline prediction (average)
    shap_values_at_risk[tp_index, :], # The SHAP values for this one student
    X_test_df.iloc[tp_index, :], # The feature values for this student
    matplotlib=True, # Use matplotlib for non-notebook environments
    show=True
)

# --- 4.2: Plot for the NOT-AT-RISK Student (True Negative) ---
print(f"\n--- Plotting Local Explanation for NOT-AT-RISK student (index {tn_index}) ---")
shap.force_plot(
    explainer.expected_value[1], 

  
    shap_values_at_risk[tn_index, :],
    X_test_df.iloc[tn_index, :],
    matplotlib=True,
    show=True
)

# --- 5.1: Create the DeepExplainer ---
# It needs a "background" dataset to get expected values.
# A subset of the training data is recommended.
# Let's use 100 random students from the training set.

background_indices = np.random.choice(X_train_temporal_scaled.shape[0], 100, replace=False)
X_train_temporal_bg = X_train_temporal_scaled[background_indices]
X_train_static_bg = X_train_static_scaled[background_indices]

# We pass the model and the background data (as a list)
dl_explainer = shap.DeepExplainer(model, [X_train_temporal_bg, X_train_static_bg])

# --- 5.2: Calculate SHAP Values ---
# Let's get SHAP values for 500 test students (it's slow)
X_test_temporal_sample = X_test_temporal_scaled[:500]
X_test_static_sample = X_test_static_scaled[:500]
y_test_sample = y_test[:500]

# shap_values_dl will be a list of 2 arrays:
# [0] = SHAP values for temporal input (shape: 500, 91, 2)
# [1] = SHAP values for static input (shape: 500, 30)
shap_values_dl = dl_explainer.shap_values(
    [X_test_temporal_sample, X_test_static_sample]
)

# --- 5.3: Plot Explanations ---
# We can plot the static features just like before
shap_static_values = shap_values_dl[1][0] # Index 0 is for class, 1 is for static input
static_feature_names = X.columns.tolist() # Get all feature names
X_test_static_df = pd.DataFrame(X_test_static_sample, columns=static_feature_names)

shap.summary_plot(
    shap_static_values, 
    X_test_static_df, 
    plot_type="bar",
    title="Global Feature Importance (Static Features in DL Model)"
)

# --- 5.4: Plot Temporal Explanations (The Real Novelty) ---
# This is the most exciting plot.
# We'll average the SHAP values over the 500 students to see
# which days and features are most important.
shap_temporal_values = shap_values_dl[0][0] # (500, 91, 2)

# Average importance of 'daily_clicks' over all students
daily_clicks_shap = shap_temporal_values[:, :, 0] # shape (500, 91)
avg_clicks_shap_per_day = np.mean(np.abs(daily_clicks_shap), axis=0) # shape (91,)

# Average importance of 'daily_score_avg' over all students
daily_score_shap = shap_temporal_values[:, :, 1] # shape (500, 91)
avg_score_shap_per_day = np.mean(np.abs(daily_score_shap), axis=0) # shape (91,)

# Plot this
plt.figure(figsize=(12, 6))
plt.plot(avg_clicks_shap_per_day, label='Importance of Daily Clicks')
plt.plot(avg_score_shap_per_day, label='Importance of Daily Scores')
plt.xlabel('Day in Course (0 to 90)')
plt.ylabel('Average SHAP Value (Impact on Prediction)')
plt.title('Importance of Features Over Time')
plt.legend()
plt.show()

