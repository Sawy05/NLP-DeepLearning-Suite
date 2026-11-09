"""
SMS Spam vs Ham classifier with a neural network (TensorFlow / Keras)
Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
Expected file: spam.csv  (columns: v1 = label, v2 = text, + 3 empty "Unnamed" columns)

Run:  pip install tensorflow pandas scikit-learn matplotlib
      python sms_spam_nn.py
"""

# ----------------------------------------------------------------------
# STEP 0: Imports and reproducibility
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------------------------------------------
# STEP 1: Load the data
# The file is not UTF-8, so latin-1 is required.
# ----------------------------------------------------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the two useful columns and rename them
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
print("Shape:", df.shape)
print(df["label"].value_counts())          # expect ~4825 ham, ~747 spam (imbalanced)

# ----------------------------------------------------------------------
# STEP 2: Clean + encode labels
# ----------------------------------------------------------------------
df = df.drop_duplicates().reset_index(drop=True)   # dataset has ~400 duplicate rows
df["text"] = df["text"].str.lower().str.strip()
df["target"] = (df["label"] == "spam").astype(int)  # ham = 0, spam = 1
print("After dedup:", df.shape)

# ----------------------------------------------------------------------
# STEP 3: Train / validation / test split (stratified keeps the spam ratio)
#         70% train, 15% validation, 15% test
# ----------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    df["text"].values, df["target"].values,
    test_size=0.30, stratify=df["target"].values, random_state=SEED)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ----------------------------------------------------------------------
# STEP 4: Text vectorization (text -> integer sequences)
# The vectorizer is adapted on TRAIN ONLY to avoid data leakage.
# ----------------------------------------------------------------------
VOCAB_SIZE = 5000
MAX_LEN = 50          # SMS are short, 50 tokens covers almost all of them

vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN,
)
vectorizer.adapt(X_train)
print("Vocabulary size:", len(vectorizer.get_vocabulary()))

# ----------------------------------------------------------------------
# STEP 5: Build the neural network
# Embedding -> Bidirectional LSTM -> Dense -> Dropout -> Sigmoid
# ----------------------------------------------------------------------
EMBED_DIM = 32

model = models.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),      # probability of spam
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)
model.summary()

# ----------------------------------------------------------------------
# STEP 6: Handle class imbalance (spam is only ~13% of the data)
# ----------------------------------------------------------------------
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weight)

# ----------------------------------------------------------------------
# STEP 7: Train with early stopping
# ----------------------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=2,
)

# ----------------------------------------------------------------------
# STEP 8: Evaluate on the untouched test set
# ----------------------------------------------------------------------
probs = model.predict(X_test).ravel()
preds = (probs >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(y_test, preds, target_names=["ham", "spam"], digits=4))

cm = confusion_matrix(y_test, preds)
print("Confusion matrix (rows = true, cols = predicted):")
print(cm)

# ----------------------------------------------------------------------
# STEP 9: Plots (learning curves + confusion matrix)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].legend()

ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="val")
ax[1].set_title("Accuracy"); ax[1].legend()

im = ax[2].imshow(cm, cmap="Blues")
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["ham", "spam"])
ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["ham", "spam"])
ax[2].set_xlabel("Predicted"); ax[2].set_ylabel("True")
ax[2].set_title("Confusion matrix")
for i in range(2):
    for j in range(2):
        ax[2].text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# STEP 10: Predict on new messages
# ----------------------------------------------------------------------
def predict_sms(messages):
    p = model.predict(np.array([m.lower() for m in messages]), verbose=0).ravel()
    for m, prob in zip(messages, p):
        print(f"[{'SPAM' if prob >= 0.5 else 'HAM '}] ({prob:.3f})  {m}")

predict_sms([
    "WINNER!! You have been selected for a free prize. Call 09061701461 now to claim",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT! Your mobile number has won a 2000 pound bonus. Text CLAIM to 81010",
])

model.save("sms_spam_model.keras")