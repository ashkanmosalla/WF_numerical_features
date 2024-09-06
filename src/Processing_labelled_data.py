# Install required packages
# pip install shap boruta

# Import necessary libraries
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from boruta import BorutaPy
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Bidirectional, LSTM, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Function for data loading and preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, names=[
        'Index', 'Wildfire_status', 'X', 'Y', 'Dist_lakes', 'Dist_roads', 'Dist_rivers', 'TWI',
        'Temperature_july', 'Slope', 'Aspect', 'LULC', 'NDVI', 'Precipitation', 'Soil_Moisture', 'Elevation'], header=0)
    
    df = df[df['LULC'] != 0]
    replace_values_LULC = {
        1: 'Urban_areas', 2: 'Crop_land', 3: 'Grass_land', 4: 'Tree_covered', 5: 'Shrub_covered',
        6: 'Herbaceous', 7: 'Sparse_vegetation', 8: 'Bare_soil', 9: 'Snow', 10: 'Water_bodies', 11: 'Sparse_vegetation'}
    df['LULC'] = df['LULC'].replace(replace_values_LULC)
    df = df.astype({'LULC': 'category'})
    df = df.drop(columns=['Index', 'X', 'Y'], axis=1)

    return df

# Function for correlation heatmap visualization
def plot_correlation_heatmap(df):
    numerical_df = df.drop(columns=['LULC', 'Wildfire_status'])
    correlation_matrix = numerical_df.corr(method='pearson')
    plt.figure(figsize=(10, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.75})
    plt.title('Pearson Correlation Coefficients')
    plt.tight_layout()
    plt.show()

# Function to split data into training, validation, and test sets
def split_data(df):
    X = df.drop(columns=['Wildfire_status'])
    y = df['Wildfire_status']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function for feature encoding and scaling
def preprocess_features(X_train, X_val, X_test):
    categorical_columns = ['LULC']
    numerical_columns = ['Dist_lakes', 'Dist_roads', 'Dist_rivers', 'TWI', 'Temperature_july', 'Slope', 'Aspect', 'NDVI', 'Precipitation', 'Soil_Moisture', 'Elevation']
    
    encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()

    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numerical_columns]), columns=numerical_columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val[numerical_columns]), columns=numerical_columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_columns]), columns=numerical_columns)

    # Concatenate encoded and scaled features
    X_train_final = pd.concat([X_train_scaled.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
    X_val_final = pd.concat([X_val_scaled.reset_index(drop=True), X_val_encoded.reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test_scaled.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
    
    return X_train_final, X_val_final, X_test_final

# Function for feature selection using Boruta and CART
def feature_selection(X_train_final, y_train):
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    rf.fit(X_train_final, y_train)

    boruta = BorutaPy(estimator=rf, n_estimators='auto', max_iter=100, random_state=42)
    boruta.fit(X_train_final.values, y_train.values)

    boruta_feature_stats = pd.DataFrame({'Variables': X_train_final.columns, 'BorutaDecision': ['Confirmed' if decision else 'Rejected' for decision in boruta.support_]})
    cart_model = DecisionTreeClassifier(random_state=42)
    cart_model.fit(X_train_final, y_train)
    cart_feature_stats = pd.DataFrame({'Variables': X_train_final.columns, 'RelativeImportance': cart_model.feature_importances_})

    combined_feature_stats = boruta_feature_stats.merge(cart_feature_stats, on='Variables')
    removed_features = pd.concat([combined_feature_stats[combined_feature_stats['BorutaDecision'] == 'Rejected']['Variables'], combined_feature_stats[combined_feature_stats['RelativeImportance'] < 0.001]['Variables']]).unique()

    return removed_features

# Function to build and train models (DNN, RNN, Elman RNN, Encoder-Decoder BiRNN)
def build_and_train_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):
    if model_type == 'DNN':
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

    elif model_type == 'RNN':
        model = Sequential()
        model.add(SimpleRNN(32, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(SimpleRNN(16, activation='relu', return_sequences=True))
        model.add(Dropout(0.5))
        model.add(SimpleRNN(8, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

    elif model_type == 'ElmanRNN':
        model = Sequential()
        model.add(SimpleRNN(32, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

    elif model_type == 'Encoder_Decoder_BiRNN':
        units = 128
        encoder_inputs = Input(shape=(X_train.shape[1], 1))
        encoder = Bidirectional(LSTM(units, return_state=True, dropout=0.5))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        decoder_inputs = tf.keras.layers.RepeatVector(1)(state_h)
        decoder_lstm = LSTM(units * 2, return_sequences=False, dropout=0.5)
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(1, activation='sigmoid')
        outputs = decoder_dense(decoder_outputs)

        model = Model(inputs=encoder_inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    if model_type != 'DNN':
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0)

    y_train_proba = model.predict(X_train).ravel()
    y_test_proba = model.predict(X_test).ravel()
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    y_train_pred = (y_train_proba > 0.5).astype(int)
    y_test_pred = (y_test_proba > 0.5).astype(int)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    specificity = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1])
    f1 = f1_score(y_test, y_test_pred)

    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Sensitivity': recall, 'Specificity': specificity, 'F1 Score': f1}
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

    return fpr_train, tpr_train, train_auc, fpr_test, tpr_test, test_auc, cm_train, cm_test, metrics

# Prepare lists to store the results
fprs_train, tprs_train, aucs_train = [], [], []
fprs_test, tprs_test, aucs_test = [], [], []
cms_train, cms_test = [], []
metrics_data = {}

# Train models and gather ROC data and metrics
models = ['DNN', 'RNN', 'ElmanRNN', 'Encoder_Decoder_BiRNN']
for model_type in models:
    fpr_train, tpr_train, train_auc, fpr_test, tpr_test, test_auc, cm_train, cm_test, metrics = build_and_train_model(
        model_type, X_train_selected, y_train, X_val_selected, y_val, X_test_selected, y_test)

    fprs_train.append((fpr_train, tpr_train, train_auc))
    fprs_test.append((fpr_test, tpr_test, test_auc))
    cms_train.append(cm_train)
    cms_test.append(cm_test)
    metrics_data[model_type] = metrics

# Convert the metrics dictionary to a DataFrame and save to CSV
metrics_df = pd.DataFrame(metrics_data).transpose()
metrics_df.to_csv('model_metrics.csv', index=True)
print(metrics_df)

# Function for plotting ROC curves
def plot_roc(fprs_train, tprs_train, aucs_train, fprs_test, tprs_test, aucs_test, models):
    plt.figure(figsize=(8, 6))
    for i, (fpr, tpr, auc) in enumerate(fprs_train):
        plt.plot(fpr, tpr, lw=2, label=f'{models[i]} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Train')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(8, 6))
    for i, (fpr, tpr, auc) in enumerate(fprs_test):
        plt.plot(fpr, tpr, lw=2, label=f'{models[i]} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test')
    plt.legend(loc="lower right")
    plt.show()

# Function for displaying confusion matrices
def display_confusion_matrices(cms_train, cms_test, models):
    for i, model_type in enumerate(models):
        disp_train = ConfusionMatrixDisplay(confusion_matrix=cms_train[i])
        disp_train.plot()
        plt.title(f"Confusion Matrix - {model_type} - Training Data")
        plt.show()

        disp_test = ConfusionMatrixDisplay(confusion_matrix=cms_test[i])
        disp_test.plot()
        plt.title(f"Confusion Matrix - {model_type} - Testing Data")
        plt.show()

# Main execution flow
if __name__ == "__main__":
    df = load_and_preprocess_data('WF2_numeric.csv')
    plot_correlation_heatmap(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_final, X_val_final, X_test_final = preprocess_features(X_train, X_val, X_test)
    
    removed_features = feature_selection(X_train_final, y_train)
    X_train_selected = X_train_final.drop(columns=removed_features)
    X_val_selected = X_val_final.drop(columns=removed_features)
    X_test_selected = X_test_final.drop(columns=removed_features)

    models = ['DNN', 'RNN', 'ElmanRNN', 'Encoder_Decoder_BiRNN']
    fprs_train, tprs_train, aucs_train, fprs_test, tprs_test, aucs_test = [], [], [], [], [], []
    cms_train, cms_test = [], []

    for model_type in models:
        fpr_train, tpr_train, train_auc, fpr_test, tpr_test, test_auc, cm_train, cm_test, metrics = build_and_train_model(
            model_type, X_train_selected, y_train, X_val_selected, y_val, X_test_selected, y_test)
        fprs_train.append((fpr_train, tpr_train, train_auc))
        fprs_test.append((fpr_test, tpr_test, test_auc))
        cms_train.append(cm_train)
        cms_test.append(cm_test)

    plot_roc(fprs_train, tprs_train, aucs_train, fprs_test, tprs_test, aucs_test, models)
    display_confusion_matrices(cms_train, cms_test, models)
