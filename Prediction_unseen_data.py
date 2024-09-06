# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Bidirectional, LSTM, Input, RepeatVector, Concatenate
from tensorflow.keras.optimizers import Adam

# Step 1: Load and preprocess the unseen data
def load_and_preprocess_unseen_data(file_path, encoder, scaler, removed_features):
    df_unseen = pd.read_csv(file_path, names=[
        'ObjectID', 'Dist_lakes', 'Dist_roads', 'Dist_rivers', 'TWI',
        'Temperature_july', 'Slope', 'Aspect', 'LULC', 'NDVI', 
        'Precipitation', 'Soil_Moisture', 'Elevation'], header=0)

    # Reorder columns for readability
    df_unseen = df_unseen[['ObjectID', 'Dist_lakes', 'Dist_roads', 'Dist_rivers', 'TWI',
                           'Temperature_july', 'Slope', 'Aspect', 'LULC', 'NDVI',
                           'Precipitation', 'Soil_Moisture', 'Elevation']]

    # Replace -9999 with NaN and fill NaN values
    df_unseen.replace(-9999, np.nan, inplace=True)
    df_unseen['LULC'].fillna(3, inplace=True)

    # Map LULC values and convert to category type
    land_use_mapping = {
        1: 'Urban_areas', 2: 'Crop_land', 3: 'Grass_land', 4: 'Tree_covered', 5: 'Shrub_covered',
        6: 'Herbaceous', 7: 'Sparse_vegetation', 8: 'Bare_soil', 9: 'Snow', 10: 'Water_bodies', 11: 'Sparse_vegetation'
    }
    df_unseen['LULC'] = df_unseen['LULC'].replace(land_use_mapping).astype('category')

    # Fill missing values
    distance_columns = ['Dist_lakes', 'Dist_rivers', 'Dist_roads', 'Soil_Moisture', 'NDVI', 'Precipitation', 'Temperature_july']
    environmental_columns = ['Slope', 'Aspect', 'TWI']
    df_unseen[distance_columns] = df_unseen[distance_columns].fillna(df_unseen[distance_columns].mean())
    df_unseen[environmental_columns] = df_unseen[environmental_columns].fillna(df_unseen[environmental_columns].median())

    # Encode categorical and scale numerical features
    LULC_encoded = pd.DataFrame(encoder.transform(df_unseen[['LULC']]), columns=encoder.get_feature_names_out(['LULC']))
    numerical_columns = ['Dist_lakes', 'Dist_roads', 'Dist_rivers', 'TWI', 'Temperature_july', 'Slope', 'Aspect', 
                         'NDVI', 'Precipitation', 'Soil_Moisture', 'Elevation']
    numerical_scaled = pd.DataFrame(scaler.transform(df_unseen[numerical_columns]), columns=numerical_columns)

    # Concatenate the preprocessed features
    df_preprocessed = pd.concat([numerical_scaled.reset_index(drop=True), LULC_encoded.reset_index(drop=True)], axis=1)

    # Drop features removed by Boruta and CART
    df_preprocessed = df_preprocessed.drop(columns=removed_features)

    return df_preprocessed

# Step 2: Define the model creation function
def create_model(model_type, params, input_feature_count):
    if model_type == 'ANN':
        model = Sequential([
            Dense(params['units1'], activation='relu', input_dim=input_feature_count),
            Dropout(params['dropout1']),
            Dense(params['units2'], activation='relu'),
            Dropout(params['dropout2']),
            Dense(params['units3'], activation='relu'),
            Dropout(params['dropout3']),
            Dense(params['units4'], activation='relu'),
            Dropout(params['dropout4']),
            Dense(params['units5'], activation='relu'),
            Dropout(params['dropout5']),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'RNN':
        model = Sequential([
            SimpleRNN(params['units1'], activation='relu', input_shape=(input_feature_count, 1), return_sequences=True),
            Dropout(params['dropout1']),
            SimpleRNN(params['units2'], activation='relu', return_sequences=True),
            Dropout(params['dropout2']),
            SimpleRNN(params['units3'], activation='relu'),
            Dropout(params['dropout3']),
            Dense(params['units4'], activation='relu'),
            Dropout(params['dropout4']),
            Dense(params['units5'], activation='relu'),
            Dropout(params['dropout5']),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'ElmanRNN':
        model = Sequential([
            SimpleRNN(params['units1'], activation='relu', input_shape=(input_feature_count, 1), return_sequences=False),
            Dropout(params['dropout1']),
            Dense(params['units2'], activation='relu'),
            Dropout(params['dropout2']),
            Dense(params['units3'], activation='relu'),
            Dropout(params['dropout3']),
            Dense(params['units4'], activation='relu'),
            Dropout(params['dropout4']),
            Dense(params['units5'], activation='relu'),
            Dropout(params['dropout5']),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'Encoder_Decoder_BiRNN':
        encoder_inputs = Input(shape=(input_feature_count, 1))
        encoder = Bidirectional(LSTM(params['units1'], return_state=True, dropout=params['dropout1']))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        decoder_inputs = RepeatVector(1)(state_h)
        decoder_lstm = LSTM(params['units1'] * 2, return_sequences=False, dropout=params['dropout2'])
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
        outputs = Dense(1, activation='sigmoid')(decoder_outputs)
        model = Model(inputs=encoder_inputs, outputs=outputs)

    return model

# Step 3: Compile and predict using the models
def predict_unseen_data(models, df_preprocessed):
    predictions = {}
    for model_name, model in models.items():
        if model_name != 'ANN':
            prediction = np.round(model.predict(np.expand_dims(df_preprocessed.values, axis=-1)), 3)
        else:
            prediction = np.round(model.predict(df_preprocessed), 3)
        predictions[model_name] = prediction.ravel()
    return predictions

# Step 4: Save predictions to CSV
def save_predictions_to_csv(predictions, df_preprocessed, file_suffix):
    for model_name, prediction in predictions.items():
        df_predictions = pd.DataFrame({
            'FID': range(len(df_preprocessed)),
            'Wildfire_probability': prediction
        })
        df_predictions.to_csv(f'wildfire_predictions_optimized_{model_name}_{file_suffix}.csv', index=False)

# Step 5: Main function to run the prediction workflow
def main_unseen_prediction_workflow(file_path, encoder, scaler, removed_features, optimized_params, file_suffix):
    # Load and preprocess unseen data
    df_preprocessed = load_and_preprocess_unseen_data(file_path, encoder, scaler, removed_features)
    input_feature_count = df_preprocessed.shape[1]

    # Create and compile models
    models = {
        'ANN': create_model('ANN', optimized_params['ANN'], input_feature_count),
        'RNN': create_model('RNN', optimized_params['RNN'], input_feature_count),
        'ElmanRNN': create_model('ElmanRNN', optimized_params['ElmanRNN'], input_feature_count),
        'Encoder_Decoder_BiRNN': create_model('Encoder_Decoder_BiRNN', optimized_params['Encoder_Decoder_BiRNN'], input_feature_count)
    }

    # Compile models
    for model in models.values():
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Predict wildfire probabilities
    predictions = predict_unseen_data(models, df_preprocessed)

    # Save predictions to CSV
    save_predictions_to_csv(predictions, df_preprocessed, file_suffix)

# Example usage for sub-area 1
optimized_params = {
    'ANN': {'units1': 64, 'units2': 32, 'units3': 16, 'units4': 8, 'units5': 8, 'dropout1': 0.3, 'dropout2': 0.3, 'dropout3': 0.3, 'dropout4': 0.3, 'dropout5': 0.3},
    'RNN': {'units1': 128, 'units2': 64, 'units3': 32, 'units4': 16, 'units5': 8, 'dropout1': 0.4, 'dropout2': 0.4, 'dropout3': 0.4, 'dropout4': 0.4, 'dropout5': 0.4},
    'ElmanRNN': {'units1': 64, 'units2': 32, 'units3': 16, 'units4': 8, 'units5': 8, 'dropout1': 0.3, 'dropout2': 0.3, 'dropout3': 0.3, 'dropout4': 0.3, 'dropout5': 0.3},
    'Encoder_Decoder_BiRNN': {'units1': 128, 'dropout1': 0.3, 'dropout2': 0.3}
}

# Assuming `encoder`, `scaler`, and `removed_features` are already fitted and available
main_unseen_prediction_workflow('WF2_Export1.csv', encoder, scaler, removed_features, optimized_params, 'subarea1')
main_unseen_prediction_workflow('WF2_Export2.csv', encoder, scaler, removed_features, optimized_params, 'subarea2')

print("Predictions for sub-area 1 and sub-area 2 have been saved.")
