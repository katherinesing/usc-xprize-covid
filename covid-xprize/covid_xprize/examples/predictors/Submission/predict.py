# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, PowerTransformer, power_transform, scale, StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OXFORD_DATA_FILE = os.path.join(ROOT_DIR, 'data', "OxCGRT_latest.csv")
SOCIAL_EXPLORER_DATA_FILE = os.path.join(ROOT_DIR, 'data', "social_explorer_data.csv")
LODES_DATA_FILE = os.path.join(ROOT_DIR, 'data', "lodes_us_states_rac_S000_JT00_2018.csv")
CASES_PIPE_FILE = os.path.join(ROOT_DIR, 'data', 'cases_pipe.pkl')

# General model params
GENERAL_MODEL_FILE = os.path.join(ROOT_DIR, "models", "General_Model.pkl")
general_nb_lookback_days = 30
general_rolling = True
general_transform = False

# US model params
US_MODEL_FILE = os.path.join(ROOT_DIR, "models", "US_model.pkl")
FS_INDEX_FILE = os.path.join(ROOT_DIR, 'data', 'lasso_fs_1102_2.npy')
us_nb_lookback_days = 30
us_static = True
us_rolling = True
us_transform = False
us_fs = True

# ARIMA params
MODEL_FOLDER = os.path.join(ROOT_DIR, "models")

ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COL = ['NewCases', '7DMA']
NPI_COLS_CATEGORICAL = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']
NPI_COLS = ['C1_School closing_0.0',
            'C1_School closing_1.0',
            'C1_School closing_2.0',
            'C1_School closing_3.0',
            'C2_Workplace closing_0.0',
            'C2_Workplace closing_1.0',
            'C2_Workplace closing_2.0',
            'C2_Workplace closing_3.0',
            'C3_Cancel public events_0.0',
            'C3_Cancel public events_1.0',
            'C3_Cancel public events_2.0',
            'C4_Restrictions on gatherings_0.0',
            'C4_Restrictions on gatherings_1.0',
            'C4_Restrictions on gatherings_2.0',
            'C4_Restrictions on gatherings_3.0',
            'C4_Restrictions on gatherings_4.0',
            'C5_Close public transport_0.0',
            'C5_Close public transport_1.0',
            'C5_Close public transport_2.0',
            'C6_Stay at home requirements_0.0',
            'C6_Stay at home requirements_1.0',
            'C6_Stay at home requirements_2.0',
            'C6_Stay at home requirements_3.0',
            'C7_Restrictions on internal movement_0.0',
            'C7_Restrictions on internal movement_1.0',
            'C7_Restrictions on internal movement_2.0',
            'C8_International travel controls_0.0',
            'C8_International travel controls_1.0',
            'C8_International travel controls_2.0',
            'C8_International travel controls_3.0',
            'C8_International travel controls_4.0',
            'H1_Public information campaigns_0.0',
            'H1_Public information campaigns_1.0',
            'H1_Public information campaigns_2.0',
            'H2_Testing policy_0.0',
            'H2_Testing policy_1.0',
            'H2_Testing policy_2.0',
            'H2_Testing policy_3.0',
            'H3_Contact tracing_0.0',
            'H3_Contact tracing_1.0',
            'H3_Contact tracing_2.0',
            'H6_Facial Coverings_0.0',
            'H6_Facial Coverings_1.0',
            'H6_Facial Coverings_2.0',
            'H6_Facial Coverings_3.0',
            'H6_Facial Coverings_4.0']
ARIMA_REGIONS = [
  'Andorra',
  'Aruba',
  'Bahamas',
  'Barbados',
  'Belize',
  'Bermuda',
  'Bhutan',
  'Brunei',
  'Cape Verde',
  'Central African Republic',
  'Comoros',
  'Congo',
  'Cyprus',
  'Djibouti',
  'Dominica',
  'Faeroe Islands',
  'Fiji',
  'Gambia',
  'Greenland',
  'Guam',
  'Iceland',
  'Jamaica',
  'Laos',
  'Macao',
  'Mauritius',
  'Monaco',
  'Mongolia',
  'New Zealand',
  'Norway',
  'Qatar',
  'San Marino',
  'Seychelles',
  'Solomon Islands',
  'South Sudan',
  'Tanzania',
  'Timor-Leste',
  'Togo',
  'Trinidad and Tobago',
  'Vanuatu',
]
# For testing, restrict training data to that before a hypothetical predictor submission date
HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-12-20")

def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path) -> None:
  """
  Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
  plans, between start_date and end_date, included.
  :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
  :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
  :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
  and end_date, for the countries and regions for which a prediction is needed
  :param output_file_path: path to file to save the predictions to
  :return: Nothing. Saves the generated predictions to an output_file_path CSV file
  with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
  """
  # !!! YOUR CODE HERE !!!
  preds_df = predict_df(start_date, end_date, path_to_ips_file, verbose=False)
  # Create the output path
  os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
  # Save to a csv file
  preds_df.to_csv(output_file_path, index=False)
  print(f"Saved predictions to {output_file_path}")

def transform_data(data):
  # pipe = make_pipeline(StandardScaler(with_std=False), PowerTransformer(standardize=True))
  pipe = make_pipeline(StandardScaler(with_std=True))
  if len(data.shape) == 1:
    transformed_data = pipe.fit_transform(data.reshape(-1,1))
    return (transformed_data, pipe)
  transformed_data = pipe.fit_transform(data)
  return (transformed_data, pipe)
def apply_transform(pipe, data):
  if len(data.shape) == 1:
    return pipe.transform(data.reshape(-1,1))
  return pipe.transform(data)
def inverse_transform(pipe, data):
  if len(data.shape) == 1:
    return pipe.inverse_transform(data.reshape(-1,1))
  return pipe.inverse_transform(data)

def load_census_data():
  census_data = pd.read_csv(SOCIAL_EXPLORER_DATA_FILE, skiprows=[1])
  census_data.drop(columns=['FIPS'], inplace=True)
  cols = census_data.columns.values
  for feat in cols:
    if 'Employed' in feat or 'Unemployed' in feat:
      census_data.drop(columns=[feat], inplace=True)
  # Standardize Census data
  census_columns = census_data.columns[1:]
  census_regions = census_data['Qualifying Name']
  census_data, census_pipe = transform_data(census_data.iloc[:, 1:].values)
  census_data = pd.DataFrame(data=census_data,columns=census_columns)
  census_data.insert(0, 'Qualifying Name', census_regions)
  census_cols = list(census_data.columns[1:].values)
  return census_data, census_cols

def load_lodes_data():
  lodes_data = pd.read_csv(LODES_DATA_FILE, skiprows=[1])
  # Standardize lodes data
  lodes_columns = lodes_data.columns[1:]
  lodes_regions = lodes_data['Qualifying Name']
  lodes_data, lodes_pipe = transform_data(lodes_data.iloc[:, 1:].values)
  lodes_data = pd.DataFrame(data=lodes_data,columns=lodes_columns)
  lodes_data.insert(0, 'Qualifying Name', lodes_regions)
  lodes_cols = list(lodes_data.columns[1:].values)
  return lodes_data, lodes_cols

def runUSModel(us_model, hist_cases_gdf, hist_ips_gdf, ips_gdf, census_data, lodes_data, census_cols, lodes_cols, us_fs_index, current_date, start_date, end_date):
  # Determine region
  countryName = ips_gdf['CountryName'].iloc[0]
  regionName = ips_gdf['RegionName'].iloc[0]

  # Get case data / NPIs for this region
  past_cases = np.array(hist_cases_gdf[CASES_COL])
  past_npis = np.array(hist_ips_gdf[NPI_COLS])
  future_npis = np.array(ips_gdf[NPI_COLS])

  # Get static data for this region
  census_gdf = census_data[census_data['Qualifying Name'] == regionName]
  lodes_gdf = lodes_data[lodes_data['Qualifying Name'] == regionName]
  all_census_data = np.array(census_gdf[census_cols].iloc[0])
  all_lodes_data = np.array(lodes_gdf[lodes_cols].iloc[0])

  # Make prediction for each day
  geo_preds = []
  days_ahead = 0
  while current_date <= end_date:
    # Prepare data
    X_cases = past_cases[-us_nb_lookback_days:, 1] if us_rolling else past_cases[-us_nb_lookback_days:, 0]
    X_npis = past_npis[-us_nb_lookback_days:]
    X_npis_or = np.bitwise_or.reduce(X_npis, axis=0)
    X_census = all_census_data
    X_lodes = all_lodes_data
    X = np.concatenate([X_cases.flatten(), X_npis_or.flatten(), X_census.flatten(), X_lodes.flatten()])
    if us_fs:
      X = X[np.array(us_fs_index)]
    
    # Make the prediction (reshape so that sklearn is happy)
    pred = us_model.predict(X.reshape(1, -1))[0]
    new_row = [pred]
    # if us_rolling:
    #   # Convert rolling avg to actual cases
    #   pred = 7*(pred-past_cases[-1, 1])+past_cases[-7, 0]
    pred = max(0, pred)  # Do not allow predicting negative cases
    new_row = [pred] + new_row
    # Add if it's a requested date
    if current_date >= start_date:
      geo_preds.append(pred)

    # Append the prediction and npi's for next day
    # in order to rollout predictions for further days.
    past_cases = np.vstack([past_cases, new_row])
    past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + 1], axis=0)

    # Move to next day
    current_date = current_date + np.timedelta64(1, 'D')
    days_ahead += 1
  return geo_preds

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def runARIMAModel(country, hist_cases_gdf, current_date, start_date, end_date):
  # step 1: pre-process hist_cases_gdf into history array for SARIMAX model
  timesteps = hist_cases_gdf['NewCases'].values.tolist()
  timesteps = moving_average(timesteps, 7)
  data = pd.DataFrame(data=timesteps, columns=['new_cases']).squeeze()
  X = data.values

  best_cfg_path = os.path.join(MODEL_FOLDER, f'{country}.pkl')
  with open(best_cfg_path, 'rb') as f:
    best_cfg = pickle.load(f)
  geo_preds = []
  # Start predicting from start_date, unless there's a gap since last known date
  days_ahead = 0
  dates = []
  while current_date <= end_date:
    model = SARIMAX(X, order=best_cfg, initialization='approximate_diffuse')
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    pred = output[0]
    pred = max(0, pred)  # Do not allow predicting negative cases
    X = np.append(X, pred)
    
    # Add if it's a requested date
    if current_date >= start_date:
      geo_preds.append(pred)
      dates.append(current_date)

    # Move to next day
    current_date = current_date + np.timedelta64(1, 'D')
    days_ahead += 1

  return geo_preds, dates

def runGeneralModel(general_model, hist_cases_gdf, hist_ips_gdf, ips_gdf, current_date, start_date, end_date):
  past_cases = np.array(hist_cases_gdf[CASES_COL])
  past_npis = np.array(hist_ips_gdf[NPI_COLS])
  future_npis = np.array(ips_gdf[NPI_COLS])
        
  # Make prediction for each day
  geo_preds = []
  days_ahead = 0
  while current_date <= end_date:
      # Prepare data
      X_cases = past_cases[-general_nb_lookback_days:, 1] if general_rolling else past_cases[-general_nb_lookback_days:, 0]
      X_npis = past_npis[-general_nb_lookback_days:]
      X_npis_or = np.bitwise_or.reduce(X_npis, axis=0)
      X = np.concatenate([X_cases.flatten(), X_npis_or.flatten()])
      
      # Make the prediction (reshape so that sklearn is happy)
      pred = general_model.predict(X.reshape(1, -1))[0]
      new_row = [pred]
      if general_rolling:
          # Convert rolling avg to actual cases
          pred = 7*(pred-past_cases[-1, 1])+past_cases[-7, 0]
      pred = max(0, pred)  # Do not allow predicting negative cases
      new_row = [pred] + new_row
      # Add if it's a requested date
      if current_date >= start_date:
          geo_preds.append(pred)
      # Append the prediction and npi's for next day
      # in order to rollout predictions for further days.
      past_cases = np.vstack([past_cases, new_row])
      past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + 1], axis=0)

      # Move to next day
      current_date = current_date + np.timedelta64(1, 'D')
      days_ahead += 1
  return geo_preds

def predict_df(start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
    """
    Generates a file with daily new cases predictions for the given countries, regions and npis, between
    start_date and end_date, included.
    :param start_date_str: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date_str: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception_date and end_date
    :param verbose: True to print debug logs
    :return: a Pandas DataFrame containing the predictions
    """
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    if us_static:
      # Load static data
      census_data, census_cols = load_census_data()
      lodes_data, lodes_cols = load_lodes_data()
    
    # Load historical intervention plans, since inception
    hist_ips_df = pd.read_csv(path_to_ips_file,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str},
                              error_bad_lines=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data",
    hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)
    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in NPI_COLS_CATEGORICAL:
      hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))
    # One-hot encode NPI columns
    for col in NPI_COLS_CATEGORICAL:
      one_hot = pd.get_dummies(hist_ips_df[col],prefix=col)
      hist_ips_df = hist_ips_df.drop(col, axis=1)
      hist_ips_df = hist_ips_df.join(one_hot)
    # Maintain same levels as trained on - a different ips set could cause issues otherwise
    for col in NPI_COLS:
      if col not in hist_ips_df:
        hist_ips_df[col] = 0
    hist_ips_df = hist_ips_df[ID_COLS + NPI_COLS]

    # Intervention plans to forecast for: those between start_date and end_date
    ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]

    # Load historical data to use in making predictions in the same way
    # This is the data we trained on
    # We stored it locally as for predictions there will be no access to the internet
    hist_cases_df = pd.read_csv(OXFORD_DATA_FILE,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    hist_cases_df['GeoID'] = hist_cases_df['CountryName'] + '__' + hist_cases_df['RegionName'].astype(str)
    # Add new cases column
    hist_cases_df['NewCases'] = hist_cases_df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    # Fill any missing case values by interpolation and setting NaNs to 0
    hist_cases_df.update(hist_cases_df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))
    hist_cases_df['7DMA'] = hist_cases_df.groupby("GeoID")['NewCases'].rolling(7, center=False).mean().reset_index(0, drop=True)
    hist_cases_df["7DMA"]= hist_cases_df["7DMA"].fillna(0)

    # Keep only the id and cases columns
    hist_cases_df = hist_cases_df[ID_COLS + CASES_COL]

    # Load US model
    with open(US_MODEL_FILE, 'rb') as model_file:
        us_model = pickle.load(model_file)
    
    # Load selected feature index
    us_fs_index = None
    if os.path.exists(FS_INDEX_FILE):
        with open(FS_INDEX_FILE, 'rb') as f:
            us_fs_index = np.load(f,allow_pickle=True)

    # Load general model
    with open(GENERAL_MODEL_FILE, 'rb') as model_file:
      general_model = pickle.load(model_file)

    # Make predictions for each country,region pair
    geo_pred_dfs = []
    for g in ips_df.GeoID.unique():
        if verbose:
            print('\nPredicting for', g)

        # Pull out all relevant data for country c
        # Start predicting from start_date, unless there's a gap since last known date
        hist_cases_gdf = hist_cases_df.loc[hist_cases_df.GeoID == g]
        last_known_date = hist_cases_gdf.Date.max()        
        current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
        hist_cases_gdf = hist_cases_gdf.loc[hist_cases_df.Date < current_date]
        hist_ips_gdf = hist_ips_df[(hist_ips_df.GeoID == g) & (hist_ips_df.Date < current_date)]
        ips_gdf = ips_df[ips_df.GeoID == g]

        countryName = g.split("__")[0]
        regionName = g.split("__")[1]
        if countryName == 'United States' and regionName != 'nan' and regionName != 'Virgin Islands':
          geo_preds = runUSModel(us_model, hist_cases_gdf, hist_ips_gdf, ips_gdf, census_data, lodes_data, census_cols, lodes_cols, us_fs_index, current_date, start_date, end_date)
          # Create geo_pred_df with pred column
          geo_pred_df = ips_gdf[ID_COLS].copy()
          geo_pred_df['PredictedDailyNewCases'] = inverse_transform(cases_pipe, np.array(geo_preds,dtype=np.float64)) if us_transform else geo_preds
          geo_pred_dfs.append(geo_pred_df)
        elif countryName in ARIMA_REGIONS:
          geo_preds, dates = runARIMAModel(countryName, hist_cases_gdf, current_date, start_date, end_date)
          # Create geo_pred_df with pred column
          geo_pred_df = ips_gdf[ID_COLS].copy()
          geo_pred_df['PredictedDailyNewCases'] = geo_preds
          geo_pred_df['CountryName'] = countryName
          geo_pred_df['RegionName'] = regionName
          geo_pred_df['GeoID'] = g
          geo_pred_df['Date'] = dates
          geo_pred_dfs.append(geo_pred_df)
        else:
          geo_preds = runGeneralModel(general_model, hist_cases_gdf, hist_ips_gdf, ips_gdf, current_date, start_date, end_date)
          geo_pred_df = ips_gdf[ID_COLS].copy()
          geo_pred_df['PredictedDailyNewCases'] = inverse_transform(cases_pipe, np.array(geo_preds,dtype=np.float64)) if general_transform else geo_preds
          geo_pred_dfs.append(geo_pred_df)


    # Combine all predictions into a single dataframe
    pred_df = pd.concat(geo_pred_dfs)

    # Drop GeoID column to match expected output format
    pred_df = pred_df.drop(columns=['GeoID'])
    return pred_df

# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
