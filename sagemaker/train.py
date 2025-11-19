import os
import pandas as pd
import numpy as np
from prophet import Prophet
import boto3
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

#config
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'stock-pipeline-data')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'stock-pipeline-models')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')

s3_client = boto3.client('s3', region_name=AWS_REGION)

def download_data_from_s3(bucket, prefix='cleaned/'):

    print(f"Downloading data from s3://{bucket}/{prefix}")
    
    #list parquet files
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        raise ValueError(f"No files found in s3://{bucket}/{prefix}")
    
    #read in parquet files
    dfs = []
    for obj in response['Contents']:
        if obj['Key'].endswith('.parquet'):
            local_file = f"/tmp/{os.path.basename(obj['Key'])}"
            s3_client.download_file(bucket, obj['Key'], local_file)
            df = pd.read_parquet(local_file)
            dfs.append(df)
            os.remove(local_file)
    
    #combine
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} records")
    #get most common ticker
    most_common = df['ticker_id'].mode()[0]
    df = df[df['ticker_id']==most_common]
    return df

def prepare_data_for_prophet(df):
    """Prepare data in Prophet's required format"""
    print("Preparing data for Prophet...")
    
    #ds and y required by Prophet
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['Date']),
        'y': df['Close']
    })
    
    #sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    #remove dupes
    prophet_df = prophet_df.drop_duplicates(subset=['ds'])
    
    print(f"Prepared {len(prophet_df)} records for training")
    print(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    
    return prophet_df

#train prophet model
def train_prophet_model(df):

    print("Training Prophet model...")
    
    #base params
    model = Prophet(
        changepoint_prior_scale=0.05,  #trend flexibility
        seasonality_prior_scale=10.0,   #seasonal flexibility
        seasonality_mode='multiplicative',  #prices
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    #fir model
    model.fit(df)
    
    print("Model training completed!")
    return model

def generate_predictions(model, periods=365):
    """Generate future predictions"""
    print(f"Generating predictions for {periods} days...")
    
    #create pred df
    future = model.make_future_dataframe(periods=periods)
    
    #make pred
    forecast = model.predict(future)
    
    return forecast

def visualize_results(model, forecast, df, output_dir='/tmp'):
    """Create visualizations"""
    print("Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    #forecast plot
    fig1 = model.plot(forecast, figsize=(12, 6))
    plt.title('Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #component plot
    fig2 = model.plot_components(forecast, figsize=(12, 10))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #hisorical vs predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['ds'], df['y'], 'k.', label='Actual', markersize=2)
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Predicted', linewidth=1)
    ax.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'], 
                     alpha=0.3, 
                     label='Uncertainty')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Historical Data vs Predictions')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #prediction distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=forecast, x='yhat', bins=50, kde=True, ax=ax)
    ax.set_xlabel('Predicted Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Prices')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def calculate_metrics(model, df):
    """Calculate model performance metrics"""
    print("Calculating performance metrics...")
    
    #make predictions
    forecast = model.predict(df)
    
    #metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(df['y'], forecast['yhat'])
    mse = mean_squared_error(df['y'], forecast['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(df['y'], forecast['yhat'])
    
    #MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((df['y'] - forecast['yhat']) / df['y'])) * 100
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    return metrics

def save_model_and_artifacts(model, forecast, metrics, bucket):
    """Save model and artifacts to S3"""
    print(f"Saving model and artifacts to s3://{bucket}/models/")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_prefix = f'models/{timestamp}'
    
    #save model
    model_path = '/tmp/prophet_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    s3_client.upload_file(model_path, bucket, f'{model_prefix}/model.pkl')
    
    #save forecast
    forecast_path = '/tmp/forecast.csv'
    forecast.to_csv(forecast_path, index=False)
    s3_client.upload_file(forecast_path, bucket, f'{model_prefix}/forecast.csv')
    
    #save metrics
    metrics_path = '/tmp/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    s3_client.upload_file(metrics_path, bucket, f'{model_prefix}/metrics.json')
    
    #upload visualizations
    viz_files = ['forecast.png', 'components.png', 'actual_vs_predicted.png', 
                 'prediction_distribution.png']
    for viz_file in viz_files:
        local_path = f'/tmp/{viz_file}'
        if os.path.exists(local_path):
            s3_client.upload_file(local_path, bucket, f'{model_prefix}/{viz_file}')
    
    #save latest model
    latest_path = '/tmp/latest.txt'
    with open(latest_path, 'w') as f:
        f.write(timestamp)
    s3_client.upload_file(latest_path, bucket, 'models/latest.txt')
    
    print(f"All artifacts saved with timestamp: {timestamp}")
    return timestamp

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Stock Price Forecasting with Prophet")
    print("=" * 60)
    
    try:
        #download data
        df = download_data_from_s3(BUCKET_NAME)
        
        #prep data
        prophet_df = prepare_data_for_prophet(df)
        
        #trainingh
        model = train_prophet_model(prophet_df)
        
        #predictions
        forecast = generate_predictions(model, periods=365)
        
        #metrics
        metrics = calculate_metrics(model, prophet_df)
        
        #visualizations
        visualize_results(model, forecast, prophet_df)
        
        #save
        timestamp = save_model_and_artifacts(model, forecast, metrics, MODEL_BUCKET)
        
        print("=" * 60)
        print(f"Training completed successfully!")
        print(f"Model timestamp: {timestamp}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()