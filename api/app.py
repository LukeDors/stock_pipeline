from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import pickle
import uvicorn
import boto3
import os
from io import BytesIO

#config
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'stock-pipeline-models')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')

#initialize FASTAPI
app = FastAPI(
    title="Stock Price Prediction API",
    description="Time series forecasting API using Prophet",
    version="1.0.0"
)

s3_client = boto3.client('s3', region_name=AWS_REGION)

#global vars
model = None
forecast_df = None
last_updated = None

#Pydantic models
class PredictionRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    
class PredictionResponse(BaseModel):
    date: str
    predicted_price: float
    lower_bound: float
    upper_bound: float
    trend: float
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_updated: Optional[str]
    
class BatchPredictionRequest(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

def load_model_from_s3():
    """Load the latest model from S3"""
    global model, forecast_df, last_updated
    
    try:
        #model timestamp
        latest_obj = s3_client.get_object(Bucket=MODEL_BUCKET, Key='models/latest.txt')
        timestamp = latest_obj['Body'].read().decode('utf-8').strip()
        
        #load model
        model_key = f'models/{timestamp}/model.pkl'
        model_obj = s3_client.get_object(Bucket=MODEL_BUCKET, Key=model_key)
        model = pickle.loads(model_obj['Body'].read())
        
        #load forecast
        forecast_key = f'models/{timestamp}/forecast.csv'
        forecast_obj = s3_client.get_object(Bucket=MODEL_BUCKET, Key=forecast_key)
        forecast_df = pd.read_csv(BytesIO(forecast_obj['Body'].read()))
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        
        last_updated = timestamp
        
        print(f"Model loaded successfully. Timestamp: {timestamp}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("Loading model on startup...")
    success = load_model_from_s3()
    if not success:
        print("Warning: Failed to load model on startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        last_updated=last_updated
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Get prediction for a specific date"""
    if model is None or forecast_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        #parse date
        target_date = pd.to_datetime(request.date)
        
        #find pred
        prediction = forecast_df[forecast_df['ds'] == target_date]
        
        if prediction.empty:
            #if date not in forecast, generate new prediction
            future = pd.DataFrame({'ds': [target_date]})
            prediction = model.predict(future)
        else:
            prediction = prediction.iloc[0]
        
        return PredictionResponse(
            date=target_date.strftime('%Y-%m-%d'),
            predicted_price=float(prediction['yhat']),
            lower_bound=float(prediction['yhat_lower']),
            upper_bound=float(prediction['yhat_upper']),
            trend=float(prediction['trend'])
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Get predictions for a date range"""
    if model is None or forecast_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        #parse dates
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        #filter forecast for date range
        mask = (forecast_df['ds'] >= start_date) & (forecast_df['ds'] <= end_date)
        predictions = forecast_df[mask]
        
        if predictions.empty:
            #generate new predicitons
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            future = pd.DataFrame({'ds': date_range})
            predictions = model.predict(future)
        
        #format response
        results = []
        for _, row in predictions.iterrows():
            results.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'predicted_price': float(row['yhat']),
                'lower_bound': float(row['yhat_lower']),
                'upper_bound': float(row['yhat_upper']),
                'trend': float(row['trend'])
            })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making predictions: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Reload model from S3"""
    success = load_model_from_s3()
    
    if success:
        return {"status": "success", "message": "Model reloaded successfully", "timestamp": last_updated}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None or forecast_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "loaded",
        "last_updated": last_updated,
        "forecast_start": forecast_df['ds'].min().strftime('%Y-%m-%d'),
        "forecast_end": forecast_df['ds'].max().strftime('%Y-%m-%d'),
        "total_predictions": len(forecast_df)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "reload_model": "/model/reload"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)