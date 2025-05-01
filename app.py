import os
import time
import gspread
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
from schedule import every, run_pending

# Load environment variables
load_dotenv()

class PestDetectionSystem:
    def _init_(self):
        # Load ML model with absolute path
        self.model = self.load_model(
            os.path.join(os.path.dirname(_file_), 'models', 'pest_detection_model_2.pkl')
        )
        
        # Initialize Google Sheets connection
        self.scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(
            os.path.join(os.path.dirname(_file_), 'config', 'credentials.json'), 
            self.scope
        )
        self.client = gspread.authorize(self.creds)
        self.sheet = self.client.open_by_key(os.getenv('SHEET_ID')).sheet1

    def load_model(self, model_path):
        """Load trained ML model with error handling"""
        try:
            return joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_new_entries(self):
        """Main processing loop for new sensor data"""
        try:
            records = self.sheet.get_all_records()
            df = pd.DataFrame(records)
            
            # Initialize columns if missing
            self.ensure_columns_exist()
            
            # Process unmarked rows
            unprocessed = df[df['Processed'].astype(str) == '']
            for idx in unprocessed.index:
                self.process_row(idx + 2)  # +2 for header and 1-based index
                
        except Exception as e:
            print(f"Processing error: {str(e)}")

    def ensure_columns_exist(self):
        """Ensure required columns exist in the sheet"""
        header = self.sheet.row_values(1)
        required_cols = ['Processed', 'Prediction']
        updates = False
        
        for col in required_cols:
            if col not in header:
                self.sheet.insert_cols([col], len(header)+1)
                header.append(col)
                updates = True
                time.sleep(1)  # Rate limit
                
        if updates:
            print("Added missing columns")

    def process_row(self, row_num):
        """Process individual sensor data row"""
        try:
            row_data = self.sheet.row_values(row_num)
            features = self.extract_features(row_data)
            
            if not features:
                self.mark_processed(row_num, "Invalid data")
                return

            prediction = self.model.predict([features])[0]
            confidence = self.get_confidence(features)
            
            self.update_sheet(row_num, prediction, confidence)
            
        except Exception as e:
            self.mark_processed(row_num, f"Error: {str(e)}")
            print(f"Row {row_num} error: {str(e)}")

    def extract_features(self, row_data):
        """Convert sheet row to feature vector"""
        try:
            return [
                float(row_data[1]),  # Moisture_Sensor
                float(row_data[2]),  # Humidity
                float(row_data[3]),  # Temperature
                float(row_data[4]),  # Infrared_Sensor
                float(row_data[5]),  # Motion_Sensor
                float(row_data[6]),  # Vibration_Sensor
                float(row_data[7])   # Gas_Sensor
            ]
        except (IndexError, ValueError):
            return None

    def get_confidence(self, features):
        """Get prediction confidence score"""
        if hasattr(self.model, 'predict_proba'):
            return round(100 * max(self.model.predict_proba([features])[0]), 2)
        return None

    def update_sheet(self, row_num, prediction, confidence):
        """Update Google Sheet with results"""
        prediction_text = f"{'Pest Detected' if prediction == 1 else 'No Pest'}"
        if confidence:
            prediction_text += f" ({confidence}%)"
            
        self.sheet.update_cell(row_num, 9, prediction_text)
        self.mark_processed(row_num)

    def mark_processed(self, row_num, message=None):
        """Mark row as processed"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = message or f"Processed at {timestamp}"
        self.sheet.update_cell(row_num, 10, status)

if _name_ == "_main_":
    try:
        system = PestDetectionSystem()
        print("🚀 Pest Detection System Started")
        
        # Run every 2 minutes
        every(2).minutes.do(system.process_new_entries)
        
        while True:
            run_pending()
            time.sleep(1)
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)
