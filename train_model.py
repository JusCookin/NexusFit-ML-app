import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def debug_dataset(df):
    """Debug function to analyze dataset"""
    print("=== DATASET ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nUnique values per column:")
    for col in df.columns:
        unique_vals = df[col].unique()
        print(f"{col}: {len(unique_vals)} unique values")
        if len(unique_vals) <= 10:
            print(f"  Values: {unique_vals}")
        else:
            print(f"  Sample values: {unique_vals[:5]}...")
    print("=" * 50)

def prepare_dataset():
    """Load and prepare the dataset"""
    try:
        # Load dataset
        df = pd.read_csv("gym recommendation.csv")
        print("✓ Dataset loaded successfully")
        
        # Debug dataset
        debug_dataset(df)
        
        # Clean dataset - remove rows with missing values
        initial_shape = df.shape
        df = df.dropna()
        print(f"✓ Removed {initial_shape[0] - df.shape[0]} rows with missing values")
        
        # Clean string columns - remove extra whitespace
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: 'gym recommendation.csv' file not found!")
        print("Please make sure the dataset file is in the same directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def calculate_bmi(df):
    """Calculate BMI with proper handling"""
    if 'BMI' not in df.columns:
        if 'Height' in df.columns and 'Weight' in df.columns:
            # Check if height is in cm (convert to meters)
            if df['Height'].mean() > 10:
                print("Height appears to be in cm, converting to meters...")
                df['Height'] = df['Height'] / 100
            
            # Calculate BMI
            df['BMI'] = df['Weight'] / (df['Height'] ** 2)
            print("✓ BMI calculated successfully")
        else:
            print("❌ Cannot calculate BMI: Height or Weight column missing")
            return False
    else:
        print("✓ BMI column already exists")
    return True

def create_bmi_level(df):
    """Create BMI level column if it doesn't exist"""
    if 'Level' not in df.columns and 'BMI' in df.columns:
        df['Level'] = pd.cut(df['BMI'], 
                           bins=[0, 18.5, 24.9, 29.9, float('inf')], 
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        print("✓ BMI Level column created from BMI values")
    elif 'Level' in df.columns:
        print("✓ Level column already exists")
    else:
        print("❌ Cannot create Level column: BMI column missing")
        return False
    return True

def encode_features(df):
    """Encode categorical features with error handling"""
    label_encoders = {}
    categorical_cols = ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Level']
    
    # Store dataset information for later use
    dataset_info = {}
    
    for col in categorical_cols:
        if col in df.columns:
            try:
                # Store original unique values
                dataset_info[col] = df[col].unique().tolist()
                
                # Create and fit label encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
                
                print(f"✓ Encoded {col}: {le.classes_}")
                
            except Exception as e:
                print(f"❌ Error encoding {col}: {e}")
                return None, None
        else:
            print(f"⚠️  Warning: Column '{col}' not found in dataset")
    
    return label_encoders, dataset_info

def prepare_features_target(df):
    """Prepare features and target variables"""
    # Define all possible feature columns
    all_feature_cols = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 
                       'Diabetes', 'BMI', 'Level', 'Fitness Goal']
    
    # Use only available columns
    available_features = [col for col in all_feature_cols if col in df.columns]
    print(f"✓ Available features: {available_features}")
    
    X = df[available_features]
    
    # Find target column
    target_col = None
    possible_targets = ['Fitness Type', 'Recommendation', 'Exercise Type', 'Workout Type']
    
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Use last column as target
        target_col = df.columns[-1]
        print(f"⚠️  Using last column '{target_col}' as target")
    else:
        print(f"✓ Using '{target_col}' as target column")
    
    y = df[target_col]
    
    return X, y, target_col

def train_model():
    """Main training function"""
    print("🚀 Starting model training...")
    
    # Step 1: Load and prepare dataset
    df = prepare_dataset()
    if df is None:
        return False
    
    # Step 2: Calculate BMI
    if not calculate_bmi(df):
        return False
    
    # Step 3: Create BMI level
    if not create_bmi_level(df):
        return False
    
    # Step 4: Encode categorical features
    label_encoders, dataset_info = encode_features(df)
    if label_encoders is None:
        return False
    
    # Step 5: Prepare features and target
    X, y, target_col = prepare_features_target(df)
    
    # Step 6: Encode target variable
    try:
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))
        print(f"✓ Target encoded: {target_encoder.classes_}")
    except Exception as e:
        print(f"❌ Error encoding target: {e}")
        return False
    
    # Step 7: Split data for validation
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"✓ Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(f"⚠️  Could not stratify split: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    # Step 8: Train model with better parameters
    try:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        model.fit(X_train, y_train)
        print("✓ Model training completed")
        
        # Evaluate model
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        print(f"✓ Training Accuracy: {train_accuracy:.3f}")
        print(f"✓ Testing Accuracy: {test_accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Feature Importance:")
        print(feature_importance)
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    
    # Step 9: Save model and encoders
    try:
        joblib.dump(model, 'model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(target_encoder, 'target_encoder.pkl')
        joblib.dump(dataset_info, 'dataset_info.pkl')
        
        print("✅ Model and encoders saved successfully!")
        print("Files created:")
        print("  - model.pkl")
        print("  - label_encoders.pkl") 
        print("  - target_encoder.pkl")
        print("  - dataset_info.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def test_saved_model():
    """Test the saved model with sample data"""
    try:
        print("\n🧪 Testing saved model...")
        
        # Load saved components
        model = joblib.load('model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        dataset_info = joblib.load('dataset_info.pkl')
        
        print("✓ All components loaded successfully")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Sex': ['Male'],
            'Age': [25],
            'Height': [1.75],
            'Weight': [70],
            'Hypertension': ['No'],
            'Diabetes': ['No'],
            'BMI': [22.86],
            'Level': ['Normal'],
            'Fitness Goal': ['Weight Loss']
        })
        
        # Encode sample data
        for col in ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Level']:
            if col in label_encoders and col in sample_data.columns:
                # Check if value exists in encoder classes
                if sample_data[col].iloc[0] in label_encoders[col].classes_:
                    sample_data[col] = label_encoders[col].transform(sample_data[col])
                else:
                    print(f"⚠️  Value '{sample_data[col].iloc[0]}' not in {col} classes, using 0")
                    sample_data[col] = 0
        
        # Make prediction
        prediction = model.predict(sample_data)
        result = target_encoder.inverse_transform(prediction)[0]
        
        print(f"✅ Test prediction successful: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🏋️  Gym Recommendation Model Training")
    print("=" * 40)
    
    # Train the model
    success = train_model()
    
    if success:
        # Test the saved model
        test_saved_model()
        print("\n🎉 Training completed successfully!")
        print("You can now run your Flask application.")
    else:
        print("\n❌ Training failed. Please check the errors above.")