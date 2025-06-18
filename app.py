import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, render_template, request
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def debug_dataset():
    """Debug function to check dataset labels"""
    try:
        df = pd.read_csv("gym recommendation.csv")
        print("=== DATASET DEBUG INFO ===")
        print("Columns:", df.columns.tolist())
        print("\nDataset shape:", df.shape)
        print("\nUnique values in each column:")
        for col in df.columns:
            unique_vals = df[col].unique()
            print(f"{col}: {unique_vals} (count: {len(unique_vals)})")
        print("=" * 50)
        return df
    except FileNotFoundError:
        print("Dataset file 'gym recommendation.csv' not found!")
        return None
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None

def safe_label_encode(encoder, values, column_name):
    """Safely encode labels, handling unseen values"""
    try:
        return encoder.transform(values)
    except ValueError as e:
        print(f"Warning: Unseen labels in {column_name}: {e}")
        # Handle unseen labels by mapping them to the most common class
        encoded_values = []
        for value in values:
            if value in encoder.classes_:
                encoded_values.append(encoder.transform([value])[0])
            else:
                print(f"Mapping unseen value '{value}' to first class '{encoder.classes_[0]}'")
                encoded_values.append(0)  # Use first class
        return np.array(encoded_values)

def create_robust_mappings():
    """Create comprehensive mappings for different variations"""
    return {
        'bmi_mapping': {
            'Underweight': ['Underweight', 'Under Weight', 'Thin', 'Low', 'Below Normal', 'Skinny'],
            'Normal': ['Normal', 'Normal Weight', 'Healthy', 'Average', 'Good', 'Ideal'],
            'Overweight': ['Overweight', 'Over Weight', 'High', 'Above Normal', 'Heavy'],
            'Obese': ['Obese', 'Obesity', 'Very High', 'Extremely High', 'Severely Overweight', 'Very Heavy']
        },
        'goal_mapping': {
            'Weight Loss': ['Weight Loss', 'Lose Weight', 'Fat Loss', 'Cut', 'Cutting', 'Slim Down', 'Reduce Weight'],
            'Weight Gain': ['Weight Gain', 'Gain Weight', 'Bulk', 'Bulking', 'Mass Gain', 'Increase Weight'],
            'Muscle Gain': ['Muscle Gain', 'Build Muscle', 'Muscle Building', 'Strength', 'Hypertrophy', 'Muscle Growth'],
            'Maintain': ['Maintain', 'Maintenance', 'Stay Fit', 'General Fitness', 'Keep Fit', 'Fitness'],
            'Endurance': ['Endurance', 'Cardio', 'Stamina', 'Aerobic', 'Running', 'Cycling'],
            'Flexibility': ['Flexibility', 'Stretching', 'Yoga', 'Mobility', 'Range of Motion']
        },
        'sex_mapping': {
            'Male': ['Male', 'M', 'Man', 'male', 'm'],
            'Female': ['Female', 'F', 'Woman', 'female', 'f']
        },
        'binary_mapping': {
            'Yes': ['Yes', 'Y', 'yes', 'y', '1', 1, True, 'true', 'True'],
            'No': ['No', 'N', 'no', 'n', '0', 0, False, 'false', 'False']
        }
    }

def map_value_to_dataset(user_value, dataset_values, mapping_dict=None):
    """Map user input to dataset's expected values"""
    # Direct match first
    if user_value in dataset_values:
        return user_value
    
    # If mapping dictionary provided, try to map
    if mapping_dict:
        for dataset_val in dataset_values:
            for standard_val, variations in mapping_dict.items():
                if (user_value == standard_val and dataset_val in variations) or \
                   (dataset_val in variations and user_value == standard_val) or \
                   (user_value in variations and dataset_val == standard_val):
                    return dataset_val
    
    # Case-insensitive match
    for dataset_val in dataset_values:
        if str(user_value).lower() == str(dataset_val).lower():
            return dataset_val
    
    # Partial match
    for dataset_val in dataset_values:
        if str(user_value).lower() in str(dataset_val).lower() or \
           str(dataset_val).lower() in str(user_value).lower():
            return dataset_val
    
    # Return first available value as fallback
    print(f"Warning: Could not map '{user_value}' to any dataset value. Using '{dataset_values[0]}'")
    return dataset_values[0]

def load_or_create_model():
    """Load existing model or create new one"""
    try:
        model = joblib.load('model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        dataset_info = joblib.load('dataset_info.pkl')
        print("✓ Loaded existing model successfully")
        return model, label_encoders, target_encoder, dataset_info
    except FileNotFoundError:
        print("Creating new model...")
        return create_model()
    except Exception as e:
        print(f"Error loading model: {e}. Creating new model...")
        return create_model()

def create_model():
    """Create and train the model with robust error handling"""
    df = debug_dataset()
    if df is None:
        raise FileNotFoundError("Cannot create model without dataset")
    
    try:
        # Clean the dataset
        df = df.dropna()  # Remove rows with missing values
        print(f"Dataset shape after cleaning: {df.shape}")
        
        # Calculate BMI if not present
        if 'BMI' not in df.columns:
            if 'Height' in df.columns and 'Weight' in df.columns:
                # Ensure height is in meters (convert if in cm)
                if df['Height'].mean() > 10:  # Likely in cm
                    df['Height'] = df['Height'] / 100
                df['BMI'] = df['Weight'] / (df['Height'] ** 2)
                print("✓ Calculated BMI column")
        
        # Handle Level column
        if 'Level' not in df.columns and 'BMI' in df.columns:
            df['Level'] = pd.cut(df['BMI'], 
                               bins=[0, 18.5, 24.9, 29.9, float('inf')], 
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            print("✓ Created Level column from BMI")
        
        # Store dataset information for later use
        dataset_info = {}
        categorical_cols = ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Level']
        
        for col in categorical_cols:
            if col in df.columns:
                dataset_info[col] = df[col].unique().tolist()
        
        # Encode categorical columns
        label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Clean the column
                df[col] = df[col].astype(str).str.strip()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
                print(f"✓ Encoded {col}: {le.classes_}")
        
        # Prepare features - use only available columns
        all_feature_cols = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal']
        available_features = [col for col in all_feature_cols if col in df.columns]
        print(f"Available features: {available_features}")
        
        X = df[available_features]
        
        # Determine target column
        target_col = None
        possible_targets = ['Fitness Type', 'Recommendation', 'Exercise Type', 'Workout Type']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Use last column as target
            target_col = df.columns[-1]
            print(f"Using last column '{target_col}' as target")
        
        y = df[target_col]
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))
        
        print(f"Target classes: {target_encoder.classes_}")
        print(f"Training data shape: {X.shape}")
        
        # Train model
        model = RandomForestClassifier(
            random_state=42, 
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'  # Handle imbalanced classes
        )
        model.fit(X, y_encoded)
        
        # Save everything
        joblib.dump(model, 'model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(target_encoder, 'target_encoder.pkl')
        joblib.dump(dataset_info, 'dataset_info.pkl')
        
        print("✓ Model trained and saved successfully")
        return model, label_encoders, target_encoder, dataset_info
        
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        raise

# Initialize model with error handling
try:
    model, label_encoders, target_encoder, dataset_info = load_or_create_model()
    print("✓ Model initialization successful")
    print(f"Available dataset values: {dataset_info}")
except Exception as e:
    print(f"Critical error initializing model: {e}")
    model, label_encoders, target_encoder, dataset_info = None, {}, None, {}

def calculate_bmi(height, weight):
    """Calculate BMI with input validation"""
    try:
        # Convert height to meters if it appears to be in cm
        if height > 10:
            height = height / 100
        
        bmi = weight / (height ** 2)
        return round(bmi, 2)
    except (ValueError, ZeroDivisionError):
        return 22.0  # Default normal BMI

def get_bmi_level(bmi):
    """Get BMI level based on standard ranges"""
    try:
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    except:
        return "Normal"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if model is None:
            return render_template('result.html', 
                                 message="Model not initialized. Please check your dataset and try again.")
        
        # Get form data with error handling
        try:
            sex = str(request.form.get('sex', 'Male')).strip()
            age = int(request.form.get('age', 25))
            height = float(request.form.get('height', 1.7))
            weight = float(request.form.get('weight', 70))
            hypertension = str(request.form.get('hypertension', 'No')).strip()
            diabetes = str(request.form.get('diabetes', 'No')).strip()
            fitness_goal = str(request.form.get('goal', 'Maintain')).strip()
        except (ValueError, TypeError) as e:
            return render_template('result.html', 
                                 message=f"Invalid input data: {e}. Please check your inputs.")
        
        # Calculate BMI
        bmi = calculate_bmi(height, weight)
        calculated_level = get_bmi_level(bmi)
        
        # Get mappings
        mappings = create_robust_mappings()
        
        # Map user inputs to dataset values
        if 'Sex' in dataset_info:
            sex = map_value_to_dataset(sex, dataset_info['Sex'], mappings['sex_mapping'])
        
        if 'Hypertension' in dataset_info:
            hypertension = map_value_to_dataset(hypertension, dataset_info['Hypertension'], mappings['binary_mapping'])
        
        if 'Diabetes' in dataset_info:
            diabetes = map_value_to_dataset(diabetes, dataset_info['Diabetes'], mappings['binary_mapping'])
        
        if 'Fitness Goal' in dataset_info:
            fitness_goal = map_value_to_dataset(fitness_goal, dataset_info['Fitness Goal'], mappings['goal_mapping'])
        
        if 'Level' in dataset_info:
            mapped_level = map_value_to_dataset(calculated_level, dataset_info['Level'], mappings['bmi_mapping'])
        else:
            mapped_level = calculated_level
        
        print(f"Mapped values - Sex: {sex}, Goal: {fitness_goal}, Level: {mapped_level}")
        
        # Prepare user data
        user_data = pd.DataFrame({
            'Sex': [sex],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Hypertension': [hypertension],
            'Diabetes': [diabetes],
            'BMI': [bmi],
            'Level': [mapped_level],
            'Fitness Goal': [fitness_goal]
        })
        
        # Encode categorical variables with safe encoding
        for col in ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Level']:
            if col in label_encoders and col in user_data.columns:
                user_data[col] = safe_label_encode(label_encoders[col], user_data[col], col)
        
        # Ensure all required features are present
        feature_cols = list(label_encoders.keys()) + ['Age', 'Height', 'Weight', 'BMI']
        available_cols = [col for col in feature_cols if col in user_data.columns]
        
        # Make prediction
        try:
            prediction = model.predict(user_data[available_cols])
            fitness_type = target_encoder.inverse_transform(prediction)[0]
            
            # Calculate prediction confidence
            prediction_proba = model.predict_proba(user_data[available_cols])
            confidence = round(max(prediction_proba[0]) * 100, 2)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            fitness_type = "General Fitness"
            confidence = 0
        
        # Generate recommendations
        result = generate_recommendations(fitness_type, fitness_goal, calculated_level, bmi)
        result['Confidence'] = f"{confidence}%"
        
        return render_template('result.html', 
                             bmi=bmi, 
                             level=calculated_level, 
                             result=result)
    
    except Exception as e:
        print(f"Error in recommendation: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', 
                             message=f"An error occurred: {str(e)}. Please try again with different values.")

def generate_recommendations(fitness_type, goal, bmi_level, bmi_value):
    """Generate comprehensive recommendations"""
    
    recommendations = {
        'Fitness Type': fitness_type,
        'Goal': goal,
        'BMI Level': bmi_level,
        'Exercises': '',
        'Equipment': '',
        'Diet': '',
        'Schedule': '',
        'Recommendation': ''
    }
    
    # Goal-based recommendations
    goal_lower = goal.lower()
    
    if 'weight loss' in goal_lower or 'loss' in goal_lower or 'cut' in goal_lower:
        recommendations['Exercises'] = 'High-intensity cardio (30-45 min), Full-body strength training, HIIT workouts 3-4x/week, Walking 8000+ steps daily'
        recommendations['Equipment'] = 'Treadmill, Elliptical, Dumbbells, Kettlebells, Resistance bands, Jump rope'
        recommendations['Diet'] = 'Caloric deficit of 500-750 calories, High protein (1.2-1.6g/kg), Reduce processed foods, Increase vegetables and fiber'
        recommendations['Schedule'] = 'Mon/Wed/Fri: Strength training, Tue/Thu: Cardio, Weekend: Active recovery'
        
    elif 'weight gain' in goal_lower or 'gain' in goal_lower or 'bulk' in goal_lower:
        recommendations['Exercises'] = 'Compound movements: Squats, Deadlifts, Bench press, Rows. 3-4 sets of 6-8 reps with progressive overload'
        recommendations['Equipment'] = 'Barbell, Dumbbells, Power rack, Bench press, Pull-up bar, Cable machine'
        recommendations['Diet'] = 'Caloric surplus of 300-500 calories, High protein (1.6-2.2g/kg), Complex carbs, Healthy fats, Frequent meals'
        recommendations['Schedule'] = '4-5 days/week strength training, 2 days rest, Focus on major muscle groups'
        
    elif 'muscle' in goal_lower or 'strength' in goal_lower or 'hypertrophy' in goal_lower:
        recommendations['Exercises'] = 'Hypertrophy training: 8-12 reps, 3-4 sets, Rest 60-90 seconds, Focus on time under tension'
        recommendations['Equipment'] = 'Free weights, Cable machines, Adjustable bench, Various grips and attachments'
        recommendations['Diet'] = 'High protein (1.8-2.5g/kg), Post-workout nutrition within 2 hours, Adequate carbs for recovery'
        recommendations['Schedule'] = 'Upper/Lower split or Push/Pull/Legs, 4-6 days/week, 48-72 hours rest per muscle group'
        
    elif 'endurance' in goal_lower or 'cardio' in goal_lower or 'stamina' in goal_lower:
        recommendations['Exercises'] = 'Long steady-state cardio, Interval training, Circuit training, Sport-specific activities'
        recommendations['Equipment'] = 'Cardio machines, Running shoes, Heart rate monitor, Cycling equipment'
        recommendations['Diet'] = 'Adequate carbohydrates for energy, Proper hydration, Electrolyte balance'
        recommendations['Schedule'] = '5-6 days cardio, 2-3 days strength training, Progressive distance/time increases'
        
    else:  # Maintain or general fitness
        recommendations['Exercises'] = 'Balanced routine: 150 min moderate cardio/week + 2-3 strength sessions, Flexibility work'
        recommendations['Equipment'] = 'Basic gym equipment, Dumbbells, Resistance bands, Cardio machines'
        recommendations['Diet'] = 'Balanced macronutrients, Whole foods, Adequate hydration, Regular meal timing'
        recommendations['Schedule'] = '3-4 days/week mixed training, 2-3 rest days, Include variety to prevent boredom'
    
    # BMI-based adjustments
    if bmi_level == 'Underweight':
        recommendations['Recommendation'] = f'With BMI {bmi_value}, focus on healthy weight gain through strength training and increased caloric intake. Consult a nutritionist for personalized meal planning.'
    elif bmi_level == 'Overweight':
        recommendations['Recommendation'] = f'With BMI {bmi_value}, prioritize gradual weight loss (1-2 lbs/week) through moderate caloric deficit and regular exercise. Focus on sustainable lifestyle changes.'
    elif bmi_level == 'Obese':
        recommendations['Recommendation'] = f'With BMI {bmi_value}, start with low-impact exercises and consult healthcare providers. Focus on sustainable lifestyle changes and consider professional guidance.'
    else:
        recommendations['Recommendation'] = f'With BMI {bmi_value} in the normal range, focus on maintaining your current weight while working towards your fitness goals.'
    
    # Add safety notes
    recommendations['Recommendation'] += ' Always warm up before exercising and cool down afterwards. Listen to your body and adjust intensity as needed.'
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)