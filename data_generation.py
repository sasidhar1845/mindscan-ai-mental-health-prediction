import pandas as pd
import numpy as np
import os

def generate_data(num_samples=1000):
    np.random.seed(42)
    
    # Feature ranges and distributions
    age = np.random.randint(18, 65, num_samples)
    gender = np.random.choice(['Male', 'Female', 'Non-binary'], num_samples)
    sleep_hours = np.random.normal(7, 1.5, num_samples).clip(4, 10)
    physical_activity = np.random.randint(0, 7, num_samples)  # days per week
    stress_level = np.random.randint(1, 11, num_samples)      # 1 to 10 scale
    social_interaction = np.random.randint(1, 11, num_samples) # 1 to 10 scale
    diet_quality = np.random.randint(1, 11, num_samples)      # 1 to 10 scale
    work_pressure = np.random.randint(1, 11, num_samples)     # 1 to 10 scale
    
    # Calculate a score to determine the depression state
    # Lower sleep, lower activity, higher stress, lower social, higher pressure -> more likely depression
    score = (
        (10 - sleep_hours) * 1.5 +
        (7 - physical_activity) * 1.0 +
        stress_level * 2.0 +
        (10 - social_interaction) * 1.5 +
        (10 - diet_quality) * 1.0 +
        work_pressure * 2.0
    )
    
    # Determine category based on score quartiles
    # Adjust thresholds to get a reasonable distribution
    thresholds = np.percentile(score, [25, 50, 75])
    
    depression_state = []
    for s in score:
        if s <= thresholds[0]:
            depression_state.append('No Depression')
        elif s <= thresholds[1]:
            depression_state.append('Mild')
        elif s <= thresholds[2]:
            depression_state.append('Moderate')
        else:
            depression_state.append('Severe')
            
    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Sleep_Hours': sleep_hours,
        'Physical_Activity': physical_activity,
        'Stress_Level': stress_level,
        'Social_Interaction': social_interaction,
        'Diet_Quality': diet_quality,
        'Work_Pressure': work_pressure,
        'Depression_State': depression_state
    })
    
    output_path = r'C:\Users\user\.gemini\antigravity\scratch\mindscan_ai\mental_health_data.csv'
    data.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path}")

if __name__ == "__main__":
    generate_data()
