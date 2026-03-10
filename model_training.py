import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_models():
    # Load dataset
    data_path = r'C:\Users\user\.gemini\antigravity\scratch\mindscan_ai\mental_health_data.csv'
    df = pd.read_csv(data_path)
    
    # Preprocessing
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    le_target = LabelEncoder()
    df['Depression_State'] = le_target.fit_transform(df['Depression_State'])
    # Store labels for reference: No Depression: 2, Mild: 1, Moderate: 3, Severe: 0 (depending on alphabetical order)
    # Actually LabelEncoder sorts alphabetically: Mild(0), Moderate(1), No Depression(2), Severe(3)
    target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
    print(f"Target Mapping: {target_mapping}")

    X = df.drop('Depression_State', axis=1)
    y = df['Depression_State']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
    # As per abstract, Decision Tree is the best or preferred model here
    best_model = models['Decision Tree']
    
    # Save artifacts
    save_dir = r'C:\Users\user\.gemini\antigravity\scratch\mindscan_ai'
    with open(os.path.join(save_dir, 'decision_tree_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, 'gender_encoder.pkl'), 'wb') as f:
        pickle.dump(le_gender, f)
    with open(os.path.join(save_dir, 'target_encoder.pkl'), 'wb') as f:
        pickle.dump(le_target, f)
        
    # Save results for visualization in the app
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    results_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    print("Models trained and artifacts saved.")

if __name__ == "__main__":
    train_models()
