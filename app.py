import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import shap
from textblob import TextBlob
import time

# Page configuration
st.set_page_config(page_title="MindScan AI Pro", page_icon="🧠", layout="wide")

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'mental_health_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'decision_tree_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
GENDER_ENC_PATH = os.path.join(BASE_DIR, 'gender_encoder.pkl')
TARGET_ENC_PATH = os.path.join(BASE_DIR, 'target_encoder.pkl')
COMPARISON_PATH = os.path.join(BASE_DIR, 'model_comparison.csv')

# Load resources
@st.cache_resource
def load_model_assets():
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None, None, None
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(GENDER_ENC_PATH, 'rb') as f:
            le_gender = pickle.load(f)
        with open(TARGET_ENC_PATH, 'rb') as f:
            le_target = pickle.load(f)
        return model, scaler, le_gender, le_target
    except Exception as e:
        return None, None, None, None

@st.cache_data
def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            return None
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        return None

# Styling
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4CAF50; color: white; font-weight: bold; }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-top: 20px; }
    .chat-bubble { padding: 15px; border-radius: 15px; margin-bottom: 15px; font-size: 16px; line-height: 1.5; color: #ffffff; }
    .user-bubble { background-color: #1976d2; border-left: 5px solid #0d47a1; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); }
    .bot-bubble { background-color: #388e3c; border-left: 5px solid #1b5e20; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); }
    
    /* Breathing animation */
    .breathing-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 350px;
        background: radial-gradient(circle, #f0f4f8 0%, #d9e2ec 100%);
        border-radius: 20px;
    }
    .circle {
        background: linear-gradient(135deg, #81c784 0%, #4caf50 100%);
        height: 120px;
        width: 120px;
        border-radius: 50%;
        animation: breathe 8s infinite ease-in-out;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 10px 25px rgba(76, 175, 80, 0.4);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(2.2); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🧠 MindScan AI Pro: Advanced Mental Health Suite")
    
    # Sidebar for API Key and Navigation
    st.sidebar.image("https://img.icons8.com/plasticine/200/brain.png", width=100)
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Google API Key (for MindBot)", type="password", help="Enable advanced chatbot features with your Gemini API Key")
    if api_key:
        genai.configure(api_key=api_key)
    
    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", ["Home", "Assessment & Explainability", "MindBot Chat", "Intervention Tools", "Data Insights", "Model Performance"])

    model, scaler, le_gender, le_target = load_model_assets()
    df = load_data()

    if page == "Home":
        show_home()
    elif page == "Assessment & Explainability":
        show_assessment(model, scaler, le_gender, le_target, df)
    elif page == "MindBot Chat":
        show_mindbot(api_key)
    elif page == "Intervention Tools":
        show_intervention()
    elif page == "Data Insights":
        show_insights(df)
    elif page == "Model Performance":
        show_performance()

def show_home():
    st.header("Empowering Your Well-being with AI")
    st.write("""
    MindScan AI Pro is a next-generation mental health companion. We move beyond simple assessments by providing **transparency** through Explainable AI, **empathy** through Generative AI, and **immediate relief** through interactive intervention tools.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🤖 MindBot\nEngage in supportive, AI-driven conversations to express your feelings and receive empathetic feedback.")
    with col2:
        st.success("### ⚖️ Explainable AI\nNo more black boxes. See exactly which lifestyle factors contributed most to your self-assessment.")
    with col3:
        st.warning("### 🧘 Interventions\nInstant access to guided breathing tools and mental health resources whenever you need them.")

    st.markdown("---")
    st.markdown("### Ethical Use & Safety")
    st.write("MindScan Pro utilizes data-driven insights to promote awareness. It is designed to complement, not replace, professional clinical care.")
    st.error("**If you are in immediate danger or experiencing a crisis, please contact your local emergency services or a crisis hotline immediately.**")

def show_assessment(model, scaler, le_gender, le_target, df):
    st.header("Advanced Assessment with explainability")
    if model is None:
        st.error("Model artifacts not found. Please ensure the project is initialized correctly.")
        return

    st.write("Complete the questionnaire to receive a predicted state and a detailed analysis of your results.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 28)
        gender = st.selectbox("Gender", le_gender.classes_)
        sleep_hours = st.slider("Average Sleep (Hours)", 4.0, 10.0, 7.0)
        physical_activity = st.slider("Activity (Days/Week)", 0, 7, 3)
    with col2:
        stress_level = st.select_slider("Daily Stress Level (1-10)", list(range(1, 11)), 5)
        social_interaction = st.select_slider("Social Interaction Quality (1-10)", list(range(1, 11)), 5)
        diet_quality = st.select_slider("Diet Quality (1-10)", list(range(1, 11)), 5)
        work_pressure = st.select_slider("Work/Academic Pressure (1-10)", list(range(1, 11)), 5)

    if st.button("Generate AI Assessment"):
        with st.spinner("Analyzing patterns..."):
            # Preprocess
            gender_enc = le_gender.transform([gender])[0]
            input_df = pd.DataFrame([[age, gender_enc, sleep_hours, physical_activity, stress_level, social_interaction, diet_quality, work_pressure]], 
                                    columns=['Age', 'Gender', 'Sleep_Hours', 'Physical_Activity', 'Stress_Level', 'Social_Interaction', 'Diet_Quality', 'Work_Pressure'])
            input_scaled = scaler.transform(input_df)
            
            prediction = model.predict(input_scaled)
            state = le_target.inverse_transform(prediction)[0]
            
            # UI Styling
            colors = {'No Depression': '#d1e7dd', 'Mild': '#fff3cd', 'Moderate': '#f8d7da', 'Severe': '#ea868f'}
            border_colors = {'No Depression': '#0f5132', 'Mild': '#856404', 'Moderate': '#842029', 'Severe': '#58151c'}
            
            st.markdown(f'''
            <div class="prediction-box" style="background-color: {colors[state]}; color: {border_colors[state]}; border: 2px solid {border_colors[state]};">
                Result: {state}
            </div>
            ''', unsafe_allow_html=True)
            
            # --- Explainable AI (SHAP) ---
            st.divider()
            st.markdown("### ⚖️ Explainable AI: Feature Importance")
            st.write(f"The chart below shows how each factor influenced the prediction of **{state}**.")
            
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                
                # Multi-class SHAP handling
                class_idx = prediction[0]
                
                # SHAP returns a list of arrays for multi-class (one array per class)
                # Each array has shape (n_samples, n_features)
                if isinstance(shap_values, list):
                    # Get SHAP values for the predicted class only
                    shap_for_class = shap_values[class_idx]
                    # Extract values for the first (and only) sample
                    shap_vals = shap_for_class[0]
                else:
                    # Binary classification or single output
                    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                
                # Ensure 1D array
                shap_vals = np.array(shap_vals).flatten()
                
                # If we still have too many values (e.g., 32 instead of 8), it means
                # SHAP returned values for all classes concatenated
                n_features = len(input_df.columns)
                if len(shap_vals) > n_features:
                    # Reshape and extract only the predicted class values
                    # The array is likely shaped as (n_classes * n_features,)
                    # We need to reshape it to (n_classes, n_features) and select the predicted class
                    n_classes = len(shap_vals) // n_features
                    shap_vals = shap_vals.reshape(n_classes, n_features)[class_idx]
                
                # Create feature importance plot
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_names = input_df.columns.tolist()
                
                # Debug: Check lengths
                st.write(f"DEBUG: Feature names count: {len(feature_names)}")
                st.write(f"DEBUG: SHAP values count: {len(shap_vals)}")
                st.write(f"DEBUG: Feature names: {feature_names}")
                st.write(f"DEBUG: SHAP values shape: {shap_vals.shape}")
                
                # Create a horizontal bar plot
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_vals
                })
                importance_df = importance_df.sort_values('SHAP Value', key=abs, ascending=True)
                
                colors = ['#d32f2f' if x < 0 else '#388e3c' for x in importance_df['SHAP Value']]
                ax.barh(importance_df['Feature'], importance_df['SHAP Value'], color=colors)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
                ax.set_title(f'Feature Contribution to "{state}" Prediction', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
                st.caption("**Interpretation**: Green bars (positive values) pushed the prediction toward this state, while red bars (negative values) pushed away from it. Longer bars indicate stronger influence.")
            except Exception as e:
                st.warning(f"Could not generate visual explanation: {e}")

def show_mindbot(api_key):
    st.header("💬 MindBot: Supportive Conversation")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(f'<div class="chat-bubble {role_class}"><b>{"You" if msg["role"] == "user" else "MindBot"}:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    prompt = st.chat_input("How are you feeling right now?")
    if prompt:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Sentiment Analysis
        sentiment = TextBlob(prompt).sentiment.polarity
        
        with st.spinner("MindBot is thinking..."):
            if api_key:
                try:
                    gen_model = genai.GenerativeModel('gemini-pro')
                    response = gen_model.generate_content(
                        f"You are MindBot, a compassionate mental health support AI assistant. "
                        f"The user says: '{prompt}'. Sentiment score: {sentiment:.2f} (negative=distressed, positive=happy). "
                        f"Instructions: Be empathetic, warm, and supportive. Respond in 2-3 sentences. "
                        f"If they ask for breathing exercises or tools, direct them to the 'Intervention Tools' page in the sidebar. "
                        f"If they mention specific emotions (sad, anxious, stressed), acknowledge and validate their feelings. "
                        f"Never diagnose. If very distressed (sentiment < -0.5), gently suggest professional help.")
                    reply = response.text
                except Exception as e:
                    # Show detailed error for debugging
                    error_msg = str(e)
                    st.error(f"⚠️ API Error: {error_msg}")
                    # Fallback to local responses on API error
                    reply = get_local_response(prompt, sentiment)
            else:
                # Enhanced local empathy logic
                reply = get_local_response(prompt, sentiment)
        
        # Add bot message to state
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

def get_local_response(prompt, sentiment):
    """Generate empathetic local responses based on sentiment and keywords"""
    prompt_lower = prompt.lower()
    
    # Check for breathing/intervention tool requests
    breathing_words = ['breathing', 'breath', 'breathing exercise', 'intervention', 'calm down', 'relax']
    if any(word in prompt_lower for word in breathing_words):
        return "Great idea! I have guided breathing exercises available for you. Please navigate to the **Intervention Tools** page using the sidebar on the left. There you'll find a Box Breathing exercise that can help you feel calmer and more centered."
    
    # Check for crisis keywords
    crisis_words = ['suicide', 'kill myself', 'end it all', 'want to die', 'hurt myself']
    if any(word in prompt_lower for word in crisis_words):
        return "I'm really concerned about what you're sharing. Please reach out to a crisis helpline immediately: Call 988 (USA) or text HOME to 741741. You don't have to face this alone - professional help is available 24/7."
    
    # Stress/anxiety responses
    stress_words = ['stress', 'anxious', 'anxiety', 'worried', 'overwhelmed', 'panic']
    if any(word in prompt_lower for word in stress_words):
        return "It sounds like you're carrying a lot right now. Stress and anxiety can feel overwhelming, but you're taking a positive step by expressing it. Have you tried the breathing exercise in our Intervention Tools? Sometimes a few deep breaths can help create space to think more clearly."
    
    # Sadness/depression responses
    sad_words = ['sad', 'depressed', 'down', 'hopeless', 'empty', 'lonely']
    if any(word in prompt_lower for word in sad_words):
        return "I hear you, and I'm sorry you're feeling this way. These feelings are valid, and it's okay to not be okay sometimes. Remember that reaching out - whether to friends, family, or a professional - can make a real difference. You're not alone in this."
    
    # Positive sentiment
    if sentiment > 0.3:
        return "That's wonderful to hear! It's great that you're experiencing positive moments. What's been helping you feel this way? Recognizing and celebrating the good times is an important part of mental wellness."
    
    # Negative sentiment
    elif sentiment < -0.3:
        return "Thank you for sharing how you're feeling. It takes courage to express difficult emotions. I'm here to listen. Would you like to talk more about what's troubling you, or would you prefer to try one of our intervention tools?"
    
    # Neutral sentiment
    else:
        return "I appreciate you opening up. Sometimes it helps just to express what's on our mind. I'm here to support you. Feel free to share more about what you're experiencing, or explore the other tools available in this app."

def show_intervention():
    st.header("🧘 Immediate Intervention Tools")
    
    tab1, tab2 = st.tabs(["Guided Breathing", "Interactive Resources"])
    
    with tab1:
        st.subheader("Box Breathing Exercise")
        st.write("Box breathing is a powerful stress-reliever. Focus on the circle and match your breath to its rhythm.")
        st.info("Grow (Inhale) -> Hold -> Shrink (Exhale) -> Hold")
        st.markdown("""
        <div class="breathing-container">
            <div class="circle">Breathe</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Perform for 3-5 cycles to notice a decrease in stress levels.")
    
    with tab2:
        st.subheader("Personalized Recommendations")
        # Logic can be expanded based on the *latest* assessment result if stored in session_state
        st.write("Based on common patterns, here are some helpful steps you can take today:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🌿 Mindfulness")
            st.write("- Try a 3-minute body scan meditation.")
            st.write("- Step outside for a 10-minute mindful walk.")
            st.markdown("#### ⚡ Energy Boost")
            st.write("- Drink a glass of water and stretch for 2 minutes.")
        with col2:
            st.markdown("#### ✍️ Reflection")
            st.write("- Write down 3 things you are grateful for today.")
            st.write("- List one small task you can complete in under 5 minutes.")
        
        st.divider()
        st.subheader("Emergency Resources")
        st.write("If you need to talk to someone right now:")
        st.write("- **National Suicide Prevention Lifeline**: 988 (USA)")
        st.write("- **Crisis Text Line**: Text HOME to 741741")
        st.write("- **Global Directory**: [Find a Helpline](https://findahelpline.com/)")

def show_insights(df):
    st.header("Data Analytics Dashboard")
    if df is not None:
        st.write(f"Analyzing {len(df)} samples from our behavioral dataset.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution of States")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='Depression_State', palette='magma')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        with col2:
            st.subheader("Stress vs Lifestyle Factors")
            feature = st.selectbox("Compare Stress with:", ['Sleep_Hours', 'Physical_Activity', 'Work_Pressure', 'Diet_Quality'])
            fig, ax = plt.subplots()
            sns.regplot(data=df, x=feature, y='Stress_Level', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            st.pyplot(fig)
    else:
        st.error("No data available to display insights.")

def show_performance():
    st.header("Model Evaluation Analytics")
    try:
        results = pd.read_csv(COMPARISON_PATH)
        st.write("Comparative accuracy across multiple algorithms:")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=results, x='Model', y='Accuracy', palette='coolwarm')
        plt.ylim(0, 1)
        st.pyplot(fig)
        
        st.table(results)
    except:
        st.error("Model performance data not found.")

if __name__ == "__main__":
    main()
