import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du risque de Congenital heart disease",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un meilleur design
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<p class="main-header">🫀 Prédiction du Risque Cardiaque (CHD)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse basée sur Machine Learning • Pipeline Scikit-learn</p>', unsafe_allow_html=True)

# Sidebar avec informations
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
    st.title("ℹ️ À propos")
    st.markdown("""
    ### Technologie
    - **Framework**: Streamlit
    - **Modèle**: Régression Logistique
    - **Prétraitement**: ACP + Pipeline
    - **Dataset**: CHD.csv
    
    ### Variables considérées
    - Âge du patient
    - Pression systolique (SBP)
    - LDL (cholestérol)
    - Adiposité
    - Obésité (IMC)
    - Antécédents familiaux
    
    ### ⚠️ Avertissement
    Cette application est à **but pédagogique uniquement** 
    et ne remplace en aucun cas un diagnostic médical professionnel.
    """)
    
    st.divider()
    st.caption("Développé avec ❤️ pour l'éducation en santé")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Le fichier Model.pkl est introuvable. Assurez-vous qu'il est dans le même répertoire.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        st.stop()

model = load_model()

# Affichage du message de succès pour le chargement
st.success("✅ Modèle chargé avec succès")

# Onglets pour organiser le contenu
tab1, tab2 = st.tabs(["📋 Prédiction", "📊 Statistiques"])

with tab1:
    st.subheader("🩺 Saisir les informations du patient")
    
    # Formulaire amélioré
    with st.form("chd_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("👤 Âge", min_value=10, max_value=100, value=50, help="Âge du patient en années")
            sbp = st.number_input("💉 Pression systolique (mmHg)", min_value=80.0, max_value=250.0, value=140.0, 
                                 help="Pression artérielle systolique")
            ldl = st.number_input("🧪 LDL - Cholestérol (mmol/L)", min_value=0.0, max_value=15.0, value=4.0, step=0.1,
                                 help="Niveau de mauvais cholestérol")
        
        with col2:
            adiposity = st.number_input("📏 Adiposité", min_value=0.0, max_value=60.0, value=25.0, step=0.5,
                                       help="Pourcentage de graisse corporelle")
            obesity = st.number_input("⚖️ Obésité (IMC)", min_value=10.0, max_value=60.0, value=26.0, step=0.5,
                                     help="Indice de Masse Corporelle")
            famhist = st.selectbox("👨‍👩‍👧‍👦 Antécédents familiaux", 
                                  ["Absent", "Present"],
                                  help="Présence de maladies cardiaques dans la famille")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button("🔍 Analyser le risque", use_container_width=True)
    
    # Prédiction
    if submitted:
        # Préparation des données
        input_data = {
            "sbp": sbp,
            "ldl": ldl,
            "adiposity": adiposity,
            "obesity": obesity,
            "age": age,
            "famhist": famhist
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Affichage des données
        with st.expander("📄 Voir les données saisies", expanded=False):
            st.dataframe(input_df, use_container_width=True)
        
        # Prédiction
        with st.spinner("🔄 Analyse en cours..."):
            proba_chd = model.predict_proba(input_df)[0, 1]
            pred_chd = model.predict(input_df)[0]
        
        st.divider()
        st.subheader("🎯 Résultat de l'analyse")
        
        # Jauge de probabilité avec Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = proba_chd * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilité de CHD (%)", 'font': {'size': 24}},
            delta = {'reference': 50, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#90EE90'},
                    {'range': [30, 60], 'color': '#FFD700'},
                    {'range': [60, 100], 'color': '#FF6B6B'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': proba_chd * 100
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Interprétation
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Probabilité CHD", f"{proba_chd:.1%}", 
                     delta=f"{(proba_chd - 0.5):.1%}" if proba_chd > 0.5 else None,
                     delta_color="inverse")
        
        with col_res2:
            st.metric("Prédiction", 
                     "Risque Élevé ⚠️" if pred_chd == 1 else "Risque Faible ✅",
                     delta=None)
        
        # Message d'alerte
        if pred_chd == 1:
            st.error("""
            🚨 **RISQUE ÉLEVÉ DÉTECTÉ**
            
            Le modèle indique une forte probabilité de maladie cardiaque coronarienne. 
            **Recommandations** :
            - Consulter immédiatement un cardiologue
            - Réaliser des examens complémentaires
            - Adopter un mode de vie sain
            """)
        else:
            st.success("""
            ✅ **RISQUE FAIBLE DÉTECTÉ**
            
            Le modèle indique une faible probabilité de CHD pour le moment.
            **Recommandations** :
            - Maintenir un mode de vie sain
            - Surveillance médicale régulière
            - Contrôle des facteurs de risque
            """)
        
        # Facteurs de risque identifiés
        st.divider()
        st.subheader("⚠️ Analyse des facteurs de risque")
        
        risk_factors = []
        if age > 60:
            risk_factors.append("• Âge supérieur à 60 ans")
        if sbp > 140:
            risk_factors.append("• Hypertension artérielle (SBP > 140 mmHg)")
        if ldl > 4.5:
            risk_factors.append("• Taux de LDL élevé (> 4.5 mmol/L)")
        if obesity > 30:
            risk_factors.append("• Obésité (IMC > 30)")
        if famhist == "Present":
            risk_factors.append("• Antécédents familiaux de maladies cardiaques")
        
        if risk_factors:
            st.warning("**Facteurs de risque identifiés :**\n" + "\n".join(risk_factors))
        else:
            st.info("Aucun facteur de risque majeur identifié dans les données saisies.")

with tab2:
    st.subheader("📊 Statistiques et informations")
    
    st.markdown("""
    ### 📈 Valeurs de référence
    
    | Paramètre | Valeur normale | Valeur à risque |
    |-----------|----------------|-----------------|
    | SBP | < 120 mmHg | > 140 mmHg |
    | LDL | < 3.0 mmol/L | > 4.5 mmol/L |
    | IMC | 18.5-24.9 | > 30 |
    | Adiposité | < 25% | > 30% |
    
    ### 🔬 Performance du modèle
    
    Le modèle utilise une régression logistique avec réduction de dimensionnalité (ACP) 
    pour prédire le risque de maladie cardiaque coronarienne basée sur 6 variables cliniques.
    
    **Pipeline complet :**
    1. Prétraitement des données (normalisation, encodage)
    2. Analyse en Composantes Principales (ACP)
    3. Régression Logistique
    
    ### 💡 Conseils de prévention
    
    - 🏃 Exercice physique régulier 
    - 🥗 Alimentation équilibrée 
    - 🚭 Arrêt du tabac
    - 😌 Gestion du stress et regler le system nerveux
    - 💊 Suivi médical régulier
    """)

# Footer
st.divider()
st.caption("© 2024 | Application de prédiction CHD | Données à usage pédagogique uniquement")