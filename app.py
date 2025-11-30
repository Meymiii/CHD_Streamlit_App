import streamlit as st
import pandas as pd
import joblib

# Configuration de base de la page
st.set_page_config(page_title="Prédiction CHD", page_icon="🫀")

# Titre principal
st.title("🫀 Prédiction du Risque Cardiaque dévelopée par Maryem")
st.write("Application développée par MARYEM")

# Barre latérale avec informations
st.sidebar.title("ℹ️ Informations")
st.sidebar.write("""
Cette application prédit le risque de maladie cardiaque coronarienne (CHD).

**Variables utilisées:**
- Âge
- Pression artérielle (SBP)
- Cholestérol LDL
- Adiposité
- IMC (Obésité)
- Antécédents familiaux
""")
st.sidebar.warning("⚠️ Application à but pédagogique uniquement")

# Charger le modèle ML
@st.cache_resource
def load_model():
    """Charge le modèle sauvegardé une seule fois"""
    try:
        return joblib.load("Model.pkl")
    except:
        st.error("❌ Fichier Model.pkl introuvable")
        st.stop()

model = load_model()
st.success("✅ Modèle chargé")

# Section de saisie des données
st.subheader("📋 Entrez les informations du patient")

# Créer deux colonnes pour organiser les champs
col1, col2 = st.columns(2)

# Colonne 1: Champs de saisie
with col1:
    age = st.slider("Âge", 10, 100, 50)
    sbp = st.number_input("Pression systolique (mmHg)", 80.0, 250.0, 140.0)
    ldl = st.number_input("LDL Cholestérol (mmol/L)", 0.0, 15.0, 4.0)

# Colonne 2: Champs de saisie
with col2:
    adiposity = st.number_input("Adiposité", 0.0, 60.0, 25.0)
    obesity = st.number_input("IMC (Obésité)", 10.0, 60.0, 26.0)
    famhist = st.selectbox("Antécédents familiaux", ["Absent", "Present"])

# Bouton pour lancer la prédiction
if st.button("🔍 Analyser le risque", use_container_width=True):
    
    # Créer un dictionnaire avec les données saisies
    data = {
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }
    
    # Convertir en DataFrame (format attendu par le modèle)
    input_df = pd.DataFrame([data])
    
    # Faire la prédiction
    probabilite = model.predict_proba(input_df)[0, 1]  # Probabilité de CHD
    prediction = model.predict(input_df)[0]  # 0 ou 1
    
    # Afficher les résultats
    st.divider()
    st.subheader("🎯 Résultats")
    
    # Afficher la probabilité
    st.metric("Probabilité de CHD", f"{probabilite:.1%}")
    
    # Afficher l'interprétation
    if prediction == 1:
        st.error("""
        🚨 **RISQUE ÉLEVÉ**
        
        Le modèle détecte un risque élevé de maladie cardiaque.
        Consultez un médecin pour des examens approfondis.
        """)
    else:
        st.success("""
        ✅ **RISQUE FAIBLE**
        
        Le modèle détecte un risque faible.
        Maintenez un mode de vie sain et faites des contrôles réguliers.
        """)
    
    # Identifier les facteurs de risque
    st.subheader("⚠️ Facteurs de risque détectés")
    
    facteurs = []
    if age > 60:
        facteurs.append("• Âge > 60 ans")
    if sbp > 140:
        facteurs.append("• Hypertension (SBP > 140)")
    if ldl > 4.5:
        facteurs.append("• LDL élevé (> 4.5)")
    if obesity > 30:
        facteurs.append("• Obésité (IMC > 30)")
    if famhist == "Present":
        facteurs.append("• Antécédents familiaux")
    
    if facteurs:
        st.warning("\n".join(facteurs))
    else:
        st.info("Aucun facteur de risque majeur identifié")

# Footer
st.divider()
st.caption("© 2024 | Développé par [Votre Nom] | Usage pédagogique uniquement")