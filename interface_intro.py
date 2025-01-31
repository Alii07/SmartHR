import streamlit as st
import base64

def intro_interface():
    # Masquer la barre latérale complètement
    st.markdown("""
        <style>
        .main-container {
            padding: 2rem;
            text-align: center;
        }
        
        .hero-title {
            font-size: 5em !important;
            color: #386161;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .hero-subtitle {
            font-size: 2.2em !important;
            color: #8ac447;
            margin-bottom: 2rem;
            font-style: italic;
        }
        
        .feature-container {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin: 2rem auto;
            max-width: 1200px;
        }
        
        .feature-card {
            flex: 1;
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #8ac447;
            height: 100%;
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            min-height: 400px; /* Ensure equal height */
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            border: 2px solid #386161;
            background-color: rgba(138, 196, 71, 0.05);
        }
        
        .feature-icon {
            font-size: 3em;
            color: #386161;
            margin-bottom: 1rem;
        }
        
        .feature-title {
            font-size: 1.8em;
            color: #386161;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .feature-description {
            color: #666666;
            font-size: 1.1em;
            line-height: 1.6;
            flex-grow: 1;
        }
        
        a {
            text-decoration: none !important;
        }

        /* Masquer le style par défaut des boutons Streamlit */
        .stButton > button.feature-button {
            width: 100%;
            height: 100%;
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        
        .stButton > button.feature-button:hover {
            background-color: transparent !important;
            border: none !important;
        }
        
        .icon-text {
            font-size: 3em;
            color: #386161;
            margin-bottom: 1rem;
        }
        
        .title-text {
            font-size: 1.8em;
            color: #386161;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .description-text {
            color: #666666;
            font-size: 1.1em;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)

    

    # Section héros
    st.markdown("""
        <div class="main-container">
            <h1 class="hero-title">SmartHR</h1>
            <p class="hero-subtitle">L'intelligence artificielle au service de votre paie</p>
            <p style="font-size: 1.2em; color: #666666; max-width: 800px; margin: 0 auto 3rem auto;">
                Simplifiez la gestion de vos bulletins de paie grâce à notre solution innovante 
                qui combine expertise RH et intelligence artificielle.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Section des fonctionnalités avec cartes cliquables
    col1, col2, col3 = st.columns(3)

    with col1:
        extraction_container = st.container()
        extraction_container.markdown("""
            <div class="feature-card" onclick="handleClick('Extraction des données')">
                <div class="feature-icon">📤</div>
                <div class="feature-title">Extraction</div>
                <p class="feature-description">
                    Transformez vos bulletins de paie PDF en données structurées et exploitables en quelques clics.
                    Extraction intelligente et automatisée.
                </p>
            </div>
            """, unsafe_allow_html=True)
        if extraction_container.button("Extraction des données", key="extraction_btn"):
            st.session_state["menu"] = "Extraction des données"
            st.rerun()

    with col2:
        modification_container = st.container()
        modification_container.markdown("""
            <div class="feature-card" onclick="handleClick('Modification cotisation employée')">
                <div class="feature-icon">✏️</div>
                <div class="feature-title">Modification</div>
                <p class="feature-description">
                    Interface intuitive pour modifier et ajuster les cotisations par employé.
                    Contrôle en temps réel et validation instantanée des modifications.
                </p>
            </div>
            """, unsafe_allow_html=True)
        if modification_container.button("Modification des cotisations", key="modification_btn"):
            st.session_state["menu"] = "Modification cotisation employée"
            st.rerun()

    with col3:
        detection_container = st.container()
        detection_container.markdown("""
            <div class="feature-card" onclick="handleClick('Détection d\\'anomalies')">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Détection</div>
                <p class="feature-description">
                    Détection automatique des anomalies dans vos cotisations grâce à notre 
                    IA spécialisée. Rapports détaillés et recommandations personnalisées.
                </p>
            </div>
            """, unsafe_allow_html=True)
        if detection_container.button("Détection d'anomalies", key="detection_btn"):
            st.session_state["menu"] = "Détection d'anomalies"
            st.rerun()

    # Ajouter le JavaScript pour gérer les clics
    st.markdown("""
        <script>
        function handleClick(menuOption) {
            const buttons = {
                'Extraction des données': 'extraction_btn',
                'Modification cotisation employée': 'modification_btn',
                'Détection d\\'anomalies': 'detection_btn'
            };
            const buttonId = buttons[menuOption];
            if (buttonId) {
                document.getElementById(buttonId).click();
            }
        }
        </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    intro_interface()
