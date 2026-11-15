import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. TEXT PREPROCESSING & INDICATOR RULES
# ==========================================

def clean_text(text: str) -> str:
    """Normalizes whitespace, lowercases, removes HTML tags, URLs, and punctuation."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http[s]?://\S+', ' http_url ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_addr ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_security_indicators(raw_text: str) -> list[dict]:
    """Identifies heuristic phishing indicators directly from email content."""
    indicators = []
    text_lower = raw_text.lower()
    
    # Urgency & coercion
    urgency_terms = ['urgent', 'immediately', 'suspended', 'unauthorized', 'action required', 'terminate', 'within 24 hours', 'expires']
    matched_urgency = [w for w in urgency_terms if w in text_lower]
    if matched_urgency:
        indicators.append({
            "category": "Urgency / Intimidation",
            "detail": f"High-pressure terminology found: {', '.join(matched_urgency)}",
            "risk": "High"
        })
        
    # Financial & credential harvesting
    cred_terms = ['verify password', 'routing number', 'bank account', 'reset credentials', 'gift card', 'wire transfer', 'ssn']
    matched_creds = [w for w in cred_terms if w in text_lower]
    if matched_creds:
        indicators.append({
            "category": "Credential / Financial Harvesting",
            "detail": f"Requests sensitive actions or credentials: {', '.join(matched_creds)}",
            "risk": "Critical"
        })

    # Generic salutations
    if re.search(r'\b(dear customer|dear user|dear account holder|valued client|dear employee)\b', text_lower):
        indicators.append({
            "category": "Impersonal Salutation",
            "detail": "Generic greeting often used in bulk credential-harvesting campaigns.",
            "risk": "Medium"
        })
        
    # Suspicious link structures
    raw_links = re.findall(r'http[s]?://\S+', raw_text)
    ip_links = [l for l in raw_links if re.search(r'//\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', l)]
    if ip_links:
        indicators.append({
            "category": "Direct IP Address Link",
            "detail": f"Direct numerical IP address instead of domain hostname: {ip_links[0]}",
            "risk": "Critical"
        })
    elif raw_links:
        indicators.append({
            "category": "Embedded Links",
            "detail": f"Detected {len(raw_links)} embedded link(s). Validate domain authenticity.",
            "risk": "Low"
        })

    return indicators

# ==========================================
# 2. DATASET & MODEL PIPELINE (CACHED)
# ==========================================

@st.cache_resource(show_spinner="Training classification pipeline...")
def train_and_cache_model():
    """Generates a starter dataset (or loads emails.csv if present) and trains the ML pipeline."""
    csv_path = "emails.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        col_mapping = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=col_mapping, inplace=True)
        if 'label' in df.columns and df['label'].dtype == object:
            df['label'] = df['label'].astype(str).str.strip().str.lower()
            df['label'] = df['label'].map({'phishing': 1, 'spam': 1, '1': 1, 'ham': 0, 'legitimate': 0, '0': 0})
    else:
        # Balanced dataset covering operational, financial, and spoofed contexts
        synthetic_corpus = {
            "text": [
                "URGENT: Your corporate Office365 password expires in 2 hours. Retain access at http://192.168.1.1/login",
                "Action Required: Unrecognized sign-in detected from IP 45.22.1.0. Verify your credentials immediately.",
                "Dear Valued Customer, please provide your SSN and routing number to release held wire funds.",
                "Payroll notice: Click here to review your salary adjustment and confirm direct deposit details.",
                "Your account has been suspended due to policy violations. Reset your password within 24 hours.",
                "Hi Sarah, could you review the attached Q3 financial slides before tomorrow's 10 AM standup?",
                "Monthly IT notice: Our scheduled multi-factor authentication upgrade takes place this Saturday.",
                "Lunch plans today? Let me know if you'd like to check out the new place down the street.",
                "Attached is the vendor invoice for the cloud server renewal. Net 30 payment terms apply.",
                "Quarterly company all-hands meeting scheduled for next Thursday at 3 PM in Conference Room B."
            ] * 25,
            "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 25
        }
        df = pd.DataFrame(synthetic_corpus)

    df = df.dropna(subset=['text', 'label'])
    df['clean_text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Compute evaluation figures
    y_pred = pipeline.predict(X_test)
    rep = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"], output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phish"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test Set Confusion Matrix", fontsize=10)
    plt.tight_layout()

    return pipeline, rep, fig

# ==========================================
# 3. STREAMLIT APPLICATION INTERFACE
# ==========================================

st.set_page_config(page_title="Phishing Email Screener", layout="wide", page_icon="🛡️")

pipeline, eval_report, cm_fig = train_and_cache_model()

st.title("🛡️ Enterprise Phishing Email Detection & Security Analysis")
st.markdown("Automated email threat screening combining **TF-IDF + Logistic Regression** with heuristic behavioral indicators.")

# Pre-populated test fixtures
TEST_SUITE = {
    "Select a pre-loaded test case...": "",
    "Phishing - Credential Harvesting": (
        "Subject: Urgent: Password Expiry Notification\n\n"
        "Dear Employee,\nYour enterprise single-sign-on credentials will expire in 2 hours. "
        "Log in immediately to http://192.168.1.105/auth to avoid account suspension."
    ),
    "Phishing - Financial Wire Fraud": (
        "Subject: Immediate Wire Transfer Request\n\n"
        "Dear Account Holder,\nPlease reply immediately with your routing number and bank account details "
        "to verify and release the incoming $24,500 wire transfer."
    ),
    "Legitimate - DevOps Notification": (
        "Subject: Scheduled Network Maintenance\n\n"
        "Hi Team,\nThe internal VPN and staging cluster will be undergoing routine firmware updates this Saturday between "
        "2:00 AM and 4:00 AM UTC. No action is required on your part."
    ),
    "Ambiguous - Urgency Marketing": (
        "Subject: Final Hours: 50% Off Cloud Training Passes\n\n"
        "Don't miss out! Our annual developer conference registration closes tonight at midnight. "
        "Click the link to reserve your seat before prices increase."
    )
}

tab_scanner, tab_eval, tab_limits = st.tabs(["📧 Threat Scanner", "📊 Model Evaluation & Metrics", "⚠️ Enterprise Limitations"])

with tab_scanner:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Email Ingestion")
        
        selection = st.selectbox("Quick-load Test Payload:", list(TEST_SUITE.keys()))
        default_text = TEST_SUITE[selection] if selection != "Select a pre-loaded test case..." else ""
        
        raw_email = st.text_area("Paste Raw Email Body & Headers:", value=default_text, height=220)
        threshold = st.slider("Phishing Detection Probability Threshold", 0.1, 0.9, 0.5, 0.05)
        scan_btn = st.button("Analyze Threat Profile", type="primary")

    with col_output:
        st.subheader("2. Security Verdict")
        if scan_btn and raw_email.strip():
            cleaned = clean_text(raw_email)
            probs = pipeline.predict_proba([cleaned])[0]
            phish_prob = probs[1]
            is_phishing = phish_prob >= threshold

            # Classification Card
            if is_phishing:
                st.error(f"🚨 **VERDICT: POTENTIAL PHISHING / THREAT DETECTED** (Score: {phish_prob*100:.1f}%)")
            else:
                st.success(f"✅ **VERDICT: LIKELY LEGITIMATE** (Phishing Risk: {phish_prob*100:.1f}%)")

            st.progress(float(phish_prob))

            # Explainable Indicators
            st.markdown("#### Detected Indicators")
            indicators = extract_security_indicators(raw_email)
            if indicators:
                for ind in indicators:
                    tag_color = "red" if ind['risk'] in ["High", "Critical"] else "orange" if ind['risk'] == "Medium" else "blue"
                    st.markdown(f"- **:{tag_color}[[{ind['risk']} Risk]] {ind['category']}:** {ind['detail']}")
            else:
                st.info("No high-risk keyword triggers, direct IP links, or generic salutations observed.")
        elif scan_btn:
            st.warning("Please supply or select an email message first.")

with tab_eval:
    st.subheader("Performance & Validation")
    ecol1, ecol2 = st.columns([1, 1])
    with ecol1:
        st.pyplot(cm_fig)
    with ecol2:
        st.text("Model Classification Metrics:\n\n" + eval_report)

with tab_limits:
    st.subheader("Enterprise Security Limitations")
    st.markdown("""
    * **Adversarial Typography (Homoglyphs & Zero-Width Spaces):** Attackers bypass TF-IDF by replacing Latin letters with visually identical Cyrillic characters or invisible unicode separators.
    * **Ephemeral & Zero-Day Infrastructure:** A text classifier cannot resolve DNS records or domain age. High-assurance systems must check domain registration records (e.g., flags for domains registered `< 30 days`).
    * **Non-Text Media (Quishing & Image Attacks):** Attacks contained entirely inside PDF attachments, QR codes, or image renders contain no selectable text tokens and require integrated OCR pre-processing.
    """)
 