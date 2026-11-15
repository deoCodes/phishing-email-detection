import io, sys, contextlib

import app  # noqa: F401  (executes module top-level: trains model, builds figure)

# Module-level top already ran train_and_cache_model(); grab pieces again fresh
pipeline, rep, fig = app.train_and_cache_model()
print("=== Classification report ===")
print(rep)

samples = {
    "phish": "Dear Employee, your password expires in 2 hours. Login at http://192.168.1.105/auth immediately.",
    "legit": "Hi Sarah, lunch tomorrow? The quarterly meeting moved to Thursday at 3 PM.",
}
for name, txt in samples.items():
    cleaned = app.clean_text(txt)
    prob = pipeline.predict_proba([cleaned])[0][1]
    inds = app.extract_security_indicators(txt)
    print(f"{name}: phishing_prob={prob:.3f}, indicators={[i['category'] for i in inds]}")
print("VERIFY_OK")
