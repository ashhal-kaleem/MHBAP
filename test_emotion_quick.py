import sys; sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.INFO)
from ml.models.emotion_recognizer import EmotionRecognizer
import numpy as np

rec = EmotionRecognizer(device='cpu', auto_download=False)
print('Ready:', rec.is_ready)

for color, name in [([200,180,160], 'skin'), ([0,0,0], 'black'), ([255,255,255], 'white')]:
    frame = np.full((224,224,3), color, dtype=np.uint8)
    r = rec.predict(frame)
    label = r['emotion_label']
    v = r['valence']
    a = r['arousal']
    probs = list(r['emotion_scores'].values())
    max_p = max(probs)
    entropy = -sum(p*np.log(p+1e-8) for p in probs)
    print(f"{name}: {label} | V={v:.3f} A={a:.3f} | max_p={max_p:.3f} entropy={entropy:.3f}")
