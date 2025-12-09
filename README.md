# Deep Learning for Signal Processing – Anomaly detection

**Authors** : Yanis Allal and Marine Vieillard
---

**Run** : 
```bash
python train.py  
```

**Tensorboard** :
```bash
tensorboard --logdir=runs
```

**Test sans anomalies** : 
```bash
python test.py --checkpoint runs\20251209-160344_run\checkpoint.pt --test Data\test.hdf5  
```

**Test anomalies** : 

```bash
# Configuration par défaut
python test_with_unknown.py --checkpoint runs\20251209-160344_run\checkpoint.pt   

# Exemple 
python test_with_unknown.py --checkpoint runs\20251209-160344_run\checkpoint.pt    --method knn --k 5 --strictness extreme

