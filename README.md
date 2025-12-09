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

# Détection plus stricte
python test_with_unknown.py --checkpoint runs\20251209-160344_run\checkpoint.pt    --strictness high

# Détection très stricte
python test_with_unknown.py --checkpoint runs\20251209-160344_run\checkpoint.pt    --strictness very_high

# Méthode multi-critères au lieu de score combiné
python test_with_unknown.py --checkpoint runs\20251209-160344_run\checkpoint.pt    --method multi_criteria --strictness high
```