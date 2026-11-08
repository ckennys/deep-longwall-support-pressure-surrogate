"""Frozen field-combination rule extracted from the historical source."""
import numpy as np

def combine(predictions, rule, steps):
    result = np.zeros_like(predictions[0], dtype=np.float32)
    if rule['mode'] == 'step':
        for step in np.unique(steps):
            selected = steps == step
            weights = rule['weights'][str(int(step))]
            result[selected] = sum(float(weight) * pred[selected] for weight, pred in zip(weights, predictions))
    else:
        result = sum(float(weight) * pred for weight, pred in zip(rule['weights'], predictions)).astype(np.float32)
    return result
