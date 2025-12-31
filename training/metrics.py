from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
class MetricsCalculator:
    """Calculate classification metrics."""
    
    @staticmethod
    def compute_metrics(preds, targets, threshold=0.5):
        """Compute all metrics."""
        preds_binary = (preds > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds_binary),
            'f1': f1_score(targets, preds_binary),
            'auc': roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.0,
            'sensitivity': MetricsCalculator._sensitivity(targets, preds_binary),
            'specificity': MetricsCalculator._specificity(targets, preds_binary)
        }
        
        return metrics
    
    @staticmethod
    def _sensitivity(targets, preds):
        """Recall for positive class."""
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def _specificity(targets, preds):
        """Recall for negative class."""
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
