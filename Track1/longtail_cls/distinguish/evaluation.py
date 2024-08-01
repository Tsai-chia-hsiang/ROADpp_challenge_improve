from typing import Any, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

def classification(pred:np.ndarray, gth:np.ndarray, to_pydefault_type:bool=False, cm:Optional[Literal["np","pd"]]=None) -> dict[str, Any]:
    
    f1 = f1_score(gth, pred, average=None)
    metrics = {
        'precision':precision_score(gth, pred, average=None),
        'recall':recall_score(gth, pred, average=None),
        'f1':f1,
        'macro f1': np.mean(f1)
    }
    if to_pydefault_type:
        for k in ["precision", "recall", "f1"]:
            metrics[k] = metrics[k].tolist()
        metrics['macro f1'] = float(metrics['macro f1'])
    if cm is None:
        return metrics
    
    cm_df = confusion_matrix(y_true=gth, y_pred=pred, normalize='pred')    
    match cm:
        case "pd":
            cm_df = pd.DataFrame(
                cm_df, 
                columns=[f'Pred_{i}' for i in np.unique(gth)]
            )
            cm_df.insert(0, 'Actual', np.unique(gth))
    
        case "np":
            if to_pydefault_type:
                cm_df = cm_df.tolist()
    
    metrics['confusion matrix'] = cm_df
    return metrics