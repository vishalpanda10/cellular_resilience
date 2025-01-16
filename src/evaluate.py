import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        X_test_t = torch.Tensor(X_test[:, np.newaxis]).to("cuda")
        y_test_t = torch.Tensor(y_test[:, np.newaxis]).to("cuda")
        
        y_pred = model(X_test_t).cpu().numpy()
        y_true = y_test_t.cpu().numpy()
        y_pred = y_pred.squeeze(axis=1)
        y_true = y_true.squeeze(axis=1)
        
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    #nrmse = rmse / (np.max(y_true) - np.min(y_true))
    r2 = r2_score(y_true, y_pred)
    
    return {"RMSE": rmse, "R2": r2}