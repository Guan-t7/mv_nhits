from sklearn.preprocessing import StandardScaler
import torch
import torch as t

Tensor = torch.FloatTensor


def inv_transform(scaler: StandardScaler, Y: Tensor):
    '''pytorch version and takes (n_features, n_samples)
    '''
    if scaler.with_mean == False and scaler.with_std == False:
        return Y
    
    dev = Y.device
    scale = t.from_numpy(scaler.scale_).to(dev).unsqueeze(-1)
    mean = t.from_numpy(scaler.mean_).to(dev).unsqueeze(-1)
    Y = Y * scale + mean

    return Y


def MSE(y_hat, y, mask=None):
    if mask is None:
        mask = t.ones_like(y_hat)

    mse = (y - y_hat)**2
    mse = mask * mse
    mse = t.mean(mse)
    return mse


def MAELoss(y_hat, y, mask=None):
    if mask is None:
        mask = t.ones_like(y_hat)

    mae = t.abs(y - y_hat) * mask
    mae = t.mean(mae)
    return mae


def xape3(arr_pred: Tensor, arr_true: Tensor):
    true_abs = arr_true.abs()
    wape = (arr_pred-arr_true).abs().sum() / true_abs.sum()
    nz = torch.where(true_abs > 0)
    Pnz = arr_pred[nz]
    Anz = arr_true[nz]
    mape = ((Pnz - Anz).abs() / (Anz.abs() + 1e-5)).mean()
    smape = (2 * (Pnz - Anz).abs() / (Pnz.abs() + Anz.abs() + 1e-5)).mean()
    return wape, mape, smape


def rrse(arr_pred: Tensor, arr_true: Tensor):
    total_se = ((arr_true - arr_pred) ** 2).sum()
    total_se_g = ((arr_true - arr_true.mean()) ** 2).sum()
    rse = (total_se / total_se_g).sqrt()
    return rse


def emp_corr(arr_pred: Tensor, arr_true: Tensor):
    '''input of shape (N, T)'''
    sigma_p = arr_pred.std(1, keepdim=True)
    sigma_g = arr_true.std(1, keepdim=True)
    mean_p = arr_pred.mean(1, keepdim=True)
    mean_g = arr_true.mean(1, keepdim=True)
    index = (sigma_g != 0)
    cov = ((arr_pred - mean_p) * (arr_true - mean_g)).mean(1, keepdim=True)
    cor = cov / (sigma_p*sigma_g)
    
    cor = cor[index].mean()
    return cor


def crps_ensemble(observations: Tensor, forecasts: Tensor) -> Tensor:
    '''pytorch version of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187
    
    This implementation is based on the identity:

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.
    '''
    assert observations.ndim == forecasts.ndim - 1
    assert observations.shape == forecasts.shape[1:]  # ensemble ~ first axis

    score = (forecasts - observations).abs().mean(axis=0)
    # insert new axes so forecasts_diff expands with the array broadcasting
    forecasts_diff = (torch.unsqueeze(forecasts, 0) -
                      torch.unsqueeze(forecasts, 1))
    score += -0.5 * forecasts_diff.abs().mean(axis=(0,1))
    return score