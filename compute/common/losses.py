import torch 

class MQLoss():
    """
    Multi-quantile loss function for time series forecasting.
    This loss function computes the quantile loss for multiple quantiles,
    allowing for weighted contributions from different time horizons.
    Args:
        quantiles (list[float]): List of quantiles to compute the loss for.
        gamma_h (list[float], optional): Weights for different time horizons. Defaults to None, which means equal weighting.
    """

    def __init__(self,
                 quantiles:list[float],
                 gamma_h: float = 0.0
                 ) -> None:

        self.quantiles = quantiles
    
        self.output_size = len(quantiles)
        self.output_cols = [f'y_hat_{int(q*100):02d}' for q in quantiles]
        
        self.gamma_h = gamma_h

    def __call__(self,
                 y:torch.Tensor,
                 y_hat:torch.Tensor
                ) -> torch.Tensor:
        
        h = y_hat.size(1)
        quantiles = torch.tensor(self.quantiles, device=y.device).unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(-1).expand(-1,-1,self.output_size)

        nan_mask = torch.isnan(y)
        y = torch.nan_to_num(y, nan=0.0)
        
        error = y_hat - y
        loss = torch.maximum(quantiles * -error, (1 - quantiles) * error) 
        
        if self.gamma_h != 0.0:
            steps = torch.linspace(0, 1, steps=y_hat.size(1), device=y_hat.device)
            h_weights = torch.exp(self.gamma_h * steps)
            h_weights = h_weights / h_weights.mean()
            loss = loss * h_weights.unsqueeze(0).unsqueeze(-1)
        
        total_loss = loss[~nan_mask].mean()

        return total_loss