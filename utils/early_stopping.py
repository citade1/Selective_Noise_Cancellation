class EarlyStopping:
    
    def __init__(self, patience=2, min_delta=0.0):
        """
        Args:
            patience(int): Number of epochs to wait after no improvement
            min_delta(float): Minimum change to quality as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True