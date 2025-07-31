import numpy as np

class EarlyStopper:
    '''
    Basic implementation of early stopping.
    '''

    def __init__(self, patience=1, min_delta=0):
        '''
        Params
        :patience (int) -> iterations of patience.
        :min_delta (int) -> minimum value accepted as model improvement.
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def calculate(self, validation_loss):
        '''
        Method to control early stopping.
        '''
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False