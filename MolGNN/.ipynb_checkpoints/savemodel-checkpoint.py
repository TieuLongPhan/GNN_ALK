from sklearn.metrics import f1_score, average_precision_score
import torch
import torch_geometric
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class SaveModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self):
        self._valid_loss = None
        
    def __call__(self,epochs, model, optimizer, criterion):
        print("Saving...")
        torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'Model/GNN_model.pth')