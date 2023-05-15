from sklearn.metrics import f1_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

        
def historical_plot(data_train, data_val):

    sns.set()

    plt.figure(figsize =(10,8))
    plt.plot(data_train, label='Training')
    plt.plot(data_val, label='Validation')
    plt.legend(frameon=False)
    return plt
