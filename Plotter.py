from Libs import *

class Plotter: 
    def __init__(self, save_loc=None, model_name=None):
        self.save_loc = save_loc
        self.model_name = model_name
        
    def create_file_name(self, figure_name):
        if self.save_loc is None: 
            if self.model_name is None: return f"{figure_name}.jpg"
            else: return f"{self.model_name}_{figure_name}.jpg"
        else:
            if self.model_name is None: return f"{self.save_loc}/{figure_name}.jpg"
            else: return f"{self.save_loc}/{self.model_name}_{figure_name}.jpg"

    def plot_learnining_curves(self, losses, accuracies, figure_name='Learning Curves'):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        t = f.suptitle(figure_name, fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)

        epoch_list = range(len(losses["train"]))
        
        ax1.plot(epoch_list, accuracies['train'], label='Train Accuracy')
        ax1.plot(epoch_list, accuracies['validation'], label='Validation Accuracy')
        ax1.set_ylabel('Accuracy Value')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Accuracy')
        l1 = ax1.legend(loc="best")

        ax2.plot(epoch_list, losses['train'], label='Train Loss')
        ax2.plot(epoch_list, losses['validation'], label='Validation Loss')
        ax2.set_ylabel('Loss Value')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss')
        l2 = ax2.legend(loc="best")
        
        plt.savefig(self.create_file_name(figure_name))
        plt.clf()
        plt.close()
        return
    
    def plot_confusion_matrix(self, actuals, predictions, numclasses, figure_name="confusion_matrix"):
        ax= plt.subplot()
        cm = confusion_matrix(actuals, predictions, labels=range(numclasses))
        cm = cm/np.sum(cm, axis=1)
        sns.heatmap(cm, annot=True, fmt=".2f", vmax=1, vmin=0, ax=ax)
        ax.set_xlabel('Predicted labels'), ax.set_ylabel('True labels'), ax.set_title(figure_name)

        plt.savefig(self.create_file_name(figure_name))
        plt.clf()
        plt.close()
        return
    
    def plot_lr(self, lr_list):
        plt.plot(lr_list)
        plt.xlabel("Epoch"), plt.ylabel("Learning rate"), plt.title("Learning Rate Schedule")

        plt.savefig(self.create_file_name("lr_schedule"))
        plt.clf()
        plt.close()
        return