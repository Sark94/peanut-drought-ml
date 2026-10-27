"""Plots computed from predictions, never from manually entered counts."""
def plot_confusion(y,pred,path,classes=('Control','Drought'),title='External validation',dpi=300):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(y,pred,labels=[0,1],display_labels=classes,
        cmap='Blues',colorbar=False,ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path,dpi=dpi,bbox_inches='tight')
    plt.close(fig)
