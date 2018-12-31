import matplotlib.pyplot as plt

def plot_graph(history):
    #graph of Train and validation accuracy
    plt.clf()   # clear figure
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.show()

    #graph of Train and validation loss
    plt.clf()   # clear figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.show()