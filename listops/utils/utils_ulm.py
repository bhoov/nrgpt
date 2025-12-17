from torch.optim.lr_scheduler import ExponentialLR

import torch
import numpy as np
import os
import matplotlib.pyplot as plt



class AnnealingLR(ExponentialLR):
    def __init__(self, optimizer, lr_min, lr_max, epochs, last_epoch=-1):
        self.epochs = epochs
        self.epoch_counter = 0 # we use this to know when to stop decaying the LR
        self.get_decay_rate(lr_min, lr_max, epochs)
        super(AnnealingLR, self).__init__(optimizer, self.gamma, last_epoch)
    
    def get_decay_rate(self, lr_min, lr_max, epochs):
        self.gamma = (lr_min/lr_max)**(1/epochs)
    
    # we can redefine step to change the learning rate only before "epochs" epochs
    def step(self):
        if self.epoch_counter < self.epochs:
            self.epoch_counter += 1
            super(AnnealingLR, self).step()
        else:
            pass

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.reset()
        
    def reset(self, energy=float('inf')):
        self.best_energy = energy
        self.patience_counter = 0
        self.early_stopping_triggered = False
        
    def check_early_stopping(self, energy):
        if energy < self.best_energy - self.min_delta:
            self.reset(energy)
        else:
            self.patience_counter += 1
            self.early_stopping_triggered = False
        if self.patience_counter >= self.patience:
            self.early_stopping_triggered = True
            # print("Early stopping")
            
    def __call__(self, energy):
        self.check_early_stopping(energy)
        return self.early_stopping_triggered
    
    

# Train and test splits
def train_test_split(data, test_size=0.1):
    """
    Splits the data into train and test sets.
    """
    n = int((1 - test_size) * len(data))
    train_data = data[:n]
    test_data = data[n:]
    return train_data, test_data

def get_batch(data, block_size=64, batch_size=64):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, eval_iters=10, block_size=64,batch_size=64):
    model.eval()
    losses = []
    for k in range(eval_iters):
        X, Y = get_batch(data,batch_size=batch_size, block_size=block_size)
        logits, loss = model(X, Y)
        losses += [loss.item()]
    model.train()
    return np.mean(losses)


def early_stopping(metric_list,
            small_window = 32,
            big_window = 1000,
            stop_delta_ratio = 1e-3, verbose=False):
    if len(metric_list) < 2*small_window:
        return False
    # check if chenges within big window and small window are smaller then the ratio
    big_window = max(big_window, 2*small_window)
    last = np.mean(metric_list[-small_window:])
    dl_small =  abs(last - np.mean(metric_list[-2*small_window:-small_window]))
    idx = max(0,len(metric_list)-big_window)
    dl_big = abs(last - np.mean(metric_list[idx:idx+small_window]))
    ratio = dl_small / dl_big
    if verbose: 
        print(f'step: {len(metric_list)}, Loss change ratio: {ratio:.3g}', end='\r')
        # print(f'Loss change ratio: {ratio:.3g}', end='\r')
    return ratio < stop_delta_ratio 



def plot_loss(model_config, history,figsize=(10, 5)):
    """
    Plot the loss history and save it to a file.
    """
    #plot the loss
    plt.figure(figsize=figsize)
    # plt.plot(np.cumsum(time_history), loss_history, label='training')
    plt.plot(np.cumsum(history['time']), history['train_loss'], label='train')
    plt.plot(np.cumsum(history['time']), history['val_loss'], label='val')
    # plt.plot(np.cumsum(valid_time), valid_loss, label='validation')
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    # plt.title('Loss history ' + ' layer #:' + str(N_LAYER)+ ' embedding #:' + str(N_EMBD) + ' model: ' + MODEL)
    plt.title(f'Loss history {model_config.run_name}')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(model_config.save_path, f'loss_{model_config.run_name}.pdf'), bbox_inches='tight')
    plt.close()
        
        
# Create a minimal dummy class to replace wandb
class DummyWandb:
    class Run:
        def __init__(self):
            self.name = "local_run"
            self.project = "local_project"
        
        def use_artifact(self, *args, **kwargs):
            pass
            
        def save(self):
            pass

    class Config(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def update(self, *args, **kwargs):
            pass
            
        def keys(self):
            return []

    class Artifact:
        def __init__(self, name, *args, **kwargs):
            self.name = name
            
        def add_file(self, *args, **kwargs):
            pass
    
    def __init__(self):
        self.run = self.Run()
        self.config = self.Config()
        
    def init(self, *args, **kwargs):
        print("Using dummy wandb logger")
        return self
        
    def log(self, *args, **kwargs):
        pass
        
    def finish(self):
        pass
        
    def Table(self, *args, **kwargs):
        class DummyTable:
            def add_data(self, *args, **kwargs):
                pass
        return DummyTable()
        
    def use_artifact(self, *args, **kwargs):
        class DummyArtifact:
            def __init__(self):
                self.name = "dummy_artifact"
        return DummyArtifact()
        
    def log_artifact(self, artifact):
        return artifact