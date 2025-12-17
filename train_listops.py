from listops.config import Config
from listops.utils.utils_ulm import get_batch, estimate_loss, train_test_split, AnnealingLR, plot_loss
import models as Model
from listops.data.tokenizer import Tokenizer, TOKEN_EOS
from listops.training.train_utils import save_data_artifact, get_model_class, strip_num
from listops.training.evaluator import ListOpsTest
import os
import time
import torch
import pickle
import numpy as np
import wandb

# for FLOPS
from fvcore.nn import FlopCountAnalysis

DATA_DIR = './data/listops/'
SAVE_DIR = './results/'
os.makedirs(SAVE_DIR, exist_ok=True)


def training_loop(model, train_data, val_data, optimizer, annealer, model_config):
    """
    Training loop for the model.
    """
    for iter in range(model_config.eval_interval):
        # sample a batch of data
        xb, yb = get_batch(train_data, block_size=model_config.block_size, batch_size=model_config.batch_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        #clamp the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        annealer.step()

    val_loss = estimate_loss(model, val_data, eval_iters=model_config.eval_iters, 
                block_size=model_config.block_size, batch_size=model_config.batch_size) 
    return loss.item(), val_loss

def train(config=None):
    # Initialize a new wandb run with settings to disable auto-plural parameters
    with wandb.init(
        project="LOPS+NRGPT",  # project name
        config=config,
        ):  
        
        # get wandb config parameters
        print("wandb config is: ", wandb.config)
        model_config = Config(  save_path=os.path.join(SAVE_DIR, wandb.run.project), 
                                **wandb.config)
        if model_config.device in ['cuda', 'cpu']: DEVICE = model_config.device
        elif isinstance(model_config.device, str): DEVICE = int(model_config.device)
        else: raise ValueError(f"Invalid device: {model_config.device}")
        
        # load the data
        with open(model_config.data_file, 'rb') as f:
            data = pickle.load(f)
            
        # train_text = data['train']
        # test_data = data['test']
        vocab = data['metadata']['vocab']
        tokenizer = Tokenizer(vocab=vocab)

        # model_config.vocab = vocab
        model_config.vocab_size = len(vocab)
        # set max_iter if not in config
        if 'max_iters' not in wandb.config:
            model_config.max_iters = int(model_config.iter_per_batch * data['metadata'].get('num_train', 20_000))
            print("max_iters not in wandb config, setting to", model_config.max_iters)
        
        wandb.config.update({k:v for k, v in model_config.__dict__.items() if k not in wandb.config.keys()})
        wandb.run.name = model_config.run_name
        os.makedirs(model_config.save_path, exist_ok=True)
        
        from pprint import pprint
        print("####### model_config:#######\n", model_config.__dict__)
        
        save_data_artifact(model_config, data)
        
        train_data_encoded = torch.tensor(data['train'], dtype=torch.long, device=DEVICE)
        train_data, val_data = train_test_split(train_data_encoded, test_size=model_config.validation_split)
        
        model_class = get_model_class(model_config, Model_default=Model)
        print(f"Using model class: {model_class}")
        model = model_class(model_config)
        model = model.to(DEVICE)
        # print the number of parameters in the model
        number_of_params = sum(p.numel() for p in model.parameters())
        wandb.log({"number_of_parameters": number_of_params})
        
        model_config.number_of_params = number_of_params
        print(f'Nr. Params: {number_of_params:.3g}')
        # get FLOPS 
        flops = FlopCountAnalysis(model, torch.zeros(1, model_config.block_size, dtype=torch.long, 
                                                    device=DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)

        # Annealing learning rate
        annealer = AnnealingLR(optimizer, 
                            lr_max=model_config.learning_rate, 
                            lr_min=model_config.min_lr, 
                            epochs=model_config.max_iters)

        print("right before test:", vocab)
        test = ListOpsTest( #model, 
                        data, 
                        num_tests=model_config.num_tests, 
                        device=DEVICE,
                        preencoded=True )

        
        ## Training
        history = {'train_loss': [], 'val_loss': [], 'time': [], 'accuracy': [], 'acc_steps': [],
                    'accuracy_pad': [], }
        t0 = time.time()
        counter = 0
        
        for iter in range(0, model_config.max_iters, model_config.eval_interval):
            train_loss, val_loss = training_loop(model, train_data, val_data, optimizer, annealer, model_config)
            t1 = time.time()
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['time'].append(t1-t0)
            wandb.log({"train_loss": train_loss})
            try:
                if hasattr(model.blocks[0], 'scale_ff'):
                    wandb.log({"scaleFF": model.blocks[0].scale_ff.item()})
            except TypeError:
                if hasattr(model.blocks, 'scale_ff'):
                    wandb.log({"scaleFF": model.blocks.scale_ff.item()})
            wandb.log({"val_loss": val_loss})
            wandb.log({"train_time": history['time'][-1]})
            # log lr
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
            t0 = time.time()
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, time {history['time'][-1]:.2f}s", end='\n')
            
            # get acc every 10 steps
            if iter % (model_config.eval_interval*10) == 0:
                # accuracy, answers = test_accuracy(model, test_data, model_config, num_tests=model_config.num_tests_per_epoch)
                accuracy, answers = test.test_accuracy(model, num_tests=model_config.num_tests_per_epoch, use_pad=True)
                # accuracy, answers = test.test_accuracy(num_tests=model_config.num_tests_per_epoch)
                print(f"\nTest accuracy: {accuracy*100:.1f}%")
                history['acc_steps'].append(iter)
                # the test are padded now. This decreases the accuracy
                history['accuracy_pad'].append(accuracy)
                wandb.log({"accuracy_pad": accuracy})
                if 'test_single_ops' in data:
                    for op in data['test_single_ops']:
                        accuracy_op, answers_op = test.test_acc_single_op(model, op=op, num_tests=model_config.num_tests_per_epoch, use_pad=True)
                        wandb.log({f"accuracy_{op}": accuracy_op})
        
                #print(i, equations_encoded[i]['character'], answer[-1], encode([answer[-1]]), equations_encoded[i]['result'])
            
            ## Early stopping
            if iter >  2000 and model_config.early_stop:
                w=5
                d_loss = (np.mean(history['val_loss'][-w:]) - np.mean(history['val_loss'][-2*w:-w]) )
                wandb.log({"delta_loss": d_loss})
                # if (np.abs(d_loss) < model_config.min_delta) and (d_loss > 0):
                if (abs(d_loss) < model_config.min_delta) or (d_loss > 2*model_config.min_delta):
                    counter += 1
                    #print('counter', counter, np.abs(np.mean(loss_history_val[-100:])-loss.item()))
                else:
                    # counter = 0
                    counter -= 1
                    counter = max(counter, 0)
                print(f"Counter: {counter}, delta loss: {d_loss:.4f}")
                wandb.log({"d_loss_counter": counter})
                
                # if counter >= model_config.patience or np.mean(val_loss[-4:]) < val_loss[-1]:
                if counter >= model_config.patience:    
                    print(f'Early stopping at step {iter} with patience {model_config.patience} and delta {model_config.min_delta}')
                    print(f'Last train loss: {train_loss:.4f}, last val loss: {val_loss:.4f}')
                    print(f'Counter: {counter}, delta: {d_loss:.4f}')
                    break

        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(model_config.save_path, f"model_{model_config.run_name}.pt"))
        # save the optimizer
        torch.save(optimizer.state_dict(), os.path.join(model_config.save_path, f"optimizer_{model_config.run_name}.pt"))
        #dictionary for saving the results
        results = {}
        results['config'] = model_config.__dict__
        results['training_history'] = history
        
        #----------------------------------------------------------------------------------------------------
        # generate from the model
        generated_text = []
        if model_config.do_generate:
            context = torch.tensor(tokenizer.encode(TOKEN_EOS), dtype=torch.long, device=DEVICE).unsqueeze(0)  # shape (1, seq_len)
            print("Context:",context)
            gen = model.generate(context, max_new_tokens=200)
            print("Generated tokens:", gen.shape, gen)
            gen = gen[0].tolist()
            generated_text = tokenizer.decode(gen)
            print(generated_text[:100])
            results['generated_text'] = generated_text
        #----------------------------------------------------------------------------------------------------
        # accuracy, answers = test.test_accuracy_no_pad(model, num_tests=model_config.num_tests)
        accuracy, answers = test.test_accuracy(model, num_tests=model_config.num_tests, use_pad=False)
        wandb.log({"accuracy_final": accuracy})
        results['answers'] = answers
        print("Test accuracy:", accuracy)
        if 'test_single_ops' in data:
            for op in data['test_single_ops']:
                accuracy_op, answers_op = test.test_acc_single_op(model, op=op, num_tests=model_config.num_tests, use_pad=False)
                print(f"Test accuracy for {op}: {accuracy_op}")
                wandb.log({f"accuracy_final_{op}": accuracy_op})
                results[f'accuracy_final_{op}'] = accuracy_op
                results[f'answers_{op}'] = answers_op
        #----------------------------------------------------------------------------------------------------
        
        wandb.log({"last_val_loss": np.mean(history['val_loss'][-3:])})
        
        ops = ','.join([strip_num(func) for func in data['metadata']['funcs_to_use']])
        wandb.log({"ops": ops})

        table = wandb.Table(columns=["Generated Text", "number of Parameters"])
        table.add_data(generated_text[:500], number_of_params)
        wandb.log({"generated_text_table": table})

        #save the result in a pickle file
        with open(os.path.join(model_config.save_path, f"results_{model_config.run_name}.pkl"), 'wb') as f:
            pickle.dump(results, f)
            
        plot_loss(model_config, history)

        wandb.finish()  
        
        
if __name__ == "__main__":
    train()
