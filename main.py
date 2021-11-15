import os
import time
import shutil
import pickle
import argparse
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import RecurrentAttention
from utils import *
from data_loader import get_data_loader
from torchviz import make_dot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

torch.set_printoptions(threshold=10_000)


class Main:
    
    def __init__(self):
        
        # Glimpse Network Params
        self.num_glimpses = 8 # number of glimpses
        self.patch_size = 10 # size of extracted patch at highest res
        self.glimpse_scale = 1.5 # scale of successive patches
        self.num_patches = 3 # number of downscaled patches per glimpse
        
        # Training params
        self.batch_size = 128 # number of images in each batch of data
        self.epochs = 800 # number of epochs to train for
        self.lr = 1e-5 # Initial learning rate value
        self.lr_patience = 70 # Number of epochs to wait before reducing lr
        self.start_epoch = 0
        self.lr_threshold = 0.01
        self.train_patience = 100 # Number of epochs to wait before stopping train

        # Other params
        self.random_seed = 1 # Seed to ensure reproducibility
        self.best = True # Load best model or most recent for testing
        self.best_valid_acc = 0
        self.counter = 0
        
        # Set the seed
        torch.manual_seed(self.random_seed)
        
         # Check if the gpu is available
        if torch.cuda.is_available():
            
            self.device = torch.device("cuda")
            
            torch.cuda.manual_seed(self.random_seed)
            
            self.num_workers = 1
            self.pin_memory = True
            self.preload = True
            
        else:
            self.device = torch.device("cpu")

        print(f"\n[*] Device: {self.device}")
        
        # Build the model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale
        )
            
        # Set the model to the device
        self.model.to(self.device)

        # Start the optimizer
        self.optimizer = torch.optim.Adam([
                {'params': self.model.glimpse.parameters()},
                {'params': self.model.core.parameters()},
                {'params': self.model.regressor.parameters()},
                {'params': self.model.locator.parameters(), 'lr': 1e-5},
                {'params': self.model.baseliner.parameters(), 'lr': 1e-5},
            ], lr=self.lr)
        
        # Start the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=self.lr_patience, threshold=self.lr_threshold)


    def train(self, resume=None, plot_graph=True):

        # Set the data loader
        data_loaders = get_data_loader(self.batch_size, is_train=True)
        self.train_loader = data_loaders[0]
        self.valid_loader = data_loaders[1]
        
        self.num_train = len(self.train_loader.sampler)
        self.num_valid = len(self.valid_loader.sampler)
          
        self.plot_graph = plot_graph
        
        # Should resume the train
        if resume is None:
            
            # Set the folder time for each execution
            folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

            # Set the model name
            self.model_name = f"ng{self.num_glimpses}_ps{self.patch_size}_np{self.num_patches}_s{self.glimpse_scale}_lr{self.lr}_lrp{self.lr_patience}_{folder_time}"

            # Set the folders
            self.output_path = os.path.join('/content/drive/MyDrive/yuri/RAM-VO/out', self.model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.glimpse_path = os.path.join(self.output_path, 'glimpse')
            self.heatmap_path = os.path.join(self.output_path, 'heatmap')
            self.loss_path = os.path.join(self.output_path, 'loss')
            
            # Create the folders
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if not os.path.exists(self.glimpse_path):
                os.makedirs(self.glimpse_path)
            if not os.path.exists(self.heatmap_path):
                os.makedirs(self.heatmap_path)
            if not os.path.exists(self.loss_path):
                os.makedirs(self.loss_path)
        
        else:
            
            # Set the model to be loaded
            self.model_name = resume
            
            # Set the folders
            self.output_path = os.path.join('/content/drive/MyDrive/yuri/RAM-VO/out', self.model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.glimpse_path = os.path.join(self.output_path, 'glimpse')
            self.heatmap_path = os.path.join(self.output_path, 'heatmap')
            self.loss_path = os.path.join(self.output_path, 'loss')
            
            # Load the model
            self._load_checkpoint(best=False)            
        
        print(f"[*] Output Folder: {self.output_path}")
                
        # Print the model info
        count_parameters(self.model, print_table=False)
        
        # Save the configuration as image
        self._save_config()      

        print(f"[*] Train on {self.num_train} samples, validate on {self.num_valid} samples")

        train_losses = []
        train_accuracy = []
        val_losses = []
        val_accuracy = []
        train_rewards = []
        val_rewards = []
        # For each epoch
        for epoch in range(self.start_epoch, self.epochs):

            # Get the current lr
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch: {epoch+1}/{self.epochs} - LR: {current_lr}")

            # Train one epoch
            train_acc, train_f1, train_reward, train_loss, train_rl, train_entropy, train_data = self._train_one_epoch(epoch)
            train_losses.append(train_loss)
            train_accuracy.append(train_acc)
            train_rewards.append(train_reward)

            # Validate one epoch
            val_acc, val_f1, val_reward, val_loss, val_rl, val_data = self._validate(epoch)
            val_losses.append(val_loss)
            val_accuracy.append(val_acc)
            val_rewards.append(val_reward)

            # Reduce lr if validation loss plateaus
            self.scheduler.step(val_loss)

            # Check if it is the best model
            is_best = val_acc > self.best_valid_acc
            
            msg = "acc: {:.3f}%, f1: {:.3f}, loss: {:.3f}, rl: {:.3f}, H: {:.3f} | val_acc: {:.3f}%, val_f1: {:.3f}, val loss: {:.3f}, val rl: {:.3f}"
            
            # Check for improvement
            if is_best:
                msg += " [*]"
                self.best_valid_acc = val_acc
            
            print(msg.format(train_acc*100, train_f1, train_loss, train_rl, train_entropy, val_acc*100, val_f1, val_loss, val_rl))
            
            # Save the checkpoint for each epoch
            self._save_checkpoint({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                }, is_best
            )
            
            # Dump the losses
            with open(os.path.join(self.loss_path, f"loss_epoch_{epoch+1}.p"), "wb") as f:
                
                data = (train_data, val_data)
                
                pickle.dump(data, f)

    def _train_one_epoch(self, epoch, save_glimpse=False):
         
        self.model.train()
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        rl_bar = AverageMeter()
        entropy_bar = AverageMeter()
        
        # Store the losses array
        loss_action_array = []
        loss_baseline_array = []
        loss_reinforce_array = []
        reward_array = []
        accuracy = []
        f1s = []
        rewards = []

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            
            # For each batch
            for i, (x, y) in enumerate(self.train_loader):
                                
                # Set data to the respected device
                x, y = x.to(self.device), y.to(self.device)
                x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
                labels = y
                y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=7).float()

                glimpse_location = []
                log_pi = []
                baselines = []
                entropy_pi = []
                
                # Reset the model parameters
                l_t = torch.zeros([self.batch_size, 2], dtype=torch.float32).to(self.device)
                
                h_state = (torch.zeros(1, self.batch_size, 512).to(self.device), torch.zeros(1, self.batch_size, 512).to(self.device))
                
                # For each glimpse
                for t in range(self.num_glimpses):
                    
                    # Store the glimpse location for both frames
                    glimpse_location.append(l_t)
                                        
                    # Call the model
                    h_state, l_t, b_t, predicted, p, _, entropy = self.model(x, l_t, h_state)
                    
                    # Get the glimpse location loss
                    log_pi.append(p)
                    entropy_pi.append(entropy)
                    
                    # Add the baseline
                    baselines.append(b_t)

                # Convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                entropy_pi = torch.stack(entropy_pi).transpose(1, 0)
                
                # Denormalize the predictions
                predicted_denormalized = torch.stack([denormalize_displacement(l, 48) for l in predicted])

                weight = torch.FloatTensor([7.1, 65.8, 7., 3.9, 5.9, 9., 5.7])
                loss_cross = torch.nn.CrossEntropyLoss(weight=weight)
                #loss_cross = torch.nn.CrossEntropyLoss()
                loss_mse = torch.nn.MSELoss()
                                
                # Compute the reward based on L2 norm
                R = (1/(1 + torch.square(torch.sub(predicted_denormalized.detach(), y.detach())).mean(dim=1))) 
                
                R_unsqueeze = R.unsqueeze(1).repeat(1, self.num_glimpses)
                    
                # Compute losses for differentiable modules
                loss_action = loss_cross(predicted_denormalized, labels.type(torch.LongTensor))
                loss_baseline = loss_mse(baselines, R_unsqueeze)

                # Compute reinforce loss, summed over timesteps and averaged across batch
                advantages = R_unsqueeze - baselines.detach()

                loss_reinforce = -(log_pi*advantages).sum(dim=1).mean()

                # Join the losses
                loss = loss_action + loss_baseline + loss_reinforce
                
                # Store the losses
                loss_action_array.append(loss_action.cpu().data.numpy())
                loss_baseline_array.append(loss_baseline.cpu().data.numpy())
                loss_reinforce_array.append(loss_reinforce.cpu().data.numpy())
                reward_array.append(torch.mean(R).cpu().data.numpy())  

                # Store the loss and metric
                losses.update(loss.item(), x.size()[0])
                rl_bar.update(loss_reinforce.item(), x.size()[0])
                entropy_bar.update(entropy_pi.mean().item(), x.size()[0])

                # Compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                
                #plot_grad_flow(self.model.named_parameters(), plot=True)
                
                self.optimizer.step()

                predictions = torch.argmax(predicted_denormalized, dim=-1)
                acc = balanced_accuracy_score(labels.cpu(), predictions.cpu())
                f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
                accuracy.append(acc)
                f1s.append(f1)
                rewards.append(torch.mean(R).cpu().data.numpy())

                # Measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                # Set the var description
                pbar.set_description(("{:.1f}s - acc: {:.3f}%, f1: {:.3f}, loss: {:.3f}, rl: {:.3f}".format((toc-tic), acc*100, f1, loss.item(), loss_reinforce.item())))
                
                # Update the bar
                pbar.update(self.batch_size)
                
                # Plot the graph
                if i == 0 and self.plot_graph:
                       
                    print(f"[*] Plotting the graph")
                    
                    # Generate the plot
                    make_dot(loss, params=dict(self.model.named_parameters()), engine="neato").render(os.path.join(self.output_path, "ramvo_neato"), format="pdf", cleanup=True)
                    make_dot(loss, params=dict(self.model.named_parameters()), engine="dot").render(os.path.join(self.output_path, "ramvo_dot"), format="pdf", cleanup=True)

                # Save glimpses for the first batch
                if i == 0 and save_glimpse:
                    
                    # Format the data for storage
                    img_0 = [g.cpu().data.numpy().squeeze() for g in x]
                    img_1 = [g.cpu().data.numpy().squeeze() for g in x]
                    loc_0 = [l.cpu().data.numpy() for l in glimpse_location]
                    loc_1 = [l.cpu().data.numpy() for l in glimpse_location]
                    
                    # Build the data to be saved
                    data = ((img_0, loc_0), (img_1, loc_1))
                    
                    # Dump the glimpses
                    with open(os.path.join(self.glimpse_path, f"glimpses_epoch_{epoch+1}.p"), "wb") as f:
                        pickle.dump(data, f)
                        
                 # Save glimpses for the heatmap every 5 minibatches
                if i % 5 == 0: #  or epoch == 0
                                        
                    # Dump the glimpses for heatmap
                    with open(os.path.join(self.heatmap_path, f"epoch_{epoch+1}_minibatch_{i}.p"), "wb") as f:
                        pickle.dump(torch.stack(glimpse_location), f)
                
            # Build the train data array
            train_data = (accuracy, f1s, reward_array, loss_action_array, loss_baseline_array, loss_reinforce_array)

            # Convert to numpy array
            train_data = map(np.asarray, train_data)
            
            mean_acc = sum(accuracy) / float(len(accuracy))
            mean_f1 = sum(f1s) / float(len(f1s))
            mean_reward = sum(rewards) / float(len(rewards))

        return mean_acc, mean_f1, mean_reward, losses.avg, rl_bar.avg, entropy_bar.avg, train_data

    @torch.no_grad()
    def _validate(self, epoch):
  
        losses = AverageMeter()
        rl_bar = AverageMeter()
            
        # Store the losses array
        loss_action_array = []
        loss_baseline_array = []
        loss_reinforce_array = []
        reward_array = []
        accuracy = []
        f1s = []
        rewards = []

        for i, (x, y) in enumerate(self.valid_loader):
            
            # Set data to the respected device
            x, y = x.to(self.device), y.to(self.device)
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
            labels = y
            y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=7).float()


            glimpse_location = []
            log_pi = []
            baselines = []
            predicted = None
            
            # Reset the model parameters
            l_t = torch.zeros([self.batch_size, 2], dtype=torch.float32).to(self.device)
            
            h_state = (torch.zeros(1, self.batch_size, 512).to(self.device), torch.zeros(1, self.batch_size, 512).to(self.device))
            
            # For each glimpse
            for t in range(self.num_glimpses):
                
                # Call the model
                h_state, l_t, b_t, predicted, p, _, _ = self.model(x, l_t, h_state)

                # Store the glimpse location for both frames
                glimpse_location.append(l_t)
                
                # Get the glimpse location loss
                log_pi.append(p)
                
                # Add the baseline
                baselines.append(b_t)

            # Convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # Denormalize the predictions
            predicted_denormalized = torch.stack([denormalize_displacement(l, 48) for l in predicted])
            
            loss_mse = torch.nn.MSELoss()
        
            # Compute the reward based on L2 norm
            R = (1/(1 + torch.square(torch.sub(predicted_denormalized.detach(), y.detach())).mean(dim=1))) 

            R_unsqueeze = R.unsqueeze(1).repeat(1, self.num_glimpses)
                
            # Compute losses for differentiable modules
            weight = torch.FloatTensor([7.1, 65.8, 7., 3.9, 5.9, 9., 5.7])
            loss_cross = torch.nn.CrossEntropyLoss(weight=weight)
            #loss_cross = torch.nn.CrossEntropyLoss()
            loss_action = loss_cross(predicted_denormalized, labels.type(torch.LongTensor))
            loss_baseline = loss_mse(baselines, R_unsqueeze)

            # Compute reinforce loss, summed over timesteps and averaged across batch
            advantages = R_unsqueeze - baselines.detach()

            loss_reinforce = -(log_pi*advantages).sum(dim=1).mean()

            # Join the losses
            loss = loss_action + loss_baseline + loss_reinforce

            predictions = torch.argmax(predicted_denormalized, dim=-1)
            acc = balanced_accuracy_score(labels.cpu(), predictions.cpu())
            f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
            accuracy.append(acc)
            f1s.append(f1)
            rewards.append(torch.mean(R).cpu().data.numpy())
            
            # Store the losses
            loss_action_array.append(loss_action.cpu().data.numpy())
            loss_baseline_array.append(loss_baseline.cpu().data.numpy())
            loss_reinforce_array.append(loss_reinforce.cpu().data.numpy())
            reward_array.append(torch.mean(R).cpu().data.numpy())  

            # Store the loss and metric
            losses.update(loss.item(), x.size()[0])
            rl_bar.update(loss_reinforce.item(), x.size()[0])
            
        # Build the validation data array
        validation_data = (accuracy, f1s, reward_array, loss_action_array, loss_baseline_array, loss_reinforce_array)
            
        # Convert to numpy array
        validation_data = map(np.asarray, validation_data)

        mean_acc = sum(accuracy) / float(len(accuracy))
        mean_f1 = sum(f1s) / float(len(f1s))
        mean_reward = sum(rewards) / float(len(rewards))

        return mean_acc, mean_f1, mean_reward, losses.avg, rl_bar.avg, validation_data

    @torch.no_grad()
    def test(self, model_name):

        self.test_loader = get_data_loader(self.batch_size, is_train=False)[0]

        self.num_test = len(self.test_loader.sampler)
   
        # Set the model to load
        self.model_name = model_name
        
        # Set the folders
        self.output_path = os.path.join('/content/drive/MyDrive/yuri/RAM-VO/out', self.model_name)
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
        self.glimpse_path = os.path.join(self.output_path, 'glimpse')
        self.loss_path = os.path.join(self.output_path, 'loss')
        
        # Set the model to test
        self.model.locator.is_test = False
        
        # Load the model
        self._load_checkpoint(best=False)  
        
        # Print the model info
        count_parameters(self.model, print_table=False)
            
        predicted_array = None
        y_array = None
        accuracy = []
        f1s = []
        all_ys = torch.Tensor()
        all_predictions = torch.Tensor()
        
        print(f"[*] Test on {self.num_test} samples")

        for i, (x, y) in enumerate(self.test_loader):
             
            # Set data to the respected device
            x, y = x.to(self.device), y.to(self.device)
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
            labels = y
            y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=7).float()

            # Reset the model parameters
            l_t = torch.zeros([self.batch_size, 2], dtype=torch.float32).to(self.device)
            
            h_state = (torch.zeros(1, self.batch_size, 512).to(self.device), torch.zeros(1, self.batch_size, 512).to(self.device))
            
            predicted = None
            
            l_t_acc = []

            # For each glimpse
            for t in range(self.num_glimpses):

                l_t_acc.append(l_t[0].cpu().data.numpy())
                
                # Call the model
                h_state, l_t, b_t, predicted, p, _, _ = self.model(x, l_t, h_state)

            # Denormalize the predictions
            predicted_denormalized = torch.stack([denormalize_displacement(l, 48) for l in predicted])
            
            if predicted_array is None:
                predicted_array = predicted_denormalized
                y_array  = y
            else:
                predicted_array = torch.cat((predicted_array, predicted_denormalized))
                y_array = torch.cat([y_array, y])

            predictions = torch.argmax(predicted_denormalized, dim=-1)
            acc = balanced_accuracy_score(labels.cpu(), predictions.cpu())
            f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
            accuracy.append(acc)
            f1s.append(f1)

            all_ys = torch.cat((all_ys, labels))
            all_predictions = torch.cat((all_predictions, predictions))

        mean_acc = sum(accuracy) / float(len(accuracy))
        mean_f1 = sum(f1s) / float(len(f1s))

        print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        for i in range(len(labels)):
            print(f'expected {int(labels[i])} and got {predictions[i]}')
        print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(f'Accuracy: {mean_acc}, F1: {mean_f1}')

        cf_matrix = confusion_matrix(all_ys.tolist(), all_predictions.tolist())

        plt.figure(figsize = (7,7))

        matrix = sns.heatmap(cf_matrix, annot=True)
        results_path = os.path.join(self.output_path, 'confusion_matrix.png')
        plt.savefig(results_path, dpi=400)


    def _save_checkpoint(self, state, is_best):

        filename = self.model_name + "_checkpoint.tar"
        
        # Set the checkpoint path
        ckpt_path = os.path.join(self.checkpoint_path, filename)
        
        # Save the checkpoint
        torch.save(state, ckpt_path)
        
        # Save the best model
        if is_best:
            filename = os.path.join(self.output_path, self.model_name + "_best_model.tar")
            
            # Copy the checkpoint to the best model
            shutil.copyfile(ckpt_path, os.path.join(self.checkpoint_path, filename))

    def _load_checkpoint(self, best=False):

        print(f"[*] Loading model from {self.checkpoint_path}")

        # Define which model to load
        if best:
            filename = self.model_name + "_best_model.tar"
        else:
            filename = self.model_name + "_checkpoint.tar"
            
        # Set the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_path, filename)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load the variables from checkpoint
        self.start_epoch = checkpoint["epoch"]
        self.best_valid_acc = checkpoint["best_valid_acc"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optim_state"])

        if best:
            print(f"[*] Loaded {checkpoint_path} checkpoint @ epoch {self.start_epoch} with best valid acc of {self.best_valid_acc}")
        else:
            print(f"[*] Loaded {checkpoint_path} checkpoint @ epoch {self.start_epoch}")
            
            
    def _save_config(self):
            
        df = pd.DataFrame()
        df['patch size'] = [self.patch_size]
        df['glimpse scale'] = [self.glimpse_scale]
        df['num patches'] = [self.num_patches]
        df['num glimpses'] = [self.num_glimpses]
        df['batch size'] = [self.batch_size]
        df['epochs'] = [self.epochs]
        df['lr'] = [self.lr]
        df['num train'] = [self.num_train]
        df['num valid'] = [self.num_valid]
        
        df = df.astype(str)
        
        # Render the table
        render_table(df, self.output_path, 'config.svg')
                
        
def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--test", type=str, required=False, help="should train or test")
    arg.add_argument("--resume", type=str, required=False, help="should resume the train")
    arg.add_argument("--plot_graph", type=str2bool, required=False, help="should plot the graph")
    
    args = vars(arg.parse_args())
    
    return args


if __name__ == "__main__":
    
    args = parse_arguments()

    main = Main()
    
    if args['test'] is not None:
        main.test(args['test'])
    else:
        main.train(args['resume'], args['plot_graph'])
    
