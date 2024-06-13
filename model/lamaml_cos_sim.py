import random
import numpy as np
# import ipdb
import math
import torch
import torch.nn as nn
from model.lamaml_base import *



# calculates the gradient similarity
def projectgrad(gradient, memories):

    similarity = torch.nn.functional.cosine_similarity(gradient.t(), memories.t().mean(dim=0).unsqueeze(0))

    # if(oiter%100==0):
    if True:
        print('similarity : ', similarity.item())



def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1



class Net(BaseNet):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,           
                 args):
        super(Net, self).__init__(n_inputs,
                                 n_outputs,
                                 n_tasks,           
                                 args)
        self.nc_per_task = n_outputs / n_tasks

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        self.observed_tasks = []

    def take_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)

        return loss

    def take_multitask_loss(self, bt, t, logits, y):
        # compute loss on data from a multiple tasks
        # separate from take_loss() since the output positions for each task's
        # logit vector are different and we nly want to compute loss on the relevant positions
        # since this is a task incremental setting

        loss = 0.0

        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)


    def forward(self, x, t):
        output = self.net.forward(x)
        # make sure we predict classes within the current task
        offset1, offset2 = self.compute_offsets(t)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def meta_loss(self, x, fast_weights, y, bt, t):
        """
        differentiate the loss through the network updates wrt alpha
        """

        offset1, offset2 = self.compute_offsets(t)

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss_q = self.take_multitask_loss(bt, t, logits, y)

        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """

        offset1, offset2 = self.compute_offsets(t)            

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss = self.take_loss(t, logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))

        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))

        return fast_weights


    def observe(self, x, y, t):
        self.net.train() 
        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.observed_tasks.append(t)
                self.M = self.M_new.copy()
                self.current_task = t


            # compute gradient on previous tasks
            if len(self.observed_tasks) > 1:
                # for each past task
                for tt in range(len(self.observed_tasks) - 1):
                    past_task = self.observed_tasks[tt]
                    store_grad(self.parameters, self.grads, self.grad_dims,
                               past_task)


            batch_sz = x.shape[0]
            n_batches = self.args.cifar_batches
            rough_sz = math.ceil(batch_sz/n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            # get a batch by augmented incming data with old task data, used for 
            # computing meta-loss
            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)             

            for i in range(n_batches):

                batch_x = x[i*rough_sz : (i+1)*rough_sz]
                batch_y = y[i*rough_sz : (i+1)*rough_sz]

                # assuming labels for inner update are from the same 
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)   
                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch     
                if(self.real_epoch == 0):
                    self.push_to_mem(batch_x, batch_y, torch.tensor(t))
                meta_loss, logits = self.meta_loss(bx, fast_weights, by, bt, t) 
                
                meta_losses[i] += meta_loss

            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            meta_loss = sum(meta_losses)/len(meta_losses)            
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

            # check if gradient violates constraints                                                           
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                    else torch.LongTensor(self.observed_tasks[:-1])

                projectgrad(self.grads[:, t].unsqueeze(1),                                           
                              self.grads.index_select(1, indx))


            if self.args.learn_lr:
                self.opt_lr.step()

            # if sync-update is being carried out (as in sync-maml) then update the weights using the optimiser
            # otherwise update the weights with sgd using updated LRs as step sizes
            if(self.args.sync_update):
                self.opt_wt.step()
            else:            
                for i,p in enumerate(self.net.parameters()):          
                    # using relu on updated LRs to avoid negative values           
                    p.data = p.data - p.grad * nn.functional.relu(self.net.alpha_lr[i])            
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        return meta_loss.item()

