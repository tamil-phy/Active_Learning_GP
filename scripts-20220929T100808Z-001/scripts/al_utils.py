from datetime import datetime
import os

from modAL.models import ActiveLearner

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from bec import *
from monet import *

class ActLearn:
    def __init__(self, 
                regressor,
                trn_data,
                init_ids,
                save_path,
                compare_loss,
                exp_name,
                total_queries=1000,
                plot=False,
                    ):
        
        self.X_training = trn_data[['x', 'g']].to_numpy()
        self.y_training = trn_data.psi.to_numpy()
        self.compare_loss = compare_loss
        self.plot = plot
        self.total_queries = total_queries
        #ids = [int(i) for i in np.linspace(0, len(X_training)-1, 50)]
        self.init_ids_len = len(init_ids)
        self.optimizer = ActiveLearner(
            estimator=regressor,
            query_strategy=ActLearn._gp_regression_std,
            X_training=self.X_training[init_ids], 
            y_training=self.y_training[init_ids]
        )
        curr_dt = str(datetime.utcnow())
        self.save_path = save_path
        self.exp_name = exp_name
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        self.save_file_nm = f"{exp_name}_AL_{curr_dt}.txt"
        self.txt_save_path = os.path.join(self.save_path, self.save_file_nm)
        with open(self.txt_save_path, 'a') as f:
            f.write(f"#### Base Loss for {exp_name} through GP - {self.compare_loss} \n\n\n")
        
    @staticmethod
    def _gp_regression_std(regressor, X):
        _, std = regressor.predict(X, return_std=True)
        query_idx = np.argmax(std)
        return query_idx, X[query_idx]

    @staticmethod
    def _evaluate_optim(optimizer, sample, low=1, high=100, n=100):
        
        def _evaluate(g):
            gt = get_closest_sim(sample, g)
            X_val = gt[['x', 'g']].to_numpy()
            y_val = gt.psi.to_numpy()
            pr = optimizer.predict(X_val)
            return ((pr - y_val)**2).sum().mean()
        
        return np.array([_evaluate(g) for g in np.linspace(low, high, n)]).mean()

    def train(self, test_sample, low=1, high=100, n=100, gs=[1, 25, 50], min_gs=-20, max_gs=20):
        used_query = []
        losses = []
        min_loss = 999999
        for i, n in enumerate(range(self.total_queries)):
            qid, q = self.optimizer.query(self.X_training)
            self.optimizer.teach(X=self.X_training[qid].reshape(1, -1), 
                            y=self.y_training[qid].reshape(1, ))
            mean_loss = ActLearn._evaluate_optim(self.optimizer, test_sample, low=low, high=high, n=n)
            if mean_loss < min_loss:
                min_loss = mean_loss
                print('Got min loss')
                exp_name = self.exp_name
                sav_plot_path = os.path.join(self.save_path, f"{exp_name}_AL_{i}.svg")
                plot(gs, test_sample, self.optimizer, save_file=sav_plot_path, act_op=True, min_gs=min_gs, max_gs=max_gs)
            losses.append(mean_loss)
            log_text = f"Loss for {self.init_ids_len+i+1} - {mean_loss}"
            print(log_text)
            with open(self.txt_save_path, 'a') as f:
                f.write(f"{log_text}\n")
        
            used_query.append(q[0])
            if mean_loss < self.compare_loss:
                print(f"Found best")
                with open(self.txt_save_path, 'a') as f:
                    f.write(f"Found best in {self.init_ids_len+i+1} query")
                    
                    
                    
                    
def plot(gs, harmonic_sims, opt, save_file, act_op=False, min_gs=-12, max_gs=12):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [3.5, 0.8]})
    ax = axes[0]
    #max_gs = max(gs) + 5
    #min_gs = -max_gs
    for i in range(len(gs)):
        g = gs[i]
        df = get_closest_sim(harmonic_sims, g=g)
        input = df[['x', 'g']]
        
        if act_op == True:
            y_pred, sigma = opt.predict(input, return_std=True)
        else:
            y_pred, sigma = opt.predict(input)

        # refine scatter data
        refine_idx = np.arange(0, len(df.x), 16)
        ax.scatter(df.x[refine_idx], y_pred[refine_idx], alpha=0.9, s=50, c=colors[i], label='prediction' if i == 2 else None)
        
        # new section added to accomodate plot lines on same graph
        if (i == 1):
            refine_idx_ = np.arange(4, len(df.x), 16)
            ax.scatter(df.x[refine_idx_], y_pred[refine_idx_], alpha=0.9, s=50, c=colors[i], label='prediction' if i == 2 else None)
            ax.plot(df.x, df.psi, alpha=0.9, c=colors[i], linestyle='solid', label='observed' if i == 2 else None)

        ax.plot(df.x, df.psi, alpha=0.9, c=colors[i], linestyle='dashed', label='observed' if i == 2 else None)
        # ax.set_title('$g$ = {:.2f}'.format(df.g[0]))
        ax.fill(np.concatenate([df.x, df.x[::-1]]),
               np.concatenate([y_pred - 1.9600 * sigma,
                              (y_pred + 1.9600 * sigma)[::-1]]),
               alpha=.3, fc=colors[i], ec='None', label='95% confidence interval' if i == 2 else None)
        ax.set_title('Wave Function', fontsize=16)
        ax.set_ylabel('$\psi$', fontsize=16)
        ax.set_xlabel('$x$', fontsize=16)
        ax.set_xlim(min_gs, max_gs)
        ax.set_ylim(-0.02, 0.14)
        
    handles, labels = ax.get_legend_handles_labels()
    _handles = [
        mpatches.Patch(color=colors[i], label='$g$ = {:.2f}'.format(gs[i])) 
        for i in range(len(gs)) ]
    _labels = ['$g$ = {:.2f}'.format(gs[i]) for i in range(len(gs))]
    leg = fig.legend(handles + _handles, labels + _labels, loc='upper right', fontsize=11) #, bbox_to_anchor=(.9, 0.85))
    leg.legendHandles[0].set_color(black)
    leg.legendHandles[1].set_color(black)
    leg.legendHandles[2].set_color(black)
    plt.grid()
    axes[-1].set_visible(False)
    plt.savefig(save_file, dpi=500)
