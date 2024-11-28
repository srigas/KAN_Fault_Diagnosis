import torch
import itertools

import numpy as np
import pandas as pd

from kan import KAN
from paretoset import paretoset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_data(df, scaler=None, data_split=(70,15,15), final_eval=False, feat_idxs=None, device='cuda', exp_seed=42):
    """
    Creates a dataset object to be used for KAN training, as well as the test data to be used for evaluation

    Args:
    -----
        df (pandas.core.frame.DataFrame):
            full data dataframe
        scaler (sklearn.preprocessing._data.SomeScaler):
            optional scaler to perform scaling of features
        data_split (tuple):
            tuple with percentages of train/val/test data
        final_eval (bool):
            essentially instructs to combine train with val data
        feat_idxs (pandas.core.indexes.base.Index):
            list of indices corresponding to features that we keep - relevant only after feature selection step
        device (string):
            device on which the experiment will be run
        exp_seed (int):
            seed for reproducibility

    Returns:
    --------
        dataset (dict):
            dictionary containing the training, validation and test data for the KAN's training

    """
    # Feature vector
    X = df.drop(columns=['label'])
    # Labels
    y = df['label']

    # If not in feature selection mode, work only with relevant features
    if feat_idxs is not None:
        X = X[feat_idxs]

    # Get percentage of test+validation data in terms of all data
    first_split = 1.0 - (data_split[0]/sum(data_split))

    # Perform first split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=first_split, stratify=y, random_state=exp_seed)

    # Get percentage of test data in terms of test+val data
    second_split = 1.0 - (data_split[1]/(data_split[1]+data_split[2]))

    # Perform second split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=second_split, stratify=y_temp, random_state=exp_seed)
    X_train, y_train = X_train.values, y_train.values
    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    # Merge train and val if final_eval is true
    # In this case we do not care about dataset['val_input'] and dataset['val_label']
    if final_eval==True:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
    
    # Perform scaling if necessary
    if scaler:
        # Fit the scaler on the training data and transform all data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    # Convert data to Torch tensors
    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train_scaled).type(torch.float32).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.long).to(device)
    dataset['val_input'] = torch.from_numpy(X_val_scaled).type(torch.float32).to(device)
    dataset['val_label'] = torch.from_numpy(y_val).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X_test_scaled).type(torch.float32).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.long).to(device)

    return dataset


def feature_selection(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True, data_split=(70,15,15), device='cuda', exp_seed=42, verbose=True):
    """
    Performs feature selection for a given task.

    Args:
    -----
        df (pandas.core.frame.DataFrame):
            full data dataframe
        grid_size (int):
            size of grid for the KANs
        grid_eps (float):
            0.0 < grid_eps <= 1.0 - determines grid adaptability
        k (int):
            order of B-splines
        thresholds (array-like):
            array of all possible thresholds to be tested
        lambdas (array-like):
            array of all possible lambdas to be tested
        optim (string):
            either "LBFGS" or "Adam"
        epochs (int):
            number of steps for the optimizer during each training session
        use_scaler (bool):
            whether to scale the data or not
        data_split (tuple):
            tuple with percentages of train/val/test data - third value can be zero
        device (string):
            device on which the experiment will be run
        exp_seed (int):
            seed for reproducibility

    Returns:
    --------
        featsdf (pandas.core.frame.DataFrame):
            dataframe containing the full results of the grid search during feature selection

    """

    # Initialize a scaler if scaler=True
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_data(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device, exp_seed=exp_seed)

    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    f1_scores = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed, device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = dataset['train_label'].unique().shape[0]
        
        # Train vanilla model 
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed, auto_save=False, device=device)
        
        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb, loss_fn=torch.nn.CrossEntropyLoss())
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_data(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device, exp_seed=exp_seed)
        
        new_input = new_data['train_input'].shape[1]
        new_output = new_data['train_label'].unique().shape[0]

        # Train new model, using only kept features
        new_model = KAN(width=[new_input, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed, auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=0.0, loss_fn=torch.nn.CrossEntropyLoss())

        # Evaluate final model on validation data
        test_preds = torch.argmax(new_model.forward(new_data['val_input']).detach(), dim=1).cpu()
        truth = new_data['val_label'].cpu()

        # Calculate weighted f1-score
        metric = 100*f1_score(truth, test_preds, average='weighted')
    
        del model
        del new_model
    
        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed, device)
    
            features.append(feats)
            f1_scores.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved Weighted F1-Score of {score:.2f}%.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            f1_scores.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['f1_scores'] = np.array(f1_scores)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['f1_scores'] = featsdf['f1_scores'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats" : featsdf['num_feats'].values, "f1_scores" : featsdf['f1_scores'].values})
    
        # Minimize number of features and maximize F1-Score
        mask = paretoset(paretodf, sense=["min", "max"])
        
        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    return featsdf


def model_selection(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80, grid_update_num=10, stop_grid_update_step=100, alpha=0.05, beta=1.5, r2_threshold=0.0, device='cuda', exp_seed=42, verbose=True):
    """
    Performs model selection for a given task.

    Args:
    -----
        dataset (dict):
            dataset containing train and validation data
        grid_sizes (array-like):
            array of all grid sizes to test during the grid search
        grid_es (array-like):
            array of all grid_eps values to test during the grid search
        k (int):
            order of B-splines
        optim (string):
            either "LBFGS" or "Adam"
        epochs (int):
            number of steps for the optimizer during each training session
        grid_update_num (int):
            how many times to perform grid adaptation throughout training
        stop_grid_update_step (int):
            epoch number to stop performing grid adaptations
        alpha (float):
            parameter that controls the weight of complexity during symbolic expression choosing
        beta (float):
            parameter that controls the weight of the R^2 score during symbolic expression choosing
        r2_threshold (float):
            0.0 <= r2_threshold < 1.0 - threshold for symbolic function choice based on R^2 score
        device (string):
            device on which the experiment will be run
        exp_seed (int):
            seed for reproducibility

    Returns:
    --------
        modelsdf (pandas.core.frame.DataFrame):
            dataframe containing the full results of the grid search during model selection

    """

    # Get the combination of (grid_size, grid_e) pairs
    combinations = list(itertools.product(grid_sizes, grid_es))

    # Initialize lists
    f1_kan = []
    f1_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device):

        # Initialize a model
        kan_input = dataset['train_input'].shape[1]
        kan_output = dataset['train_label'].unique().shape[0]
        model = KAN(width=[kan_input, kan_output], grid=grid_size, k=k, grid_eps=grid_e, seed=exp_seed, auto_save=False, device=device)

        # Check for non adaptive training
        update_grid = False if grid_e > 0.99 else True
            
        # Train Model
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward', lamb=0.0, update_grid=update_grid, grid_update_num=grid_update_num, stop_grid_update_step=stop_grid_update_step, loss_fn=torch.nn.CrossEntropyLoss())
        
        # Evaluate on validation data
        preds = torch.argmax(model.forward(dataset['val_input']).detach(), dim=1).cpu()
        truth = dataset['val_label'].cpu()
        f1kan = 100*f1_score(truth, preds, average='weighted')
        
        # Symbolify Model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)
        # Evaluate Symbolic Version
        preds_sym = torch.argmax(model.forward(dataset['val_input']).detach(), dim=1).cpu()
        f1sym = 100*f1_score(truth, preds_sym, average='weighted')
    
        del model
    
        return f1kan, f1sym

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            f1kan, f1sym = closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device)
    
            f1_kan.append(f1kan)
            f1_sym.append(f1sym)

            if verbose:
                print(f"KAN Model: Weighted F1-Score of {f1kan:.2f}%.\t Symbolic Model: Weighted F1-Score of {f1sym:.2f}%.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            f1_kan.append(0)
            f1_sym.append(0)
        ct += 1
    
    # Gather results in dataframe, to be returned
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['f1_kan'] = f1_kan
    modelsdf['f1_sym'] = f1_sym
    # Drop None values
    modelsdf = modelsdf.dropna()
    modelsdf['f1_kan'] = modelsdf['f1_kan'].astype('float64')
    modelsdf['f1_sym'] = modelsdf['f1_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"f1_kan" : modelsdf['f1_kan'].values, "f1_sym" : modelsdf['f1_sym'].values})
    
        # Minimize number of features and maximize F1-Score
        mask = paretoset(paretodf, sense=["max", "max"])
    
        # Add a column to the DataFrame to distinguish Pareto set points
        modelsdf['pareto'] = mask

    return modelsdf


def plot_heatmaps(df, indices, savepath, interpolation='none', cmap='Spectral', titles=['Heatmap Plot']*2, x_label='x', y_label='y', cbar_labels=['Metric']*2):
    
    # Pivot the dataframe to get x as rows, y as columns, and z0/z1 as values
    data_0 = df.pivot(index=indices['y'], columns=indices['x'], values=indices['z0'])
    data_1 = df.pivot(index=indices['y'], columns=indices['x'], values=indices['z1'])

    # Extract x and y values
    x_values = data_0.columns
    y_values = data_0.index
    
    # Create 5 evenly spaced ticks for x and y (lowest, largest, and 3 in between)
    x_ticks = np.linspace(0, len(x_values) - 1, 5, dtype=int)  # Get 5 tick positions
    x_tick_labels = np.round(x_values[x_ticks],3)  # Get corresponding labels
    
    y_ticks = np.linspace(0, len(y_values) - 1, 5, dtype=int)  # Get 5 tick positions
    y_tick_labels = np.round(y_values[y_ticks],3)  # Get corresponding labels

    # Create subplots
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot first heatmap
    im0 = ax0.imshow(data_0, aspect='auto', origin='lower', cmap=cmap, interpolation=interpolation)
    ax0.set_title(titles[0])
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)

    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_tick_labels)
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_tick_labels)
    
    plt.colorbar(im0, ax=ax0, label=cbar_labels[0])
        
    # Plot second heatmap
    im1 = ax1.imshow(data_1, aspect='auto', origin='lower', cmap=cmap, interpolation=interpolation)
    ax1.set_title(titles[1])
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)

    plt.colorbar(im1, ax=ax1, label=cbar_labels[1])
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure locally
    plt.savefig(savepath, dpi=300)
    
    # Show the plot
    plt.show()


def plot_pareto(df, savepath, plotcols=['x', 'y', 'pareto'], bg_col='white', pareto_col='red', nonpareto_col='blue', labels=['Non-Pareto', 'Pareto'], title='Scatter Plot', x_label='x', y_label='y'):
    plotdf = df[plotcols]

    # Create a scatter plot using seaborn
    plt.figure(figsize=(6, 4))

    plt.gca().set_facecolor(bg_col)
    plt.gca().set_axisbelow(True)
    
    # Plot non-Pareto points
    sns.scatterplot(x=plotcols[0], y=plotcols[1], data=plotdf[~plotdf[plotcols[2]]], label=labels[0], color=nonpareto_col, alpha=0.8, s=70)
    
    # Plot Pareto points with a different color
    sns.scatterplot(x=plotcols[0], y=plotcols[1], data=plotdf[plotdf[plotcols[2]]], label=labels[1], color=pareto_col, alpha=0.8, s=120)
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add a grid
    plt.grid(True)
    
    # Add a legend to distinguish Pareto and non-Pareto points
    plt.legend(loc='best')

    # Adjust layout
    plt.tight_layout()

    # Save the figure locally
    plt.savefig(savepath, dpi=300)
    
    # Show the plot
    plt.show()


def plot_cm(truth, preds, class_names, percs=True, cmap='Blues', title='Confusion Matrix', x_label='Predicted Label', y_label='True Label'):
    # Generate the confusion matrix
    cm = confusion_matrix(truth, preds)

    # Transpose the CM for correct orientation
    cm = cm.T

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(6, 4))

    if percs == True:
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap=cmap, xticklabels=class_names,  
            annot_kws={"size": 8}, yticklabels=class_names, cbar_kws={'label': 'Percentage (%)'})
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=class_names,  
            annot_kws={"size": 8}, yticklabels=class_names)

    plt.xticks(fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel(x_label, fontsize=12, labelpad=10)
    plt.ylabel(y_label, fontsize=12, labelpad=10)
    
    plt.show()

