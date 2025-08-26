#!/usr/bin/env python3
"""
Augmented Lagrangian LMC Fairness Analysis Script
This script implements and analyzes fairness constraints in machine learning models,
specifically focusing on gender fairness in income prediction using the Adult dataset
with Augmented Lagrangian Langevin Monte Carlo (AL-LMC).
"""

import os
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from jax import vmap
import jax.numpy as jnp
from jax.nn import sigmoid

# Add project root to path
CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))
from plt_settings import plt_settings

# Plot settings
full_width = 5.5
ratio = 1 / 1.618

def load_data():
    """Load and prepare the Adult dataset."""
    from data.adult import get_data
    return get_data()

def ineqconst(beta, X, male_idx, female_idx):
    """Compute inequality constraints for fairness."""
    return jnp.array([
        100 * (sigmoid(jnp.dot(X, beta)).mean() - sigmoid(jnp.dot(X[male_idx, :], beta)).mean()) - 1,
        100 * (sigmoid(jnp.dot(X, beta)).mean() - sigmoid(jnp.dot(X[female_idx, :], beta)).mean()) - 1,
    ])

def report_acc_overall(beta, X_test, y_test):
    """Report overall accuracy metrics."""
    y_pred = jnp.where(jnp.dot(X_test, beta) < 0.0, 0.0, 1.0).astype(jnp.int32)
    print(classification_report(y_test, y_pred))
    return (y_pred == y_test).mean()

def prevalence(X, beta, male_idx, female_idx):
    """Calculate prevalence metrics."""
    y_pred = jnp.where(jnp.dot(X, beta) < 0.0, 0.0, 1.0)
    return jnp.array([
        y_pred.mean(),
        y_pred[male_idx].mean(),
        y_pred[female_idx].mean(),
    ])

def disparity(X, beta, male_idx, female_idx):
    """Calculate disparity metrics."""
    prev = prevalence(X, beta, male_idx, female_idx)
    return jnp.array([
        prev[1] - prev[0],
        prev[2] - prev[0],
    ])

def plot_dual_variables_comparison(pdlmc_lambda, allmclc_lambda_dict, penalty_rhos):
    """Plot the evolution of dual variables comparing PD-LMC and AL-LMC."""
    with plt.rc_context(plt_settings):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(full_width, 2 * ratio * full_width), dpi=300)
        
        # PD-LMC dual variables
        ax1.plot(pdlmc_lambda[:, 0], label="Male", c="dodgerblue", linewidth=1.5)
        ax1.plot(pdlmc_lambda[:, 1], label="Female", c="hotpink", linewidth=1.5)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel(r"Dual variable ($\lambda$)")
        ax1.set_title("PD-LMC Dual Variables")
        ax1.legend()
        ax1.ticklabel_format(scilimits=(0, 2))
        
        # AL-LMC dual variables for different penalty parameters
        for i, rho in enumerate(penalty_rhos):
            rho_str = str(rho).replace(".", "_")
            lambda_key = f"allmclc_lambda_{rho_str}"
            if lambda_key in allmclc_lambda_dict:
                ax2.plot(allmclc_lambda_dict[lambda_key][:, 0], '--', 
                        c=f'C{i}', alpha=0.7, linewidth=1)
                ax2.plot(allmclc_lambda_dict[lambda_key][:, 1], '-', 
                        c=f'C{i}', label=rf'$\rho = {rho}$', linewidth=1.5)
        
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel(r"Dual variable ($\lambda$)")
        ax2.set_title("AL-LMC Dual Variables")
        ax2.legend()
        ax2.ticklabel_format(scilimits=(0, 2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(CWD, "plots2", "dual_variables_comparison.png"))
        plt.close()

def plot_constraint_trackers(allmclc_c_dict, penalty_rhos):
    """Plot the evolution of constraint trackers in AL-LMC."""
    with plt.rc_context(plt_settings):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(full_width, 2 * ratio * full_width), dpi=300)
        
        # Inequality constraint trackers
        for i, rho in enumerate(penalty_rhos):
            rho_str = str(rho).replace(".", "_")
            c_key = f"allmclc_c_{rho_str}"
            if c_key in allmclc_c_dict:
                c_data = allmclc_c_dict[c_key]
                # First two elements are inequality constraints
                ax1.plot(c_data[:, 0], '--', c=f'C{i}', alpha=0.7, linewidth=1)
                ax1.plot(c_data[:, 1], '-', c=f'C{i}', label=rf'$\rho = {rho}$', linewidth=1.5)
        
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Constraint Tracker (c)")
        ax1.set_title("AL-LMC Inequality Constraint Trackers")
        ax1.legend()
        ax1.ticklabel_format(scilimits=(0, 2))
        
        # Equality constraint trackers (if any)
        for i, rho in enumerate(penalty_rhos):
            rho_str = str(rho).replace(".", "_")
            c_key = f"allmclc_c_{rho_str}"
            if c_key in allmclc_c_dict:
                c_data = allmclc_c_dict[c_key]
                if c_data.shape[1] > 2:  # If there are equality constraints
                    ax2.plot(c_data[:, 2:], '-', c=f'C{i}', label=rf'$\rho = {rho}$', linewidth=1.5)
        
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Constraint Tracker (c)")
        ax2.set_title("AL-LMC Equality Constraint Trackers")
        ax2.legend()
        ax2.ticklabel_format(scilimits=(0, 2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(CWD, "plots2", "constraint_trackers.png"))
        plt.close()

def plot_prevalence_comparison_al_lmc(y_test, test_male_idx, test_female_idx, 
                                    lmc_p, pdlmc_p, allmclc_p_dict, penalty_rhos):
    """Plot prevalence comparison including AL-LMC results."""
    plt_settings["figure.figsize"] = (full_width, ratio * full_width)
    width = 0.15

    def bar_prev(ax, y, disp_female, disp_male, mean, labels=(None, None)):
        ax.barh(y, disp_female, width, left=mean, color="hotpink", label=labels[0])
        ax.barh(y, disp_male, width, left=mean, color="dodgerblue", label=labels[1])
        ax.vlines(mean, y - 1.1 * width, y + 1.1 * width, color="black")

    def violin_prev(ax, y, samples_f, samples_m, color_f="hotpink", color_m="dodgerblue"):
        parts = ax.violinplot(
            dataset=samples_f,
            positions=[y],
            widths=0.8,
            showextrema=False,
            vert=False,
            side="high",
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color_f)
            pc.set_edgecolor(color_f)
            pc.set_alpha(0.8)

        parts = ax.violinplot(
            dataset=samples_m,
            positions=[y],
            widths=0.8,
            showextrema=False,
            vert=False,
            side="low",
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color_m)
            pc.set_edgecolor(color_m)
            pc.set_alpha(0.8)

    with plt.rc_context(plt_settings):
        fig, ax = plt.subplots(1, 1, dpi=300)
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        # Population statistics
        bar_prev(
            ax,
            4,
            y_test[test_female_idx].mean() - y_test.mean(),
            y_test[test_male_idx].mean() - y_test.mean(),
            y_test.mean(),
            ("Female", "Male"),
        )
        
        # Unconstrained model predictions
        violin_prev(
            ax,
            3,
            lmc_p[:, test_female_idx].mean(axis=1),
            lmc_p[:, test_male_idx].mean(axis=1),
        )
        ax.vlines(lmc_p.flatten().mean(), 3 - 1.1 * width, 3 + 1.1 * width, color="black")

        # PD-LMC predictions
        violin_prev(
            ax,
            2,
            pdlmc_p[:, test_female_idx].mean(axis=1),
            pdlmc_p[:, test_male_idx].mean(axis=1),
        )
        ax.vlines(pdlmc_p.flatten().mean(), 2 - 1.1 * width, 2 + 1.1 * width, color="black")

        # AL-LMC predictions for different penalty parameters
        colors = ['C0', 'C1', 'C2']
        for i, rho in enumerate(penalty_rhos):
            rho_str = str(rho).replace(".", "_")
            p_key = f"allmclc_p_{rho_str}"
            if p_key in allmclc_p_dict:
                violin_prev(
                    ax,
                    1 - i,
                    allmclc_p_dict[p_key][:, test_female_idx].mean(axis=1),
                    allmclc_p_dict[p_key][:, test_male_idx].mean(axis=1),
                    color_f=colors[i], color_m=colors[i]
                )
                ax.vlines(allmclc_p_dict[p_key].flatten().mean(), 
                         1 - i - 1.1 * width, 1 - i + 1.1 * width, color="black")

        ax.set_xlabel(r"Prevalence of $>$ \$50k (\%)")
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        ax.set_xticklabels(np.arange(0, 50, 10))
        
        # Set y-ticks for all methods
        y_positions = [4, 3, 2] + [1 - i for i in range(len(penalty_rhos))]
        y_labels = ["Population", "Unconstrained", "PD-LMC"] + [f"AL-LMC ($\\rho={rho}$)" for rho in penalty_rhos]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        
        ax.set_ylim([-len(penalty_rhos) - 0.5, 4.5])
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        
        ax.legend(
            [
                Rectangle((0, 0), 1, 1, color="hotpink", alpha=1.0),
                Rectangle((0, 0), 1, 1, color="dodgerblue", alpha=1.0),
            ],
            ["Female", "Male"],
            handlelength=0.7,
            loc="lower right",
            bbox_to_anchor=(1.01, 0.54),
        )
        
        plt.savefig(os.path.join(CWD, "plots2", "prevalence_comparison_al_lmc.png"))
        plt.close()

def plot_penalty_comparison(allmclc_p_dict, penalty_rhos, test_female_idx, test_male_idx, lmc_p):
    """Plot comparison of different penalty parameters."""
    plt_settings["figure.figsize"] = (full_width / 2, ratio * full_width / 2)
    width = 0.25

    def violin_prev(ax, y, samples_f, samples_m):
        parts = ax.violinplot(
            dataset=samples_f,
            positions=[y],
            widths=0.7,
            showextrema=False,
            vert=False,
            side="high",
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("hotpink")
            pc.set_edgecolor("hotpink")
            pc.set_alpha(0.8)

        parts = ax.violinplot(
            dataset=samples_m,
            positions=[y],
            widths=0.7,
            showextrema=False,
            vert=False,
            side="low",
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("dodgerblue")
            pc.set_edgecolor("dodgerblue")
            pc.set_alpha(0.8)

    with plt.rc_context(plt_settings):
        fig, ax = plt.subplots(1, 1, dpi=300)

        # Plot unconstrained at the top
        violin_prev(
            ax,
            len(penalty_rhos),
            lmc_p[:, test_female_idx].mean(axis=1),
            lmc_p[:, test_male_idx].mean(axis=1),
        )
        ax.vlines(lmc_p.flatten().mean(), len(penalty_rhos) - 1.1 * width, 
                 len(penalty_rhos) + 1.1 * width, color="black")

        # Plot all penalty parameters
        for i, rho in enumerate(penalty_rhos):
            rho_str = str(rho).replace(".", "_")
            p_key = f"allmclc_p_{rho_str}"
            if p_key in allmclc_p_dict:
                violin_prev(
                    ax,
                    i,
                    allmclc_p_dict[p_key][:, test_female_idx].mean(axis=1),
                    allmclc_p_dict[p_key][:, test_male_idx].mean(axis=1),
                )
                ax.vlines(allmclc_p_dict[p_key].flatten().mean(), 
                         i - 1.1 * width, i + 1.1 * width, color="black")

        ax.set_xlabel(r"Prevalence of $>$ \$50k (\%)", fontsize=8)
        ax.set_xticks(np.arange(0, 0.4, 0.1))
        ax.set_xticklabels(np.arange(0, 40, 10), fontsize=8)
        ax.set_xlim(0, 0.38)
        
        # Set y-ticks for all penalty parameters plus unconstrained
        ax.set_yticks(range(len(penalty_rhos) + 1))
        labels = [f"$\\rho = {rho}$" for rho in penalty_rhos]
        labels.append("Unconstrained")
        ax.set_yticklabels(labels, fontsize=8)

        ax.grid()
        ax.legend(
            [
                Rectangle((0, 0), 1, 1, color="hotpink", alpha=1.0),
                Rectangle((0, 0), 1, 1, color="dodgerblue", alpha=1.0),
            ],
            ["Female", "Male"],
            handlelength=0.5,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.2,
            fontsize=8,
            loc="lower right",
            bbox_to_anchor=(1.01, -0.01),
            bbox_transform=ax.transAxes
        )
        plt.savefig(os.path.join(CWD, "plots2", "penalty_comparison.png"))
        plt.close()

def plot_statistical_parity_comparison(lmc_samples, pdlmc_samples, allmclc_samples_dict, 
                                     penalty_rhos, X, male_idx, female_idx):
    """Plot the evolution of statistical parity for different methods."""
    plt_settings["figure.figsize"] = (full_width / 2, ratio * full_width / 2)
    
    def calc_stat_parity(samples):
        # Process in batches to save memory
        batch_size = 1000
        n_samples = len(samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        stat_parity = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = samples[start_idx:end_idx]
            
            # Calculate predictions for batch
            predictions = vmap(lambda beta: jnp.where(jnp.dot(X, beta) < 0.0, 0.0, 1.0))(batch)
            
            # Calculate statistical parity for batch
            male_rates = predictions[:, male_idx].mean(axis=1)
            female_rates = predictions[:, female_idx].mean(axis=1)
            stat_parity.extend(jnp.abs(male_rates - female_rates))
            
        return jnp.array(stat_parity)
    
    with plt.rc_context(plt_settings):
        _, ax = plt.subplots(1, 1, dpi=300)
        
        # Plot unconstrained case
        unconstrained_parity = calc_stat_parity(lmc_samples)
        ax.plot(unconstrained_parity, '--', color='gray', label='Unconstrained', alpha=0.7)
        
        # Plot PD-LMC case
        pdlmc_parity = calc_stat_parity(pdlmc_samples)
        ax.plot(pdlmc_parity, '--', color='black', label='PD-LMC', alpha=0.7)
        
        # Plot each penalty parameter case
        for idx, (samples, rho) in enumerate(zip(allmclc_samples_dict.values(), penalty_rhos)):
            stat_parity = calc_stat_parity(samples)
            ax.plot(stat_parity, c=f'C{idx}', label=rf'AL-LMC $\rho = {rho}$')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Statistical Parity\n" + r"$|P(Y=1|{\rm Male}) - P(Y=1|{\rm Female})|$")
        ax.ticklabel_format(scilimits=(0, 2))
        ax.legend(fontsize=8, loc='upper right')
        
        plt.savefig(os.path.join(CWD, "plots2", "statistical_parity_al_lmc.png"))
        plt.close()

def plot_feature_importance_comparison(lmc_samples, pdlmc_samples, allmclc_samples_dict, 
                                     penalty_rhos, var_names):
    """Plot feature importance comparison across all methods."""
    plt_settings["figure.figsize"] = (full_width, ratio * full_width)
    width = 0.15
    var_subset = [
        "Intercept",
        "education_Bachelors",
        "occupation_Adm-clerical",
        "race_Black",
        "race_White",
        "gender_Male",
        "native-country_Iran",
        "age_(31.0, 37.0]",
    ]

    var_labels = np.append(["Intercept"], var_names.to_numpy())
    coeff_unc = pd.DataFrame(np.exp(lmc_samples.mean(axis=0)), var_labels)
    coeff_pdlmc = pd.DataFrame(np.exp(pdlmc_samples.mean(axis=0)), var_labels)
    
    # Calculate coefficients for AL-LMC methods
    coeff_allmclc = {}
    for rho in penalty_rhos:
        rho_str = str(rho).replace(".", "_")
        samples_key = f"allmclc_samples_{rho_str}"
        if samples_key in allmclc_samples_dict:
            coeff_allmclc[rho] = pd.DataFrame(np.exp(allmclc_samples_dict[samples_key].mean(axis=0)), var_labels)

    with plt.rc_context(plt_settings):
        fig, ax = plt.subplots(1, 1, dpi=300)
        
        x_pos = np.arange(len(var_subset))
        bar_width = width
        
        # Plot bars for each method
        ax.bar(x_pos - 2*bar_width, coeff_unc.loc[var_subset], bar_width, 
               label="Unconstrained", color="C0", alpha=0.8)
        ax.bar(x_pos - bar_width, coeff_pdlmc.loc[var_subset], bar_width, 
               label="PD-LMC", color="C1", alpha=0.8)
        
        # Plot AL-LMC bars
        colors = ['C2', 'C3', 'C4']
        for i, rho in enumerate(penalty_rhos):
            if rho in coeff_allmclc:
                ax.bar(x_pos + i*bar_width, coeff_allmclc[rho].loc[var_subset], bar_width, 
                       label=f"AL-LMC $\\rho={rho}$", color=colors[i], alpha=0.8)

        ax.hlines(
            1,
            -3 * bar_width,
            len(var_subset) - 1 + 3 * bar_width,
            linestyle="dashed",
            label="_nolegend_",
            color="black",
        )
        ax.set_ylabel("Odds ratio")
        ax.legend()
        ax.set_xticks(np.arange(len(var_subset)))
        ax.set_xticklabels(
            (
                "Intercept",
                "Bachelors",
                "Adm/clerical",
                "African-Amer.",
                "White",
                "Gender",
                "Iran",
                "Age (31.0, 37.0]",
            ),
            rotation=45,
            ha="right",
        )
        plt.savefig(os.path.join(CWD, "plots2", "feature_importance_al_lmc.png"))
        plt.close()

def main():
    """Main execution function."""
    # Create plots2 directory if it doesn't exist
    os.makedirs(os.path.join(CWD, "plots2"), exist_ok=True)
    
    # Load data
    (X, y, var_names, gender_idx, male_idx, female_idx,
     X_test, y_test, test_male_idx, test_female_idx) = load_data()
    
    # Load sampling results
    ITERATIONS = int(2e4)
    BURN_IN = int(1e4)
    penalty_rhos = [0.1, 1.0, 10.0]
    
    # Load main experiment data
    sampling_data = np.load(os.path.join(CWD, "fairness.npz"))
    lmc_samples = sampling_data["lmc_x"][BURN_IN:, :]
    pdlmc_samples = sampling_data["pdlmc_x"][BURN_IN:, :]
    pdlmc_lambda = sampling_data["pdlmc_lambda"]
    
    # Load AL-LMC data
    al_lmc_data = np.load(os.path.join(CWD, "fairness_al_lmc.npz"))
    
    # Extract AL-LMC results
    allmclc_samples_dict = {}
    allmclc_lambda_dict = {}
    allmclc_c_dict = {}
    
    for rho in penalty_rhos:
        rho_str = str(rho).replace(".", "_")
        samples_key = f"allmclc_x_{rho_str}"
        lambda_key = f"allmclc_lambda_{rho_str}"
        c_key = f"allmclc_c_{rho_str}"
        
        if samples_key in al_lmc_data:
            allmclc_samples_dict[samples_key] = al_lmc_data[samples_key][BURN_IN:, :]
        if lambda_key in al_lmc_data:
            allmclc_lambda_dict[lambda_key] = al_lmc_data[lambda_key]
        if c_key in al_lmc_data:
            allmclc_c_dict[c_key] = al_lmc_data[c_key]
    
    # Calculate metrics
    def prob_test(beta):
        return jnp.where(jnp.dot(X_test, beta) < 0, 0.0, 1.0)
    
    lmc_p = vmap(prob_test)(lmc_samples)
    pdlmc_p = vmap(prob_test)(pdlmc_samples)
    
    # Calculate AL-LMC predictions
    allmclc_p_dict = {}
    for rho in penalty_rhos:
        rho_str = str(rho).replace(".", "_")
        samples_key = f"allmclc_x_{rho_str}"
        if samples_key in allmclc_samples_dict:
            allmclc_p_dict[f"allmclc_p_{rho_str}"] = vmap(prob_test)(allmclc_samples_dict[samples_key])
    
    # Calculate accuracies
    print("=== Accuracy Comparison ===")
    lmc_acc = report_acc_overall(lmc_samples.mean(axis=0), X_test, y_test)
    print(f"LMC Accuracy: {lmc_acc:.4f}")
    
    pdlmc_acc = report_acc_overall(pdlmc_samples.mean(axis=0), X_test, y_test)
    print(f"PD-LMC Accuracy: {pdlmc_acc:.4f}")
    
    for rho in penalty_rhos:
        rho_str = str(rho).replace(".", "_")
        samples_key = f"allmclc_x_{rho_str}"
        if samples_key in allmclc_samples_dict:
            allmclc_acc = report_acc_overall(allmclc_samples_dict[samples_key].mean(axis=0), X_test, y_test)
            print(f"AL-LMC (rho={rho}) Accuracy: {allmclc_acc:.4f}")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_dual_variables_comparison(pdlmc_lambda, allmclc_lambda_dict, penalty_rhos)
    print("✓ Dual variables comparison plot saved")
    
    plot_constraint_trackers(allmclc_c_dict, penalty_rhos)
    print("✓ Constraint trackers plot saved")
    
    plot_prevalence_comparison_al_lmc(y_test, test_male_idx, test_female_idx, 
                                    lmc_p, pdlmc_p, allmclc_p_dict, penalty_rhos)
    print("✓ Prevalence comparison plot saved")
    
    plot_penalty_comparison(allmclc_p_dict, penalty_rhos, test_female_idx, test_male_idx, lmc_p)
    print("✓ Penalty comparison plot saved")
    
    plot_statistical_parity_comparison(lmc_samples, pdlmc_samples, allmclc_samples_dict, 
                                     penalty_rhos, X, male_idx, female_idx)
    print("✓ Statistical parity comparison plot saved")
    
    plot_feature_importance_comparison(lmc_samples, pdlmc_samples, allmclc_samples_dict, 
                                     penalty_rhos, var_names)
    print("✓ Feature importance comparison plot saved")
    
    print(f"\nAll plots saved to: {os.path.join(CWD, 'plots2')}")

if __name__ == "__main__":
    main() 