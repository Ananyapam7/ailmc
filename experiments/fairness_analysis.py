#!/usr/bin/env python3
"""
Fairness Analysis Script
This script implements and analyzes fairness constraints in machine learning models,
specifically focusing on gender fairness in income prediction using the Adult dataset.
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

def plot_dual_variables(pdlmc_lambda):
    """Plot the evolution of dual variables."""
    with plt.rc_context(plt_settings):
        _, axs = plt.subplots(1, 1, dpi=300)
        axs.plot(pdlmc_lambda[:, 0], label="Male", c="dodgerblue")
        axs.plot(pdlmc_lambda[:, 1], label="Female", c="hotpink")
        axs.grid(True, linestyle='--', alpha=0.7)
        axs.set_xlabel("Iteration")
        axs.set_ylabel(r"Dual variable ($\lambda$)")
        axs.legend(
            [
                Rectangle((0, 0), 1, 1, color="dodgerblue", alpha=1.0),
                Rectangle((0, 0), 1, 1, color="hotpink", alpha=1.0),
            ],
            ["Male", "Female"],
            handlelength=0.7,
        )
        axs.ticklabel_format(scilimits=(0, 2))
        plt.savefig(os.path.join(CWD, "plots", "dual_variables.png"))
        plt.close()

def plot_ergodic_constraints(slacks):
    """Plot ergodic constraints slack."""
    cum_mean = np.cumsum(slacks, axis=0) / np.expand_dims(
        np.arange(1, len(slacks) + 1), axis=1
    )
    
    with plt.rc_context(plt_settings):
        _, axs = plt.subplots(1, 1, dpi=300)
        axs.plot(slacks[:, 0], label="Male", color="dodgerblue", alpha=0.6)
        axs.plot(cum_mean[:, 0], linestyle="--", label=None, color="dodgerblue")
        axs.plot(slacks[:, 1], label="Female", color="hotpink", alpha=0.6)
        axs.plot(cum_mean[:, 1], linestyle="--", label=None, color="hotpink")
        axs.grid(True, linestyle='--', alpha=0.7)
        axs.set_xlabel("Iterations")
        axs.set_ylabel(r"Ergodic constraints slack")
        axs.ticklabel_format(scilimits=(0, 2))
        plt.legend()
        plt.savefig(os.path.join(CWD, "plots", "ergodic_constraints.png"))
        plt.close()

def plot_prevalence_comparison(y_test, test_male_idx, test_female_idx, lmc_p, pdlmc_p, lmc_acc, pdlmc_acc):
    """Plot prevalence comparison between different models."""
    plt_settings["figure.figsize"] = (full_width / 2, ratio * full_width / 2)
    width = 0.25

    def bar_prev(ax, y, disp_female, disp_male, mean, labels=(None, None)):
        ax.barh(y, disp_female, width, left=mean, color="hotpink", label=labels[0])
        ax.barh(y, disp_male, width, left=mean, color="dodgerblue", label=labels[1])
        ax.vlines(mean, y - 1.1 * width, y + 1.1 * width, color="black")

    def violin_prev(ax, y, samples_f, samples_m):
        parts = ax.violinplot(
            dataset=samples_f,
            positions=[y],
            widths=1,
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
            widths=1,
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
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        # Population statistics
        bar_prev(
            ax,
            2,
            y_test[test_female_idx].mean() - y_test.mean(),
            y_test[test_male_idx].mean() - y_test.mean(),
            y_test.mean(),
            ("Female", "Male"),
        )
        
        # Unconstrained model predictions
        violin_prev(
            ax,
            1,
            lmc_p[:, test_female_idx].mean(axis=1),
            lmc_p[:, test_male_idx].mean(axis=1),
        )
        ax.vlines(lmc_p.flatten().mean(), 1 - 1.1 * width, 1 + 1.1 * width, color="black")

        # Constrained model predictions
        violin_prev(
            ax,
            0,
            pdlmc_p[:, test_female_idx].mean(axis=1),
            pdlmc_p[:, test_male_idx].mean(axis=1),
        )
        ax.vlines(pdlmc_p.flatten().mean(), 0 - 1.1 * width, 0 + 1.1 * width, color="black")

        ax.set_xlabel(r"Prevalence of $>$ \$50k (\%)")
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        ax.set_xticklabels(np.arange(0, 50, 10))
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([
            f"Constrained\n(Acc.: {100*pdlmc_acc:.0f} %)",
            f"Unconstrained\n(Acc.: {100*lmc_acc:.0f} %)",
            "Population",
        ])
        ax.set_ylim([-0.75, 2.4])
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        
        ax.legend(
            [
                Rectangle((0, 0), 1, 1, color="pink", alpha=1.0),
                Rectangle((0, 0), 1, 1, color="lightblue", alpha=1.0),
            ],
            ["Female", "Male"],
            handlelength=0.7,
            loc="lower right",
            bbox_to_anchor=(1.01, 0.54),
        )
        
        plt.savefig(os.path.join(CWD, "plots", "prevalence_comparison.png"))
        plt.close()

def plot_relaxed_constraints(pdlmc_lambda_d, deltas):
    """Plot analysis of relaxed constraints."""
    plt_settings["figure.figsize"] = (full_width / 2, ratio * full_width / 2)
    
    with plt.rc_context(plt_settings):
        _, axs = plt.subplots(1, 1, dpi=300)
        for idx, lambdas in enumerate(pdlmc_lambda_d):
            axs.plot(lambdas[:, 0], "--", c=f"C{idx}", label=None)
            axs.plot(lambdas[:, 1], c=f"C{idx}", label=rf"$\delta = {deltas[idx]}$")
        axs.grid()
        axs.set_xlabel("Iteration")
        axs.set_ylabel(r"Dual variable ($\lambda$)")
        axs.legend(loc="lower right")
        plt.savefig(os.path.join(CWD, "plots", "relaxed_constraints.png"))
        plt.close()

def plot_tolerance_comparison(pdlmc_p_d, deltas, test_female_idx, test_male_idx, lmc_p):
    """Plot comparison of different tolerance values and unconstrained model."""
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
            len(deltas),
            lmc_p[:, test_female_idx].mean(axis=1),
            lmc_p[:, test_male_idx].mean(axis=1),
        )
        ax.vlines(lmc_p.flatten().mean(), len(deltas) - 1.1 * width, len(deltas) + 1.1 * width, color="black")

        # Plot all deltas
        for i, delta in enumerate(deltas):
            violin_prev(
                ax,
                i,
                pdlmc_p_d[i][:, test_female_idx].mean(axis=1),
                pdlmc_p_d[i][:, test_male_idx].mean(axis=1),
            )
            ax.vlines(pdlmc_p_d[i].flatten().mean(), i - 1.1 * width, i + 1.1 * width, color="black")

        ax.set_xlabel(r"Prevalence of $>$ \$50k (\%)", fontsize=8)
        ax.set_xticks(np.arange(0, 0.4, 0.1))
        ax.set_xticklabels(np.arange(0, 40, 10), fontsize=8)
        ax.set_xlim(0, 0.38)
        
        # Set y-ticks for all deltas plus unconstrained
        ax.set_yticks(range(len(deltas) + 1))
        labels = [f"$\\delta = {d}$" for d in deltas]
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
        plt.savefig(os.path.join(CWD, "plots", "tolerance_comparison.png"))
        plt.close()

def plot_statistical_parity_comparison(lmc_samples, pdlmc_samples_d, deltas, X, male_idx, female_idx):
    """Plot the evolution of statistical parity for different deltas and unconstrained case."""
    plt_settings["figure.figsize"] = (full_width / 2, ratio * full_width / 2)
    
    def calc_stat_parity(samples):
        # Process in batches to save memory
        batch_size = 1000  # Adjust this based on your memory constraints
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
        
        # Plot each delta case
        for idx, (samples, delta) in enumerate(zip(pdlmc_samples_d, deltas)):
            stat_parity = calc_stat_parity(samples)
            ax.plot(stat_parity, c=f'C{idx}', label=rf'$\delta = {delta}$')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Statistical Parity\n" + r"$|P(Y=1|{\rm Male}) - P(Y=1|{\rm Female})|$")
        ax.ticklabel_format(scilimits=(0, 2))
        ax.legend(fontsize=8, loc='upper right')
        
        plt.savefig(os.path.join(CWD, "plots", "statistical_parity_comparison.png"))
        plt.close()


def plot_feature_importance(lmc_samples, pdlmc_samples, var_names):
    """Plot feature importance comparison."""
    plt_settings["figure.figsize"] = (full_width / 2, ratio * full_width / 2)
    width = 0.4
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
    coeff_cons = pd.DataFrame(np.exp(pdlmc_samples.mean(axis=0)), var_labels)

    with plt.rc_context(plt_settings):
        fig, ax = plt.subplots(1, 1, dpi=300)
        for ii, var in enumerate(var_subset):
            ax.bar(ii - width / 2, coeff_unc.loc[var], width, label="Unconstrained", color="C0")
            ax.bar(ii + width / 2, coeff_cons.loc[var], width, label="Constrained", color="C1")

        ax.hlines(
            1,
            -2 * width,
            len(var_subset) - 1 + 2 * width,
            linestyle="dashed",
            label="_nolegend_",
            color="black",
        )
        ax.set_ylabel("Odds ratio")
        ax.legend(["Unconstrained", "Constrained"])
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
        plt.savefig(os.path.join(CWD, "plots", "feature_importance.png"))
        plt.close()

def main():
    """Main execution function."""
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.join(CWD, "plots"), exist_ok=True)
    
    # Load data
    (X, y, var_names, gender_idx, male_idx, female_idx,
     X_test, y_test, test_male_idx, test_female_idx) = load_data()
    
    # Load sampling results
    ITERATIONS = int(2e4)
    BURN_IN = int(1e4)
    
    sampling_data = np.load(os.path.join(CWD, "fairness.npz"))
    lmc_samples = sampling_data["lmc_x"][BURN_IN:, :]
    pdlmc_samples = sampling_data["pdlmc_x"][BURN_IN:, :]
    pdlmc_lambda = sampling_data["pdlmc_lambda"]
    
    # Calculate metrics
    pdlmc_samples_all = sampling_data["pdlmc_x"]
    slacks = vmap(lambda beta: ineqconst(beta, X, male_idx, female_idx))(pdlmc_samples_all)
    
    def prob_test(beta):
        return jnp.where(jnp.dot(X_test, beta) < 0, 0.0, 1.0)
    
    lmc_p = vmap(prob_test)(lmc_samples)
    pdlmc_p = vmap(prob_test)(pdlmc_samples)
    
    # Calculate accuracies
    lmc_acc = report_acc_overall(lmc_samples.mean(axis=0), X_test, y_test)
    pdlmc_acc = report_acc_overall(pdlmc_samples.mean(axis=0), X_test, y_test)
    
    # Generate plots
    plot_dual_variables(pdlmc_lambda)
    plot_ergodic_constraints(slacks)
    plot_prevalence_comparison(y_test, test_male_idx, test_female_idx, 
                             lmc_p, pdlmc_p, lmc_acc, pdlmc_acc)
    
    # Load and analyze relaxed constraints
    sampling_data_d = np.load(os.path.join(CWD, "fairness_relaxed.npz"))
    deltas = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    pdlmc_lambda_d = [
        sampling_data_d[f"pdlmc_lambda_{i+1}"]
        for i in range(len(deltas))
    ]
    
    pdlmc_samples_d = [
        sampling_data_d[f"pdlmc_x_{i+1}"][BURN_IN:, :]
        for i in range(len(deltas))
    ]
    
    pdlmc_p_d = [vmap(prob_test)(samples) for samples in pdlmc_samples_d]
    
    plot_relaxed_constraints(pdlmc_lambda_d, deltas)
    plot_tolerance_comparison(pdlmc_p_d, deltas, test_female_idx, test_male_idx, lmc_p)
    plot_feature_importance(lmc_samples, pdlmc_samples, var_names)
    
    # Plot statistical parity comparison
    plot_statistical_parity_comparison(lmc_samples, pdlmc_samples_d, deltas, X, male_idx, female_idx)

if __name__ == "__main__":
    main()