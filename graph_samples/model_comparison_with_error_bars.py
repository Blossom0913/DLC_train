import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Sample data with error bars (you can replace these with actual experimental results)
models = ['CNN', 'LSTM', 'LightGBM', 'GMM']

# Simulated results with means and standard deviations
# In practice, these would come from running multiple experiments
results = {
    'CNN': {
        'accuracy': {'mean': 70.5, 'std': 2.1},
        'f1': {'mean': 68.2, 'std': 2.3},
        'class_f1': {'mean': [65.1, 58.3, 72.4, 69.8], 'std': [3.2, 4.1, 2.8, 3.5]}
    },
    'LSTM': {
        'accuracy': {'mean': 68.3, 'std': 3.2},
        'f1': {'mean': 66.1, 'std': 3.4},
        'class_f1': {'mean': [62.4, 55.7, 70.1, 67.2], 'std': [4.1, 5.2, 3.1, 4.3]}
    },
    'LightGBM': {
        'accuracy': {'mean': 73.4, 'std': 1.8},
        'f1': {'mean': 72.9, 'std': 1.9},
        'class_f1': {'mean': [68.0, 62.1, 75.6, 74.5], 'std': [2.5, 3.1, 2.1, 2.8]}
    },
    'GMM': {
        'accuracy': {'mean': 65.2, 'std': 4.1},
        'f1': {'mean': 63.8, 'std': 4.3},
        'class_f1': {'mean': [60.1, 52.4, 67.3, 65.1], 'std': [4.8, 5.9, 3.9, 4.7]}
    }
}

def create_model_comparison_with_error_bars():
    """Create model comparison graph with error bars"""
    plt.figure(figsize=(12, 8))
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Extract data
    accuracy_means = [results[model]['accuracy']['mean'] for model in models]
    accuracy_stds = [results[model]['accuracy']['std'] for model in models]
    f1_means = [results[model]['f1']['mean'] for model in models]
    f1_stds = [results[model]['f1']['std'] for model in models]
    
    # Create bar chart with error bars
    x = np.arange(len(models))
    width = 0.35
    
    # Plot bars with error bars
    bars1 = plt.bar(x - width/2, accuracy_means, width, label='Accuracy', 
                    yerr=accuracy_stds, capsize=5, alpha=0.8, color='skyblue')
    bars2 = plt.bar(x + width/2, f1_means, width, label='Weighted F1', 
                    yerr=f1_stds, capsize=5, alpha=0.8, color='lightcoral')
    
    # Customize the plot
    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.title('Model Comparison with Error Bars (No Windowing)')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison_with_error_bars.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_per_class_comparison_with_error_bars():
    """Create per-class F1 comparison with error bars"""
    plt.figure(figsize=(14, 8))
    
    classes = ['Base', 'Aggression', 'Social', 'Nonsocial']
    x = np.arange(len(classes))
    width = 0.2
    
    # Plot bars for each model with error bars
    for i, model in enumerate(models):
        means = results[model]['class_f1']['mean']
        stds = results[model]['class_f1']['std']
        
        x_pos = x + i * width - 1.5 * width
        bars = plt.bar(x_pos, means, width, label=model, 
                      yerr=stds, capsize=3, alpha=0.8)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Classes')
    plt.ylabel('F1 Score (%)')
    plt.title('Per-Class F1 Scores with Error Bars')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('per_class_f1_with_error_bars.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_stability_chart():
    """Create performance stability chart (coefficient of variation)"""
    plt.figure(figsize=(10, 6))
    
    # Calculate coefficient of variation (CV = std/mean * 100)
    cv_accuracy = []
    for model in models:
        mean = results[model]['accuracy']['mean']
        std = results[model]['accuracy']['std']
        cv = (std / mean) * 100
        cv_accuracy.append(cv)
    
    # Create bar chart
    bars = plt.bar(models, cv_accuracy, alpha=0.8, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('Performance Stability (Lower is Better)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, cv in zip(bars, cv_accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{cv:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_stability.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_comparison():
    """Create a comprehensive 2x2 subplot comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Comparison with Error Bars', fontsize=16, fontweight='bold')
    
    # 1. Overall Performance
    ax1 = axes[0, 0]
    accuracy_means = [results[model]['accuracy']['mean'] for model in models]
    accuracy_stds = [results[model]['accuracy']['std'] for model in models]
    
    bars = ax1.bar(models, accuracy_means, yerr=accuracy_stds, capsize=5, alpha=0.8)
    ax1.set_title('Overall Accuracy')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, accuracy_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean:.1f}%', ha='center', va='bottom')
    
    # 2. Per-Class F1 Scores
    ax2 = axes[0, 1]
    classes = ['Base', 'Aggression', 'Social', 'Nonsocial']
    x = np.arange(len(classes))
    width = 0.2
    
    for i, model in enumerate(models):
        means = results[model]['class_f1']['mean']
        stds = results[model]['class_f1']['std']
        x_pos = x + i * width - 1.5 * width
        ax2.bar(x_pos, means, width, label=model, yerr=stds, capsize=3, alpha=0.8)
    
    ax2.set_title('Per-Class F1 Scores')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('F1 Score (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Best vs Worst Class Performance
    ax3 = axes[1, 0]
    best_means = []
    worst_means = []
    best_stds = []
    worst_stds = []
    
    for model in models:
        class_f1 = results[model]['class_f1']['mean']
        class_std = results[model]['class_f1']['std']
        best_idx = np.argmax(class_f1)
        worst_idx = np.argmin(class_f1)
        
        best_means.append(class_f1[best_idx])
        worst_means.append(class_f1[worst_idx])
        best_stds.append(class_std[best_idx])
        worst_stds.append(class_std[worst_idx])
    
    x = np.arange(len(models))
    width = 0.35
    
    ax3.bar(x - width/2, best_means, width, label='Best Class F1', 
            yerr=best_stds, capsize=5, alpha=0.8, color='green')
    ax3.bar(x + width/2, worst_means, width, label='Worst Class F1', 
            yerr=worst_stds, capsize=5, alpha=0.8, color='red')
    
    ax3.set_title('Best vs Worst Class Performance')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('F1 Score (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Stability
    ax4 = axes[1, 1]
    cv_accuracy = []
    for model in models:
        mean = results[model]['accuracy']['mean']
        std = results[model]['accuracy']['std']
        cv = (std / mean) * 100
        cv_accuracy.append(cv)
    
    bars = ax4.bar(models, cv_accuracy, alpha=0.8, color='orange')
    ax4.set_title('Performance Stability (Lower is Better)')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, cv in zip(bars, cv_accuracy):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{cv:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison_with_error_bars.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating graphs with error bars...")
    
    # Create individual graphs
    create_model_comparison_with_error_bars()
    create_per_class_comparison_with_error_bars()
    create_performance_stability_chart()
    
    # Create comprehensive comparison
    create_comprehensive_comparison()
    
    print("All graphs created successfully!")
    print("Files saved:")
    print("- model_comparison_with_error_bars.png")
    print("- per_class_f1_with_error_bars.png")
    print("- performance_stability.png")
    print("- comprehensive_comparison_with_error_bars.png")
