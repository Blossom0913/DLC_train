import numpy as np
import matplotlib.pyplot as plt

def test_error_bar_calculation():
    """Test the error bar calculation logic"""
    print("=== Testing Error Bar Calculation ===")
    
    # Simulate multiple runs with some variation
    test_data = {
        'accuracy': [70.5, 71.2, 69.8, 70.9, 71.5],
        'weighted_f1': [68.2, 69.1, 67.8, 68.9, 69.5],
        'class_f1': [
            [65.1, 58.3, 72.4, 69.8],  # Run 1
            [66.2, 59.1, 73.1, 70.2],  # Run 2
            [64.8, 57.9, 71.9, 69.5],  # Run 3
            [65.9, 58.8, 72.8, 70.1],  # Run 4
            [66.5, 59.5, 73.5, 70.8]   # Run 5
        ]
    }
    
    # Calculate statistics
    stats = {}
    for key, values in test_data.items():
        if key == 'class_f1':
            values_array = np.array(values)
            mean_val = np.mean(values_array, axis=0)
            std_val = np.std(values_array, axis=0, ddof=1)
            stats[key] = {'mean': mean_val, 'std': std_val}
            print(f"{key}:")
            print(f"  Mean: {mean_val}")
            print(f"  Std:  {std_val}")
        else:
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            stats[key] = {'mean': mean_val, 'std': std_val}
            print(f"{key}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    # Create a simple bar chart with error bars
    plt.figure(figsize=(10, 6))
    
    metrics = ['accuracy', 'weighted_f1']
    means = [stats[m]['mean'] for m in metrics]
    stds = [stats[m]['std'] for m in metrics]
    
    x = np.arange(len(metrics))
    bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=['skyblue', 'lightcoral'])
    
    plt.xlabel('Metrics')
    plt.ylabel('Score (%)')
    plt.title('Test Error Bars')
    plt.xticks(x, metrics)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('test_error_bars.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Error bar calculation test completed!")
    print("✓ Graph saved as 'test_error_bars.png'")

if __name__ == "__main__":
    test_error_bar_calculation()
