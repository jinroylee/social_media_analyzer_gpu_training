import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def visualize_distribution(scores, title="Engagement Scores Distribution"):
    """
    Visualize the distribution of engagement scores using multiple plots.
    
    Args:
        scores: List or array of scores
        title: Title for the plots
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Convert to numpy array for easier handling
    scores = np.array(scores)

    # 1. Histogram
    axes[0, 0].hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Histogram of Engagement Scores')
    axes[0, 0].set_xlabel('Engagement Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot
    axes[0, 1].boxplot(scores, vert=True)
    axes[0, 1].set_title('Box Plot of Engagement Scores')
    axes[0, 1].set_ylabel('Engagement Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Density plot (KDE)
    axes[1, 0].hist(scores, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    # Add KDE curve
    from scipy import stats
    density = stats.gaussian_kde(scores)
    xs = np.linspace(scores.min(), scores.max(), 200)
    axes[1, 0].plot(xs, density(xs), 'r-', linewidth=2, label='KDE')
    axes[1, 0].set_title('Density Plot of Engagement Scores')
    axes[1, 0].set_xlabel('Engagement Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # 4. Q-Q plot to check normality
    from scipy.stats import probplot
    probplot(scores, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n{title} Statistics:")
    print(f"  Count: {len(scores)}")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Std Dev: {np.std(scores):.4f}")
    print(f"  Min: {np.min(scores):.4f}")
    print(f"  Max: {np.max(scores):.4f}")
    print(f"  25th Percentile: {np.percentile(scores, 25):.4f}")
    print(f"  75th Percentile: {np.percentile(scores, 75):.4f}")
    print(f"  Skewness: {stats.skew(scores):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(scores):.4f}")
    
    # Save the plot
    plt.savefig('modelfactory/data/engagement_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: modelfactory/data/engagement_distribution.png")
    
    plt.show()