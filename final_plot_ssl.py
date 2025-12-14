import numpy as np
import matplotlib.pyplot as plt
import os
import sys

DATA_DIR = r'E:\Compiled\Curtis\School\Coolegg\Rice\PhD\Y2S1\Intro to Deep Learning\Final\data_processed'
OUTPUT_PLOTS_DIR = './poster_figures'

def load_comparison_data():
    """Loads the final F1 data for both models from the Test Set."""
    try:
        cnn_test_f1_scores = [0.87, 0.72, 0.85, 0.93, 0.49, 0.18]
        cnn_data = {
            'classes': ['Bass', 'Piano', 'Guitar', 'Drums', 'Organ', 'Strings'],
            'f1_scores': cnn_test_f1_scores,
            'macro_f1': 0.6712 
        }

        # Load SSL Transformer FINAL Test Set results (saved by ssl_evaluate.py)
        ssl_data = np.load(os.path.join(DATA_DIR, 'ssl_final_test_results.npy'), allow_pickle=True).item()
        
        return cnn_data, ssl_data
    except FileNotFoundError as e:
        print(f"ERROR: Missing file required for final comparison")
        print(f"Missing file: {e}")
        sys.exit()

def plot_comparison(cnn_data, ssl_data):
    """
    Generates a comparative bar chart showing the performance of the Baseline CNN 
    vs. the SSL Fine-tuned Transformer
    """
    
    classes = cnn_data['classes']
    cnn_f1 = cnn_data['f1_scores']
    ssl_f1 = ssl_data['f1_scores']
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))

    rects1 = ax.bar(x - width/2, cnn_f1, width, label=f'CNN Baseline (F1: {cnn_data["macro_f1"]:.4f})', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, ssl_f1, width, label=f'SSL Transformer (F1: {ssl_data["macro_f1"]:.4f})', color='#ff7f0e', edgecolor='black')

    # Add text labels above the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    ax.set_title('CNN Baseline vs. SSL Transformer Performance (Test Set)', fontsize=16)
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Instrument Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Highlight the Strings
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color='red', alpha=0.1) 
    
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_PLOTS_DIR, 'figure_4_ssl_final_comparison.png')
    plt.savefig(filename)
    plt.close()
    
    print(f"\nPlot saved to {filename}")

if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Missing dependency: matplotlib")
        sys.exit()
    cnn_data, ssl_data = load_comparison_data()
    plot_comparison(cnn_data, ssl_data)