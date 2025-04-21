import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for writing to file (avoids GUI issues)
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import argparse
import json
from datetime import datetime

def parse_time_file(file_path, get_cost=False):
    """
    Parses a time file to extract input names and time or cost values.
    
    Args:
        file_path: Path to the time.txt file
        get_cost: If True, extract cost values instead of time values
    
    Returns:
        Dictionary mapping input names to time or cost values
    """
    data = {}
    # Regex to find the line structure and capture relevant parts
    line_regex = re.compile(r"inputs/(.*?)\.vrp.*?Cost\s+(\S+)\s+(\S+)\s+(\S+).*?Time\(seconds\)\s+(\S+)\s+(\S+)\s+(\S+)")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = line_regex.search(line)
                if match:
                    input_name = match.group(1)
                    if get_cost:
                        # Extract the third cost value (final solution cost)
                        val_str = match.group(4)
                    else:
                        # Extract the third time value (total execution time)
                        val_str = match.group(7)
                    
                    try:
                        val = float(val_str)
                        data[input_name] = val
                    except ValueError:
                        print(f"Warning: Could not parse {'cost' if get_cost else 'time'} '{val_str}' for input {input_name} in file {file_path}. Skipping.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return None
    return data

def generate_comparison_plots(folders, output_name):
    """
    Generate comparison plots between different implementation folders.
    
    Args:
        folders: List of folder names to compare
        output_name: Base name for output files
    """
    # Dictionaries to store data from each folder
    time_data = {}
    cost_data = {}
    
    # Parse data from all folders
    for folder in folders:
        time_file_path = os.path.join(folder, 'time.txt')
        
        # Parse time data
        time_data[folder] = parse_time_file(time_file_path, get_cost=False)
        
        # Parse cost data
        cost_data[folder] = parse_time_file(time_file_path, get_cost=True)
    
    # Find common inputs across all folders
    common_inputs = set.intersection(*[set(data.keys()) for data in time_data.values() if data is not None])
    
    if not common_inputs:
        print("Error: No common inputs found across all specified folders.")
        return
    
    # Create data for plotting
    plot_data = []
    
    for inp in common_inputs:
        entry = {'input': inp}
        
        # Add time data
        valid_time_entry = True
        for folder in folders:
            if time_data[folder] is None or inp not in time_data[folder] or time_data[folder][inp] < 0:
                valid_time_entry = False
                break
            entry[f'{folder}_time'] = time_data[folder][inp]
        
        # Add cost data
        valid_cost_entry = True
        for folder in folders:
            if cost_data[folder] is None or inp not in cost_data[folder] or cost_data[folder][inp] < 0:
                valid_cost_entry = False
                break
            entry[f'{folder}_cost'] = cost_data[folder][inp]
        
        if valid_time_entry and valid_cost_entry:
            plot_data.append(entry)
    
    if not plot_data:
        print("Error: No valid common inputs remaining after filtering.")
        return
    
    # Sort data by the first folder's time values
    plot_data.sort(key=lambda x: x[f'{folders[0]}_time'])
    
    # Extract data for plotting
    inputs = [entry['input'] for entry in plot_data]
    times = {folder: [entry[f'{folder}_time'] for entry in plot_data] for folder in folders}
    costs = {folder: [entry[f'{folder}_cost'] for entry in plot_data] for folder in folders}
    
    # Generate x-axis indices
    x_indices = np.arange(len(inputs))
    
    # Create plots directory if it doesn't exist
    folders_joined = '_'.join(folders)
    os.makedirs('plots/'+folders_joined, exist_ok=True)
    
    # --- Plot 1: Time Comparison ---
    plt.figure(figsize=(18, 10))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))]
    
    for i, folder in enumerate(folders):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(x_indices, times[folder], marker=marker, linestyle=linestyle, label=folder, markersize=4)
    
    plt.xlabel("Input Instance (Sorted by First Implementation's Time)")
    plt.ylabel("Time (seconds)")
    plt.title(f"Execution Time Comparison ({output_name})")
    plt.xticks(x_indices, inputs, rotation=90, fontsize=8)
    plt.legend()
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout(pad=1.5)
    
    time_plot_filename = os.path.join('plots/'+folders_joined, f'{output_name}_time_comparison.png')
    plt.savefig(time_plot_filename, dpi=300)
    print(f"Time comparison plot saved to {time_plot_filename}")
    plt.close()
    
    # --- Plot 2: Time Comparison (Log Scale) ---
    plt.figure(figsize=(18, 10))
    
    for i, folder in enumerate(folders):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(x_indices, times[folder], marker=marker, linestyle=linestyle, label=folder, markersize=4)
    
    plt.xlabel("Input Instance (Sorted by First Implementation's Time)")
    plt.ylabel("Time (seconds) - Log Scale")
    plt.title(f"Execution Time Comparison - Log Scale ({output_name})")
    plt.xticks(x_indices, inputs, rotation=90, fontsize=8)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='both', linestyle=':')
    plt.tight_layout(pad=1.5)
    
    time_log_plot_filename = os.path.join('plots/'+folders_joined, f'{output_name}_time_comparison_log.png')
    plt.savefig(time_log_plot_filename, dpi=300)
    print(f"Time comparison log plot saved to {time_log_plot_filename}")
    plt.close()
    
    # --- Plot 3: Cost Comparison ---
    plt.figure(figsize=(18, 10))
    
    for i, folder in enumerate(folders):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(x_indices, costs[folder], marker=marker, linestyle=linestyle, label=folder, markersize=4)
    
    plt.xlabel("Input Instance (Sorted by First Implementation's Time)")
    plt.ylabel("Solution Cost")
    plt.title(f"Solution Cost Comparison ({output_name})")
    plt.xticks(x_indices, inputs, rotation=90, fontsize=8)
    plt.legend()
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout(pad=1.5)
    
    cost_plot_filename = os.path.join('plots/'+folders_joined, f'{output_name}_cost_comparison.png')
    plt.savefig(cost_plot_filename, dpi=300)
    print(f"Cost comparison plot saved to {cost_plot_filename}")
    plt.close()
    
    # --- Plot 4: Percentage Time Improvement Relative to First Implementation ---
    if len(folders) > 1:
        plt.figure(figsize=(18, 10))
        
        baseline_times = times[folders[0]]
        
        for i, folder in enumerate(folders[1:], 1):
            improvement = [((base - comp) / base) * 100 if abs(base) > 1e-9 else 0 
                          for base, comp in zip(baseline_times, times[folder])]
            
            # Cap extreme values for better visualization
            improvement = [max(min(imp, 500), -500) for imp in improvement]
            
            colors = ['green' if imp > 0 else 'red' for imp in improvement]
            plt.bar(x_indices + (i-1)*0.25, improvement, width=0.2, label=folder, color=colors, alpha=0.7)
        
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel("Input Instance")
        plt.ylabel(f"Time Improvement Relative to {folders[0]} (%)")
        plt.title(f"Time Performance Improvement (Positive = Faster than {folders[0]})")
        plt.xticks(x_indices, inputs, rotation=90, fontsize=8)
        plt.legend()
        plt.grid(True, axis='y', linestyle=':')
        plt.tight_layout(pad=1.5)
        
        time_improvement_filename = os.path.join('plots/'+folders_joined, f'{output_name}_time_improvement.png')
        plt.savefig(time_improvement_filename, dpi=300)
        print(f"Time improvement plot saved to {time_improvement_filename}")
        plt.close()
    
    # --- Plot 5: Percentage Cost Improvement Relative to First Implementation ---
    if len(folders) > 1:
        plt.figure(figsize=(18, 10))
        
        baseline_costs = costs[folders[0]]
        
        for i, folder in enumerate(folders[1:], 1):
            improvement = [((base - comp) / base) * 100 if abs(base) > 1e-9 else 0 
                          for base, comp in zip(baseline_costs, costs[folder])]
            
            # Cap extreme values for better visualization
            improvement = [max(min(imp, 500), -500) for imp in improvement]
            
            colors = ['green' if imp > 0 else 'red' for imp in improvement]
            plt.bar(x_indices + (i-1)*0.25, improvement, width=0.2, label=folder, color=colors, alpha=0.7)
        
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel("Input Instance")
        plt.ylabel(f"Cost Improvement Relative to {folders[0]} (%)")
        plt.title(f"Solution Quality Improvement (Positive = Better than {folders[0]})")
        plt.xticks(x_indices, inputs, rotation=90, fontsize=8)
        plt.legend()
        plt.grid(True, axis='y', linestyle=':')
        plt.tight_layout(pad=1.5)
        
        cost_improvement_filename = os.path.join('plots/'+folders_joined, f'{output_name}_cost_improvement.png')
        plt.savefig(cost_improvement_filename, dpi=300)
        print(f"Cost improvement plot saved to {cost_improvement_filename}")
        plt.close()
    
    # Generate metrics summary
    metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'comparison_name': output_name,
        'folders_compared': folders,
        'num_instances': len(inputs),
        'instances': inputs,
        'time_metrics': {},
        'cost_metrics': {}
    }
    
    # Time metrics
    for folder in folders:
        folder_times = times[folder]
        metrics['time_metrics'][folder] = {
            'min': min(folder_times),
            'max': max(folder_times),
            'mean': np.mean(folder_times),
            'median': np.median(folder_times),
            'total': sum(folder_times)
        }
    
    # Cost metrics
    for folder in folders:
        folder_costs = costs[folder]
        metrics['cost_metrics'][folder] = {
            'min': min(folder_costs),
            'max': max(folder_costs),
            'mean': np.mean(folder_costs),
            'median': np.median(folder_costs),
            'total': sum(folder_costs)
        }
    
    # Time comparison metrics if there are multiple folders
    if len(folders) > 1:
        metrics['time_comparison'] = {}
        metrics['cost_comparison'] = {}
        
        base_folder = folders[0]
        
        for folder in folders[1:]:
            # Time improvements
            time_improvements = [((t1 - t2) / t1) * 100 if abs(t1) > 1e-9 else 0 
                               for t1, t2 in zip(times[base_folder], times[folder])]
            
            metrics['time_comparison'][f'{folder}_vs_{base_folder}'] = {
                'mean_improvement_pct': np.mean(time_improvements),
                'median_improvement_pct': np.median(time_improvements),
                'instances_faster': sum(1 for imp in time_improvements if imp > 0),
                'instances_slower': sum(1 for imp in time_improvements if imp < 0),
                'instances_equal': sum(1 for imp in time_improvements if imp == 0),
                'max_improvement_pct': max(time_improvements),
                'max_degradation_pct': min(time_improvements),
                'speedup_factor': np.mean([t1 / t2 if t2 > 0 else float('inf') for t1, t2 in zip(times[base_folder], times[folder])])
            }
            
            # Cost improvements
            cost_improvements = [((c1 - c2) / c1) * 100 if abs(c1) > 1e-9 else 0 
                               for c1, c2 in zip(costs[base_folder], costs[folder])]
            
            metrics['cost_comparison'][f'{folder}_vs_{base_folder}'] = {
                'mean_improvement_pct': np.mean(cost_improvements),
                'median_improvement_pct': np.median(cost_improvements),
                'instances_better': sum(1 for imp in cost_improvements if imp > 0),
                'instances_worse': sum(1 for imp in cost_improvements if imp < 0),
                'instances_equal': sum(1 for imp in cost_improvements if imp == 0),
                'max_improvement_pct': max(cost_improvements),
                'max_degradation_pct': min(cost_improvements)
            }
    
    # Save metrics to JSON file
    metrics_filename = os.path.join('plots/'+folders_joined, f'{output_name}_metrics.json')
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_filename}")
    
    # Generate a text report for quick reading
    report_filename = os.path.join('plots/'+folders_joined, f'{output_name}_report.txt')
    with open(report_filename, 'w') as f:
        f.write(f"Performance Comparison Report: {output_name}\n")
        f.write(f"Generated: {metrics['timestamp']}\n")
        f.write(f"Implementations compared: {', '.join(folders)}\n")
        f.write(f"Number of instances: {len(inputs)}\n\n")
        
        f.write("TIME METRICS\n")
        f.write("===========\n")
        for folder in folders:
            tm = metrics['time_metrics'][folder]
            f.write(f"{folder}:\n")
            f.write(f"  Min: {tm['min']:.6f} seconds\n")
            f.write(f"  Max: {tm['max']:.6f} seconds\n")
            f.write(f"  Mean: {tm['mean']:.6f} seconds\n")
            f.write(f"  Median: {tm['median']:.6f} seconds\n")
            f.write(f"  Total: {tm['total']:.6f} seconds\n\n")
        
        if len(folders) > 1:
            f.write("TIME COMPARISONS\n")
            f.write("===============\n")
            for comp_key, comp_data in metrics['time_comparison'].items():
                f.write(f"{comp_key}:\n")
                f.write(f"  Mean improvement: {comp_data['mean_improvement_pct']:.2f}%\n")
                f.write(f"  Median improvement: {comp_data['median_improvement_pct']:.2f}%\n")
                f.write(f"  Instances faster: {comp_data['instances_faster']}\n")
                f.write(f"  Instances slower: {comp_data['instances_slower']}\n")
                f.write(f"  Average speedup factor: {comp_data['speedup_factor']:.2f}x\n\n")
        
        f.write("\nCOST METRICS\n")
        f.write("===========\n")
        for folder in folders:
            cm = metrics['cost_metrics'][folder]
            f.write(f"{folder}:\n")
            f.write(f"  Min: {cm['min']:.2f}\n")
            f.write(f"  Max: {cm['max']:.2f}\n")
            f.write(f"  Mean: {cm['mean']:.2f}\n")
            f.write(f"  Median: {cm['median']:.2f}\n\n")
        
        if len(folders) > 1:
            f.write("COST COMPARISONS\n")
            f.write("===============\n")
            for comp_key, comp_data in metrics['cost_comparison'].items():
                f.write(f"{comp_key}:\n")
                f.write(f"  Mean improvement: {comp_data['mean_improvement_pct']:.2f}%\n")
                f.write(f"  Median improvement: {comp_data['median_improvement_pct']:.2f}%\n")
                f.write(f"  Instances with better cost: {comp_data['instances_better']}\n")
                f.write(f"  Instances with worse cost: {comp_data['instances_worse']}\n\n")
    
    print(f"Text report saved to {report_filename}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate performance comparison plots for VRP solvers')
    parser.add_argument('folders', metavar='folder', type=str, nargs='+',
                        help='folders containing time.txt files to compare')
    parser.add_argument('--name', type=str, default='comparison',
                        help='name for output files (default: comparison)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate plots
    generate_comparison_plots(args.folders, args.name)

if __name__ == "__main__":
    main()