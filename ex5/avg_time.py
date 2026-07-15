import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log_and_plot(file_path):
    times = []

    # Regular expression to capture frame number, relative time, and FPS
    log_pattern = re.compile(r'detection time = ([\d\.]+)ms')

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = log_pattern.search(line)
                if match:
                    time = float(match.group(1))
                    times.append(time)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    if not times:
        print("No matching data found in the log file. Please check the file content and format.")
        return

    # Calculate performance metrics
    min_time = np.min(times)
    max_time = np.max(times)
    avg_time = np.mean(times)

    # Print summary statistics to the terminal
    print(f"\nSuccessfully parsed {len(times)} frames.")
    print("=" * 40)
    print("           AVERAGE PROCESSING TIME       ")
    print("=" * 40)
    print(f"  Minimum time : {min_time:.2f}")
    print(f"  Maximum time : {max_time:.2f}")
    print(f"  Average time : {avg_time:.2f}")
    print("=" * 40)

    # Plotting FPS over Time
    # Note: To plot against Frame Number instead of Time, change 'times' to 'frames'
    plt.plot(times, label='INST FPS', color='tab:blue', linewidth=1.5)
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    # Chart styling
    plt.xlabel('Frame')
    plt.ylabel('Detection Time')
    plt.title('Detection Time - ' + log_file_path)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the output visualization
    output_plot = log_file_path[:-10] + '_detection_times.png'
    plt.savefig(output_plot, dpi=300)
    print(f"Plot successfully generated and saved to '{output_plot}'")

if __name__ == '__main__':
    if len(sys.argv) == 2:
        log_file_path = sys.argv[1]
    else:
        print("ERROR: Provide txt file")
        exit
    
    print(log_file_path)
    parse_log_and_plot(log_file_path)
