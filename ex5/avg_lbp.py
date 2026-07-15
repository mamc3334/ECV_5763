import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log_and_plot(file_path):
    frames = []
    times = []
    confs = []
    frame=0

    # Regular expression to capture frame number, relative time, and FPS
    log_pattern = re.compile(r'recognized ([\d]+) faces in ([\d.]+)ms')
    log_pattern2 = re.compile(r'Confidence: ([\d.]+)')

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = log_pattern.search(line)
                match2 = log_pattern2.search(line)
                if match:
                    time = float(match.group(2))
                    frame += 1
                    times.append(time)
                    # If 0 faces, record a NAN confidence
                    if ("0" == match.group(1)):
                        frames.append(frame)
                        confs.append(np.nan)
                        
                elif match2:
                    frames.append(frame)
                    confs.append(float(match2.group(1)))
                    # time = None # used to prevent accidental two face detection plotting
                    
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

    min_conf = np.nanmin(confs)
    max_conf = np.nanmax(confs)

    # Print summary statistics to the terminal
    print(f"\nSuccessfully parsed {len(frames)} frames.")
    print("=" * 40)
    print("           AVERAGE RECOGNITION TIME       ")
    print("=" * 40)
    print(f"  Minimum time : {min_time:.2f}")
    print(f"  Maximum time : {max_time:.2f}")
    print(f"  Average time : {avg_time:.2f}")
    print("=" * 40)
    print(f"  Minimum confidence : {min_conf:.4f}")
    print(f"  Maximum confidence : {max_conf:.4f}")
    print("=" * 40)

    # Plotting FPS over Time
    # Note: To plot against Frame Number instead of Time, change 'times' to 'frames'
    plt.plot(times, color='tab:blue', linewidth=1.5)

    # Chart styling
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlabel('Frame')
    plt.ylabel('Detection Time')
    plt.title('Detection Time - ' + log_file_path)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the output visualization
    output_plot = log_file_path[:-11] + '_recognition_times.png'
    plt.savefig(output_plot, dpi=300)
    print(f"Plot successfully generated and saved to '{output_plot}'")

    # Plot confidence over time
    plt.figure()
    plt.scatter(frames, confs, color='tab:blue', s=30)
    # Chart styling
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.title('Confidence - ' + log_file_path)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    output_plot = log_file_path[:-11] + '_confidences.png'
    plt.savefig(output_plot, dpi=300)
    print(f"Plot successfully generated and saved to '{output_plot}'")

if __name__ == '__main__':
    if len(sys.argv) == 2:
        log_file_path = sys.argv[1]
    else:
        print("ERROR: Provide txt file")
        exit
    
    parse_log_and_plot(log_file_path)
