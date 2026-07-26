#!/bin/bash

'''
Tried many things but always ran into a bottle neck of one or the other
1. disk space runs out fast when generating ppm and pgm files
2. RAM has restrictions for writing to /dev/shm - 
3. running tasks in parallel caused a lot of overhead
4. mainly disk space - but fps slowed drastically to ~3fps after 300-400 frames
'''

# exit on error
set -eu

file_strip=""

if [[ -z "$1" ]]; then
    echo "Error: no video file"
    exit 1
fi

file=$1
file_strip=${1%.*}
frame_num=1
make oneband

# while true; do
#     printf -v ppm_name "sh_frames/%s_%06d.ppm" "$file_strip" "$frame_num" 
#     (
#         ffmpeg -r 30 -i "$1" -vf "select=eq(n\,$((frame_num-1)))" -vframes 1 "$ppm_name"
    
#         if [[ ! -f "$ppm_name" ]]; then
#             break
#         fi
        
#         ./oneband "$ppm_name"
#         rm "$ppm_name"
#     ) &
    
#     ((frame_num++))
# done

# echo "Processed '$frame_num' frames"

ffmpeg -i "$file" -r 30 "/tmp/${file_strip}_%06d.ppm"

for f in "/tmp/${file_strip}"_*.ppm; do
    ./oneband "$f"
    rm "$f"  # This frees up the memory/swap space instantly for the next frames
done

ffmpeg -r 30 -i "/tmp/${file_strip}_%06d.pgm" "ob_${file_strip}.mpg"

exit 0
