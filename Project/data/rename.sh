#!/usr/bin/bash

cd "$(dirname "$0")" || exit 1

i=1

WIDTH=1024
HEIGHT=1024

for dir in */; do
    if [ -d "$dir" ]; then
        # Pass 1: Rename files to temporary names to prevent target collisions
        j=1
        for file in "$dir"/*; do
            if [ -f "$file" ]; then
                mv "$file" "$dir/.tmp_${j}"
                ((j++))
            fi
        done

        # Pass 2: Crop and move temporary files to final target names
        j=1
        for file in "$dir"/.tmp_*; do
            if [ -f "$file" ]; then
                convert "$file" \
                        -auto-orient \
                        -gravity center \
                        -resize "x${HEIGHT}" \
                        +repage \
                        -crop "${WIDTH}x${HEIGHT}+0+0" \
                        +repage \
                        "$file"

                mv "$file" "$dir/${i}_${j}.jpg"
                ((j++))
            fi
        done
        ((i++))
    fi
done