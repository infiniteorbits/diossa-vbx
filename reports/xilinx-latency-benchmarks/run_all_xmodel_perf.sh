#!/bin/bash

MODEL_DIR=$1
OUTPUT_CSV=$2

# Write header if file doesn't exist
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Model,FPS,Total_Frames,Duration_Sec" > "$OUTPUT_CSV"
fi

# Find xmodels and loop through them
find $MODEL_DIR -name "*.xmodel" | while read -r xmodel; do
    echo "Benchmarking DPU: $xmodel ..."

    # Run xdputil benchmark (using 1 thread as per your example)
    # We redirect stderr to stdout because xdputil logs to stderr
    RESULT=$(xdputil benchmark "$xmodel" 1 2>&1)

    # Parse values from the line: FPS= 628.476 number_of_frames= 37710 time= 60.0023 seconds.
    FPS=$(echo "$RESULT" | grep "FPS=" | awk -F'FPS=' '{print $2}' | awk '{print $1}')
    FRAMES=$(echo "$RESULT" | grep "number_of_frames=" | awk -F'number_of_frames=' '{print $2}' | awk '{print $1}')
    TIME=$(echo "$RESULT" | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}')

    # If parsing was successful, append to CSV
    if [ ! -z "$FPS" ]; then
        echo "$xmodel,$FPS,$FRAMES,$TIME" >> "$OUTPUT_CSV"
        echo ">> Result: $FPS FPS"
    else
        echo ">> Error benchmarking $xmodel"
    fi
done

echo "--------------------------------------"
echo "Done. Results saved in $OUTPUT_CSV"
