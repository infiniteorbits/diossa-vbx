#!/bin/bash

MODEL=$1
OUTPUT_CSV=$2

echo "Running performance test for $MODEL..."

# Run command and capture to a variable
RESULT=$(onnxruntime_perf_test -m duration -t 20 -I $MODEL)

# Extraction Logic
# We use xargs to trim any leading/trailing whitespace from the values
TPS=$(echo "$RESULT" | grep "Number of inferences per second" | cut -d':' -f2 | xargs)
AVG_LAT=$(echo "$RESULT" | grep "Average inference time cost" | cut -d':' -f2 | sed 's/ ms//' | xargs)
P50=$(echo "$RESULT" | grep "P50 Latency" | cut -d':' -f2 | sed 's/ s//' | xargs)
P90=$(echo "$RESULT" | grep "P90 Latency" | cut -d':' -f2 | sed 's/ s//' | xargs)
CPU=$(echo "$RESULT" | grep "Avg CPU usage" | head -n 1 | cut -d':' -f2 | tr -d '%' | xargs)

# 1. Check if file exists to decide on adding a header
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Model,Inferences_Per_Sec,Avg_Latency_ms,P50_s,P90_s,Avg_CPU_Percent" > "$OUTPUT_CSV"
    echo "Created new file: $OUTPUT_CSV"
fi

# 2. Append the data (using >>)
echo "$MODEL,$TPS,$AVG_LAT,$P50,$P90,$CPU" >> "$OUTPUT_CSV"

echo "Data appended to $OUTPUT_CSV"
