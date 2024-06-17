#! /usr/bin/env bash

# Remove assessment result for clean start.
ASSESSMENT_RESULT=/dli/assessment_results/PASSED
if [ -f "$ASSESSMENT_RESULT" ]; then
    rm "$ASSESSMENT_RESULT"
fi

cd /dli/assessment
chmod +x assessment_runner
chmod +x get_filtered_results.sh
./assessment_runner
