#!/bin/bash
# Verify submission format and generate summary

echo "=========================================="
echo "Submission Verification"
echo "=========================================="
echo

# Check if submission exists
if [ ! -f "outputs/baseline_submission.csv" ]; then
    echo "❌ No submission found. Run: python run_baseline.py"
    exit 1
fi

echo "✓ Submission file exists"

# Count rows
submission_rows=$(wc -l < outputs/baseline_submission.csv)
expected_rows=1873  # 1 header + 1872 test samples

if [ $submission_rows -eq $expected_rows ]; then
    echo "✓ Row count correct: $submission_rows"
else
    echo "❌ Row count incorrect: $submission_rows (expected $expected_rows)"
    exit 1
fi

# Check columns
header=$(head -n1 outputs/baseline_submission.csv)
expected_header="track_id,Clutter,Cormorants,Pigeons,Ducks,Geese,Gulls,Birds of Prey,Waders,Songbirds"

if [ "$header" = "$expected_header" ]; then
    echo "✓ Column headers correct"
else
    echo "❌ Column headers incorrect"
    echo "Expected: $expected_header"
    echo "Got:      $header"
    exit 1
fi

# Check for missing values
if grep -q ",," outputs/baseline_submission.csv; then
    echo "❌ Missing values detected"
    exit 1
else
    echo "✓ No missing values"
fi

echo
echo "=========================================="
echo "✅ Submission is valid!"
echo "=========================================="
echo
echo "File: outputs/baseline_submission.csv"
echo "Size: $(ls -lh outputs/baseline_submission.csv | awk '{print $5}')"
echo
echo "Sample predictions:"
head -n 6 outputs/baseline_submission.csv | column -t -s,
echo
echo "Ready to submit! 🚀"
