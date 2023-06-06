times=$(cat logfile.log | grep "Time for Client Online: " | awk '{print $5}' | sort -u | awk '{printf "%.6f\n", $1/1000}')
count=$(echo "$times" | wc -l)
echo "Number of unique times: $count" > results.txt
echo "["$(echo "$times" | paste -sd, -)"]" >> results.txt