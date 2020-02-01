#!/bin/bash
interface="$(route -n get default | grep 'interface:' | grep -o '[^ ]*$')"

echo "Using interface ${interface} for tcpdump"

tcpdump -w tcpdump.pcap -i en0 -c 1000


echo "Generating log files from tcpdump...."
bro -r tcpdump.pcap darpa2gurekddcup.bro > conn.list

sort -n conn.list > conn_sort.list
echo "conn_sort.list generated!"

echo "Compiling...."
gcc trafAld.c -o trafAld.out
./trafAld.out conn_sort.list

echo "Generating CSV output...."
python log_data.py

echo "SUCCESS!"