#!/bin/bash
# Script to create multiple terminal windows to create a master-worker environment in Locust
# Start Locust master node in a new virtual terminal
locust -f locust-socket.py -H http://dhruva-api.bhashini.gov.in --master &

# Start Locust worker nodes in separate virtual terminals
for ((i=1; i<=2; i++)); do
    locust -f locust-socket.py --worker &
done

# Show Locust master web interface URL
# echo "Locust master web interface is available at http://localhost:8089"