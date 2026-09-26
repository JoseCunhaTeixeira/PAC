# The SSH tunnel to the assistant's model on another machine (docker-compose.yml's tunnel
# service): an SSH client, nothing else.
FROM alpine:3.22
RUN apk add --no-cache openssh-client
# A key mounted from Windows or macOS keeps no Unix permissions, and ssh refuses a key others
# could read: the tunnel uses a copy only its user reads.
ENTRYPOINT ["/bin/sh", "-c", "install -m 600 /ssh/key /tmp/key && exec ssh -i /tmp/key \"$@\"", "ssh"]
