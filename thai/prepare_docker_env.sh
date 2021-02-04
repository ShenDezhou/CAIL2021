yum install -y yum-utils
yum-config-manager     --add-repo     https://download.docker.com/linux/centos/docker-ce.repo
yum install -y docker-ce-19.03.15 docker-ce-cli-19.03.15 containerd.io
yum install -y python3
pip3 install -y docker-compose
service docker start