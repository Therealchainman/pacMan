FROM ubuntu:20.04
WORKDIR /app
COPY . .
RUN apt update -y \
    && apt install python3.8 -y \
    && apt install python3-pip -y \
    
CMD ["/bin/bash"]