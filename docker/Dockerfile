FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive


# Here is all personal preference, with a few notes on software you may find useful
RUN apt-get update && \
apt-get install -y vim zsh zsh-doc tmux git git-lfs less man man-db graphviz && \
apt-get install -y cmake-curses-gui ccache bash-completion && \
apt-get install -y silversearcher-ag keychain htop rcm tree && \
apt-get install -y libbz2-dev liblzma-dev libssl-dev libreadline-dev sudo wget curl && \
apt-get install -y python3 python3-pip python3-dev python3-venv && \
apt-get install -y clang-18 clang++-18 clangd-18 lld-18 clang-tools libedit-dev ninja-build gdb &&\ 
apt-get install -y gettext unzip

ARG USER
ARG GID
ARG UID

RUN userdel -r ubuntu || true
RUN addgroup --gid ${GID} ${USER}
RUN useradd -d /home/${USER} -g ${GID} --no-create-home -u ${UID} --shell /usr/bin/bash ${USER}

RUN adduser ${USER} sudo

RUN mkdir /home/${USER} /var/${USER}
RUN chown ${USER}:${USER} /home/${USER} /var/${USER}

# Make sudo work without a password
RUN sed -i~ -e 's/%sudo\tALL=(ALL:ALL) ALL/%sudo\tALL=(ALL:ALL) NOPASSWD:ALL/g' /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}
ENV HOME=/home/${USER}

# Keep container alive
CMD tmux new-session -d && tail -f /dev/null
