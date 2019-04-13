from ubuntu:16.04 

#USER root
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gcc





#Minicona environment
#https://hub.docker.com/r/continuumio/miniconda/dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean


RUN mkdir /myworkdir
VOLUME ["/myworkdir"]
WORKDIR /myworkdir

RUN conda install -y git
RUN conda create -y -n deep-learning python=3.6

COPY ./requirements.txt /tmp/
COPY ./install.sh /tmp/
RUN chmod 777 /tmp/install.sh && /tmp/install.sh

#RUN source activate deep-learning && conda install -y pytorch torchvision -c pytorch 
#RUN source activate deep-learning && pip install -r requirements.txt



#CMD [ "/bin/bash" "source activate deep-learning && jupyter notebook"]
#jupyter notebook --no-browser --port 8889 --ip 0.0.0.0 --allow-root
COPY ./start-notebook.sh /tmp/start-notebook.sh
RUN chmod +x /tmp/start-notebook.sh
CMD ["/tmp/start-notebook.sh"]


#RUN apt-get install -y libfreetype6-dev
#RUN apt-get install -y libpng-dev
#RUN apt-get install -y pkg-config
#RUN apt-get install -y libpq-dev 
#RUN conda install -c conda-forge -y pygpu
#RUN apt-get install -y libgeos-dev
#RUN  apt-get remove python-ptyprocess



#RUN pip install -r /tmp/requirements.txt


