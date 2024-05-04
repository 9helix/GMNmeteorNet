#!/bin/bash

# Download the dataset from the server
scp -i /home/helix/.ssh/rms dgrzinic@gmn.uwo.ca:GMNmeteorNet/mldataset.tar.bz2 .
rm mldataset.tar
rm -rf mldataset
bzip2 -d mldataset.tar.bz2 && tar -xf mldataset.tar
