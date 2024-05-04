#!/bin/bash

# Download the dataset from the server
scp -i /home/helix/.ssh/rms dgrzinic@gmn.uwo.ca:mldataset.tar.bz2 .
rm mldataset.tar
bzip2 -d mldataset.tar.bz2 && tar -xf mldataset.tar
