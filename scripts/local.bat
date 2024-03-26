:: Download the dataset from the server
scp -i C:\Users\Dino\.ssh\rms dgrzinic@gmn.uwo.ca:mldataset.tar.bz2 .
bzip2 -d mldataset.tar.bz2 && tar -xf mldataset.tar
del mldataset.tar