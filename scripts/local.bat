:: Download the dataset from the server
cd ..
scp -i C:\Users\Dino\.ssh\rms dgrzinic@gmn.uwo.ca:mldataset.tar.bz2 .
bzip2 -d mldataset.tar.bz2 && tar -xf mldataset.tar
cd GMNmeteorNet