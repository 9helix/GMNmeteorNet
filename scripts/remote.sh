#!/bin/bash
# server-side dataset extracting and archiving
cd ~
taskset -c 0 python GMNmeteorNet/MLdataset_extract.py
rm mldataset.tar.bz2
tar -cjf mldataset.tar.bz2 mldataset