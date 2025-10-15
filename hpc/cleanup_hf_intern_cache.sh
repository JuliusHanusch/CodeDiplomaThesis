#!/usr/bin/env bash

# HF create its own (for us useless) cachefiles those cluter our cache directories with every access
# This script deletes those

# go to your cache directory
cd cache/data || exit 1

# find and delete all files matching cache-*.arrow in subfolders
find . -type f -name "cache-*.arrow" -delete

cd ../..

cd data/data_sets_raw || exit 1

# find and delete all files matching cache-*.arrow in subfolders
find . -type f -name "cache-*.arrow" -delete