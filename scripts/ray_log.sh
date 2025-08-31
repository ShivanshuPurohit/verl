#!/bin/bash

rm -f ray_all.log
PREFIX=/tmp/ray/session_latest/logs
for f in $(ls $PREFIX)
do
  cat $PREFIX/$f >> ray_all.log
done
