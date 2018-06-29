#! /bin/bash

TIME_DATE=`date "+%Y-%m-%d_%H-%M-%S"`

./run_benchmark.sh ./graphs_benchmark 0 20 graphs_benchmark__cudaTT__r415__gp102__${TIME_DATE}

