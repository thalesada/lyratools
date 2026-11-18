timeseries writes hdf5 files with aggregate data over time

first run a dry run:

python $LT/scripts/lyra_timeseries.py --snapdir $OUTPUT --groupdir $OUTPUT --out $OUT --halo 0 --dry-run | tee dryrun_zhalo1.txt

then choose how many time R_200 the subhalo tracker should look beyond the main halo to see subhalos before the fall in. Also choose the min N of DM particles that is considered a subhalo. These can be written to a config:

cat > $OUT.config.json <<'EOF'
{"sub": {"R_sel_R200": 3.0, "min_dm": 100}}
EOF

test one snapshot:

python $LT/scripts/lyra_timeseries.py --snapdir $OUTPUT --groupdir $OUTPUT --out $OUT --sim zhalo1 --config $OUT.config.json --snaps 39:39 --nproc 1

Then submit to queue:

#!/bin/bash
#SBATCH --job-name=ts_zhalo1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<cores>
#SBATCH --mem=<GB>G
#SBATCH --time=<hh:mm:ss>
#SBATCH --output=ts_zhalo1_%j.out

export LT=<path>/lyratools
export LYRATOOLS_DATA=<path>/lyra_data
export OUT=<results>/timeseries_zhalo1
export OUTPUT=<path>/zhalo1/output
export OMP_NUM_THREADS=1
source <your env activate>

python $LT/scripts/lyra_timeseries.py --snapdir $OUTPUT --groupdir $OUTPUT --out $OUT --sim zhalo1 --config $OUT.config.json --nproc auto


At the end, the snap data is combined into time arrays. This can be done on its own as well:

python $LT/scripts/lyra_timeseries.py --snapdir $OUTPUT --groupdir $OUTPUT --out $OUT --sim zhalo1 --config $OUT.config.json --combine-only