
# Generating Pokemon

spell upload data/ --name poke_gan

spell run python poke_gan.py -t V100 -m uploads/poke_gan

spell cp runs/<run number>
