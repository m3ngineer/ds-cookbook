# Style Transfer

This script can be run in Spell using the following commands:

## Upload data to Spell
`spell upload data/ --name style_transfer`

## Run program
`spell run 'python style_transfer.py --content <IMAGE NAME> --style <IMAGE NAME>' -t V100 -m uploads/style_transfer`

# Download output from run
`spell cp runs/<run number>``

# To remove uploads from Spell
`spell rm uploads/style_transfer`
