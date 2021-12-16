# Style Transfer

This script can be run in Spell using the following commands:

`spell upload data/ --name style_transfer`

`spell run python style_transfer.py -t V100 -m uploads/style_transfer --content <IMAGE NAME> --style <IMAGE NAME>`

`spell cp runs/<run number>``

# To remove uploads from Spell
`spell rm uploads/style_transfer`
