dependencies:
python 3.11
conda
python-dotenv

Run this command to create a new environment:
bash
conda env create -f environment.yml

Activate the environment:
bash
conda activate geo_env_311

run test.py:
bash
cd geo_backend
cd .. # go back to top level directory that can see full geo_backend folder
python -m geo_backend.test

