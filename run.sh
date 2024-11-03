# this script should start the following python scripts and processes.
# first it should start a redis server
# then it should start the dataset_sampler.py
# then the batch_processor.py
# then the file_writer.py
# and then it should start n instances of mcts_sampler.py

# start redis server
sudo apt-get install redis
pip install redis
pip install datasets

redis-server &
sleep 5

# start dataset_sampler.py
python3 dataset_sampler.py &
# start batch_processor.py
python3 batch_processor.py &
# start file_writer.py
python3 file_writer.py &
sleep 30
# start 4 instances of mcts_sampler.py
for i in {1..112}
do
    python3 mcts_sampler.py &
    sleep 0.05
done

