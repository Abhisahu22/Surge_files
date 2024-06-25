# Surge_files
To run the sript - 
1. Copy the files in each node. Create and  activate the virtual environent using "python -m venv .venv" and
 " source .venv/bin/activate "
2. Run command on parent node which is our MASTER NODE :
        torchrun --nnodes=3 --nproc_per_node=2 --rdzv_id=234 --rdzv_backend=c10d --rdzv_endpoint="localhost:1234" Multinode_Resnet.py 10 10
3. On child nodes run :
        torchrun --nnodes=3 --nproc_per_node=1 --rdzv_id=234 --rdzv_backend=c10d --rdzv_endpoint="id:1234" Multinode_Resnet.py 10 10
   //here id is parent node ip address

   ---- After this the multinode-training will get start---
