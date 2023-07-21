## Getting started with `lmgym_train`

1. [Connect](https://www.notion.so/Provisioning-logging-into-compute-resources-5a8328da4d084b75930e5ae1327b0d07?pvs=4) to a Lambda instance with a GPU (A10 or H100). Then, SSH into the machine.
2. Clone the repo `$ git clone https://github.com/compute-club/compute-club-fork.git`
3. `$ cd compute-club-fork/chai_research`
4. `$ touch .env`
5. Copy the values from `.env.template` into the `.env` file you just created, and add the necessary values
6. Build & run the container: `sudo docker compose up --build`

At this point, the training should begin. You will receive URLs to the WANDB and HuggingFace pages to track the progress of this training run. They will be logged to the console, and will look like this: 

```
wandb: ⭐️ View project at https://wandb.ai/markrich/huggingface
wandb: 🚀 View run at https://wandb.ai/markrich/huggingface/runs/a2uoncdd
```

To check the GPU utilization, in a separate terminal on the same machine, run: `$ watch nvidia-smi`