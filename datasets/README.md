TODO:
- run as command line argument
arguments:
- # rows
- min convo length
- max convo length
- template:
    - description
    - prompt
    - memory
    - bot lable
    - user label 
- examples: (less than 10) (json)
- dataset name 

- login to HF 
- randomize prompt
- log progress

output:
- uniformally distributed between min and max 
output: https://huggingface.co/datasets/AlekseyKorshuk/hh-lmgym-demo 
- ChatML format. introduction, prompt, memory -> system message
- first message -> user
- output -> from assistant 

output: Not CHATML format

## Getting started

```
virtualenv -p python3 env
source ./env/bin/activate
(source ./env/bin/activate.fish) if you use fish like me
```

`$ pip install -r requirements.txt`

when adding a new requirement:
`pip freeze > requirements.txt`

`$ huggingface-cli login`
