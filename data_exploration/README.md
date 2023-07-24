### Env setup

1. install (poetry)[https://python-poetry.org/docs/] - this manages python packages
2. `$ poetry config virtualenvs.in-project true`
3. `$ poetry install`
4. `$ poetry shell`
5. Create a (Nomic Atlas)[https://nomicai-production.us.auth0.com/u/login?state=hKFo2SBhZzhnTUVqd0ZKYUxDUFBsMzdMTm5pY05lWFRzcTVqb6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIHNWMl92NjJyWU45djhOeGRPaVQyY1BDZVB1cUxWNmd2o2NpZNkgVkY0MURxZEV5UzJBYXE2NHExSW9PMUVPemRwanBsbnY] account
6. `$ nomic login <token>` -> get the token from your nomic atlas account

**To run notebook in VSCode**

1. From VSCODE hit `CMD+SHIFT+P` -> "Python: Select Interpreter"
2. Add this path: "<PATH_TO_REPO>/compute-club-fork/data_exploration/.venv/bin/python3.11"
3. When you open a notebook, click "select kernel" on the top right, then select the .venv kernel

### To add a new package

`poetry add <package_name>` rather than `pip install`

## TODO

1. Read Dataset audit paper: <https://aclanthology.org/2022.tacl-1.4.pdf>
2. turn code from notebooks to package for evaluation
3. evaluate datasets:

- topic clustering + word frequency
- language detection
- profanity
- Atlas
- Chat datasets: MCL

### Relevant datasets

- Soda: <https://arxiv.org/abs/2212.10465>
- Q&A:
  - ShareGPT: <https://huggingface.co/datasets/theblackcat102/sharegpt-english>
  - Orca50k-flagged: <https://huggingface.co/datasets/teknium/orca50k-flagged>
  - GPTeacher: <https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct>
  - GPT4-LLM-Cleaned: <https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned>
  - Alpaca: <https://huggingface.co/datasets/tatsu-lab/alpaca>
  - Vicuna: <https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered>
  - WizardLM: <https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k>
  - Blenderbot: <https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/bb3/data_card.md>
- Dialog
  - Chai: <https://dataset-ideas.tiiny.site/>
  - Discord: <https://discord.com/channels/1104020730678612001/1112847498369830964/1112852844899487774>
  - Empathy dialogs: <https://twitter.com/hyunw__kim/status/1605400321840189440>
  - Open Assistant: <https://huggingface.co/datasets/OpenAssistant/oasst1>
  - D&D: <https://huggingface.co/datasets/crd3>
  - IMSDB: <https://imsdb.com/genre/Fantasy>
  - Final Fantasy Dialogue: <https://www.kaggle.com/datasets/tylerhuxtable/final-fantasy-dialogue-scripts>
  - NSFW: <https://huggingface.co/datasets/Oniichat/bluemoon_roleplay_chat_data_300k_messages>
  - e Pile books <https://pile.eleuther.ai/>
  - Gutenberg <https://github.com/pgcorpus/gutenberg>
  - RefinedWeb <https://huggingface.co/datasets/tiiuae/falcon-refinedweb>
  - OpenWebText2: <https://openwebtext2.readthedocs.io/en/latest/>
  - TinyStories: <https://arxiv.org/abs/2305.07759>
  - OpenSubtitles: <https://huggingface.co/datasets/open_subtitles>
  - Open Chat: <https://together.ai/blog/openchatkit>
  - Baize: <https://github.com/project-baize/baize-chatbot>
