1. install poetry
2. `poetry config virtualenvs.in-project true`
2. `poetry install`
3. If using VSCode, select interpreter "PATH_TO_REPO/compute-club-fork/data_exploration/.venv/bin/python3.11"
4. (also select the same .venv kernel in top right)

## TODO
1. Read Dataset audit paper: <https://aclanthology.org/2022.tacl-1.4.pdf>
2. turn code from notebooks to package for evaluation 
3. evaluate datasets:
- topic clustering + word frequency
- language detection
- profanity
- Atlas
- Chat datasets: MCL 

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
  - TinyStories: <https://arxiv.org/abs/2305.07759>
  - OpenSubtitles: <https://huggingface.co/datasets/open_subtitles>
  - Open Chat: <https://together.ai/blog/openchatkit>
  - Baize: <https://github.com/project-baize/baize-chatbot>
- General:
  - The Pile books <https://pile.eleuther.ai/>
  - Gutenberg <https://github.com/pgcorpus/gutenberg>
  - RefinedWeb <https://huggingface.co/datasets/tiiuae/falcon-refinedweb>
  - OpenWebText2: <https://openwebtext2.readthedocs.io/en/latest/>
  - 
