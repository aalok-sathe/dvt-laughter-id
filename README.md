# dvt-laughter-id

<!-- Notes for Spring 2019. -->

## installation

1. Download the repository
    - ssh:
    ```bash
    git clone git@gitlab.com:aalok-sathe/dvt-laughter-id.git
    ```
    - https:
    ```bash
    git clone https://gitlab.com/aalok-sathe/dvt-laughter-id.git
    ```

2. Get the dependencies
```bash
python3 -m pip install -r requirements.txt [--user] [--upgrade]
```

3. Make sure you have raw data OR download pre-computed embeddings

    - If you have access to _friends_ episodes,
    please place them in `video/` (must be `.mp4`)
    - If you do not have access to _friends_
    episodes, you would need to download pre-computed
    embeddings for the audio
    (not available to download yet)

## usage

1. If you have `.mp4` files, generate `.wav` audio
data from them (requires `ffmpeg`)
    ```bash
    ./utils/mp4towav.sh
    ```
    This will generate output in the directory
    `wav/`.

2. Once you have the audio files, you are good to run
the notebooks in `laughter/`. In case you did not
have the `.mp4` and so the `.wav` files, you would
need to place the embeddings you donwloaded in
`data/archive/` for the notebooks and scripts to be
able to use them

Look at the examples in the notebooks for usage.
Take a look at the code in `utils/` (particularly,
`episode.py`) for additional details. Most of the
implemented methods in scripts under `utils/` have
ample documentation within code.
