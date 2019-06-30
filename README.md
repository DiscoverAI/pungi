# pungi

[![CircleCI](https://circleci.com/gh/DiscoverAI/pungi.svg?style=svg)](https://circleci.com/gh/DiscoverAI/pungi)

> The pungi (Hindi: पुंगी) \[...\] is a wind instrument played by snake charmers on the Indian subcontinent.
> (from https://en.wikipedia.org/wiki/Pungi)

A collection of reinforcement learning agents that learn how to play snake.

## Install Dependencies
```bash
pipenv install
```
will install the production dependencies from the Pipfile.

```bash
pipenv install --dev
```
will install all the dependencies including dev dependencies from the Pipfile.

## Run tests
```bash
pipenv run pytest
```

## Train the agent
```bash
pipenv run pungi/main.py train
```

## Compute average cumulative reward and write to metrics file
```bash
pipenv run pungi/main.py eval out/model-xy.json
```

## Play a test game
```bash
pipenv run pungi/main.py play out/model-xy.json
```

## Run with batect
[Batect](https://github.com/charleskorn/batect)
allows to run a development setup in a containerized environment.
If you don't want to install anything except some docker images
to run pungi, use these commands.

```bash
./batect run -- train
./batect run -- play out/model-xy.pkl
./batect run -- eval out/model-xy.pkl
./batect test
```

Everything works from batect the same as local, you just need to update the docker images if you update the
project dependencies. One exception is the `play` command, which requires you to manually browse to `http://localhost:8080/?spectate_game_id=<G_xyz>` after you started the program.
Batect uses pre-built docker images from dockerhub. You can change that in the file `batect.yml`.
