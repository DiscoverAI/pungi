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
