# pixel_brain

Cellular automaton simulation that generates an animated GIF.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Run

```bash
.venv/bin/pixel-brain
```

Or activate the virtual environment first so you don't need the prefix:

```bash
source .venv/bin/activate
pixel-brain
```

Activation adds `.venv/bin` to your shell's PATH for the current terminal session. It wears off when you close the terminal or run `deactivate`.

## Output

Saves `pixel_cell_m1.gif` in the current directory.
