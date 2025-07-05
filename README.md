# River Raid Bot

```python
STUDENTS = ["Bárbara Pereira", "Gabriel de Paula", "Pedro Machado"]
SUBJECT = "Robótica Computacional"
SEMESTER = "2025/1"
PROFESSOR = "Marcos Laia"
```

[**`> See the documentation (pt-BR) <`**](./docs/documentation-ptbr.pdf)

&nbsp;

## 👾 About the game

River Raid is a video game for the Atari 2600 console, created by Carol Shaw in 1982. The player controls a plane flying over a river filled with enemy military targets, such as planes, helicopters, ships, and bridges. The goal is to achieve the highest score possible while managing limited fuel and destroying enemies.

&nbsp;

## 🖥️ Run the game (Tested on Linux)

1) Crete a python environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2) Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3) Run the game:

    ```bash
    python src/main.py
    ```

    You can also run the program with custom parameters by passing them using flags:

    |Parameter|Flag|Type|Default|
    |:-|:-|:-|:-|
    |ROM path|`--rom`|string|`./river-raid.a26`|
    |State path|`--state`|string|none|
    |FPS|`--fps`|int|`60`|
    |Q-Learning path|`--qlearning`|string|`./qlearning.pck`|

    Example defining parameters

    ```bash
    python main.py --rom ./river-raid.a26 --state ./states/saved_state-3.bin --fps 30
    ```
