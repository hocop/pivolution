# Evolution simulator

My artificial life/evolution simulator.

Features of this simulator:
* Possibly smart creatures
    * Behaviour is defined by a recurrent neural network
    * Can be herbavores, carnivores or omnivores, depending on one gene
    * Neural network weights are defined by genes
    * Two-gendered and one-gendered creatures are supported
    * Gene recombination like in humans (diploid organisms, meiosis, cross over and stuff)
* Designed for Huge worlds
    * Simple grid based movement mechanics
    * Implemented in python with highly efficient numpy operations (both mechanics and rendering)
    * Runs on multiple CPU cores
    * World is generated procedurally
* Simple architecture
    * CLI backend in python
    * Web browser frontend in VUE.js

>Main goals:
>* See population-level evolution instead of individial
>* See complicated behaviours (sexual reproduction, swarm intelligence)
>* Achieve this without any hard coded rules or restrictions

Screenshot:
![](example.png)

In this screenshot:
* Area colors
    * Grey - ground.
    * Blue - water (less energy for herbavores, higher movement cost).
    * Dark grey - rocks. Noone can go there.
* Dots
    * Green - herbavores. Get a little energy from nowhere.
    * Red - carnivores. Get a lot of energy from meat.
    * Yellow - (and other colors between green and red) omnivores. Get energy from nowhere, but less efficient. Get energy from meat same as carnivores. Weaker than carnivores (yes, they can fight).
    * Purple - meat, laying on the ground. Herbavores get less energy if there is meat in their cell.
* Splits
    * Map is split in 4 parts, each is run by a separate [process](https://docs.python.org/3/library/multiprocessing.html#the-process-class).
    * Cyan lines - portals. Creatures can pass through them. They are made small in order to limit I/O between processes.
    * Processes are not syncronised. Time goes faster in places where there are less creatures.

> ### Performance
> This screenshot has only 4 subworlds. There are around 12 000 creatures. In practice, there can be as many subworlds as the number of CPU's on the machine. Simulation with this many creatures usually runs at around 5-10 steps per second.

## How to run
Start backend:

```
cd backend
python app_simple.py
```

Start frontend server:

```
cd frontend
npm run dev
```

Open http://127.0.0.1:3000 in your browser and watch the game play itself.

## To be made in future
* docker image
* more complicated game mechanics