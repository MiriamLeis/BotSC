# [TFG] Artifical Intelligence for Starcraft II
Authors:
* Miriam Leis Baltanás
* Pablo Joaquín Rodríguez Hidalgo

# Start Guide
This project uses PySC2 and Starcraft II in addition to a collection of modified minimaps for Starcraft II, which are included on this project.

## Get PySC2
*The follow instructions can be found on [PySC2](https://github.com/deepmind/pysc2 "PySC2 repository") repository*
### PyPI
The easiest way to get PySC2 is to use pip:
```
$ pip install pysc2
```
You may also need to upgrade pip: `pip install --upgrade pip` for the `pysc2` install to work. If you're running on an older system you may need to install `libsdl` libraries for the `pygame` dependency.
### From Source
You can install latest PySC2 codebase from git master branch:
```
$ pip install --upgrade https://github.com/deepmind/pysc2/archive/master.zip`
```
Or from a local clone of the git repository:
```
$ git clone https://github.com/deepmind/pysc2.git
$ pip install --upgrade pysc2/
```

## Get Starcraft II
*The follow instructions can be found on [PySC2](https://github.com/deepmind/pysc2 "PySC2 repository") repository*
### Linux
Follow [Blizzard's documentation](https://github.com/Blizzard/s2client-proto#downloads) to get the linux version. By default, PySC2 expects the game to live in `~/StarCraftII/`. You can override this path by setting the `SC2PATH` environment variable or creating your own run_config.
### Windows/MacOS
Our project depends on the full StarCraft II game and only works with versions that include the API, which is 3.16.1 and above.
The easiest way to get Starcraft II is to install it from [Battle.net](https://eu.shop.battle.net/es-es). Even the [Start Edition](https://starcraft2.com/es-es/) will work. If you changed the install location, you might need to set the `SC2PATH` environment variable with the correct location.

## Get Minigames
You have modified minigames used in this project on `mini_games\` folder. Copy this folder into your `Maps\` folder from your Starcraft II installation directory.
The default installation directories are:
* Windows: `C:\Program Files (x86)\StarCraft II\`
* Mac: `/Applications/StarCraft II/`
* Linux: the installation directory is the folder you extracted the linux package into.

Maybe some folders are missing. Create `Maps` folder if it is necessary.
Also you need to add these minigames in PySC2. To do that, open `pysc2\maps\mini_games.py` and add to minigames array the following minigames names:
```python
mini_games = [ ##Now you add this few lines 
    "MoveToBeacon",  # 120s   
    "DefeatZealotswithBlink", # 120s
    "DefeatZealotswithBlink_2enemies", # 120s
    "DefeatZealotswithBlink_2vs2", # 120s
    "BuildMarines", # 900s
]
```
