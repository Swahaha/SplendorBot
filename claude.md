# Splendor Game Implementation Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Game Rules & Mechanics](#game-rules--mechanics)
5. [Data Structures](#data-structures)
6. [API Reference](#api-reference)
7. [File Structure](#file-structure)
8. [Build System](#build-system)
9. [Python Bindings](#python-bindings)
10. [RL Integration Points](#rl-integration-points)

---

## Project Overview

This is a C++ implementation of the board game **Splendor** with Python bindings via pybind11. The game engine is designed for high performance and is suitable for training reinforcement learning agents.

**Key Features:**
- Fast C++ game engine
- Python bindings for easy integration with RL frameworks
- Complete game state management
- Legal move generation
- Move execution with validation
- Terminal state detection

**Game Goal:** Players collect gem tokens to purchase development cards that provide bonuses and prestige points. First player to reach 15 prestige points triggers the end game.

---

## Architecture

### High-Level Design

The codebase follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│         Python Interface (pybind11)         │
│              splendor_game module           │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│            SplendorGame (Core)              │
│         src/game_state.cpp                  │
└─────────────────────────────────────────────┘
         │           │            │
         ▼           ▼            ▼
┌─────────────┬─────────────┬────────────────┐
│   Move      │   Move      │   Game         │
│  Generator  │  Executor   │  Initializer   │
└─────────────┴─────────────┴────────────────┘
         │           │            │
         └───────────┴────────────┘
                     ▼
        ┌────────────────────────┐
        │   State Serializer     │
        │   Player State         │
        │   Types & Constants    │
        └────────────────────────┘
```

### Design Patterns

1. **Friend Classes**: Core game state uses friend classes for controlled access
2. **Static Utility Classes**: Move generation, execution, and serialization are static utility classes
3. **Separation of Concerns**: Each module has a single, well-defined responsibility

---

## Core Components

### 1. Game State (`game_state.h`, `game_state.cpp`)

**Location:** `include/game_state.h`, `src/game_state.cpp`

The central class managing all game state.

**Key Responsibilities:**
- Maintain game state (players, market, bank, nobles, decks)
- Coordinate between subsystems
- Provide public API for game interaction
- Check terminal conditions

**Private State:**
```cpp
int num_players_;                                    // 2-4 players
int current_player_;                                 // Active player index
std::array<uint8_t, COLOR_COUNT> bank_;              // Token bank
std::array<std::vector<Card>, TIERS> decks_;        // 3 card decks
std::array<std::array<Card, SLOTS_PER_TIER>, TIERS> market_;  // 3x4 market
std::vector<Noble> nobles_;                          // Active nobles
std::vector<PlayerState> players_;                   // Player states
std::mt19937_64 rng_;                                // Random number generator
```

**Friend Classes:**
- `GameInitializer` - Setup and initialization
- `MoveExecutor` - Execute moves
- `MoveGenerator` - Generate legal moves
- `StateSerializer` - Serialize state for Python

---

### 2. Player State (`player_state.h`, `player_state.cpp`)

**Location:** `include/player_state.h`, `src/player_state.cpp`

Encapsulates individual player state.

**Structure:**
```cpp
struct PlayerState {
    std::array<uint8_t, COLOR_COUNT> tokens;    // Token inventory (6 colors)
    uint8_t prestige_points;                    // Victory points
    std::vector<Card> reserved;                 // Reserved cards (max 3)
    std::vector<Card> played_cards;             // Purchased cards
    std::vector<Noble> nobles_owned;            // Nobles acquired

    int total_tokens() const;                   // Sum of all tokens
    std::array<uint8_t, DEV_COLORS> get_bonuses() const;  // Permanent discounts
};
```

**Key Methods:**
- `total_tokens()`: Returns total token count (for hand limit validation)
- `get_bonuses()`: Returns permanent discounts from played cards

---

### 3. Move Generator (`move_generator.h`, `move_generator.cpp`)

**Location:** `include/move_generator.h`, `src/move_generator.cpp`

Generates all legal moves for the current player.

**Move Types Generated:**
1. **Buy from Market** - Purchase visible cards
2. **Buy from Reserve** - Purchase reserved cards
3. **Reserve Card** - Reserve a card from market (max 3 reserved)
4. **Take 3 Different Tokens** - 3 different colored tokens
5. **Take 2 Same Tokens** - 2 of same color (requires 4+ in bank)

**Key Logic:**
- Validates affordability (tokens + bonuses + gold)
- Checks hand limits (10 tokens max)
- Checks reserve limits (3 cards max)
- Validates bank availability

**Code Reference:** `src/move_generator.cpp:7-137`

---

### 4. Move Executor (`move_executor.h`, `move_executor.cpp`)

**Location:** `include/move_executor.h`, `src/move_executor.cpp`

Executes validated moves and updates game state.

**Move Execution Flow:**
1. Parse move tuple from Python
2. Execute specific action
3. Update player state
4. Update game state (bank, market, decks)
5. Draw replacement cards if needed
6. Check nobles (TODO: currently skipped)
7. End turn

**Key Methods:**
- `PerformMove()` - Main entry point
- `ReserveCard()` - Reserve from market, gain gold token
- `BuyCardFromMarket()` - Purchase and draw replacement
- `BuyCardFromReserve()` - Purchase from reserves
- `CanAfford()` - Affordability check with discounts

**Payment Calculation:**
```cpp
// For each color, calculate actual cost after discounts
int required = max(0, card.cost[color] - discounts[color]);
// If player lacks colored tokens, use gold
if (player.tokens[color] < required) {
    gold_needed += required - player.tokens[color];
}
```

**Code Reference:** `src/move_executor.cpp:1-150`

---

### 5. Game Initializer (`game_initializer.h`, `game_initializer.cpp`)

**Location:** `include/game_initializer.h`, `src/game_initializer.cpp`

Handles game setup and initialization.

**Initialization Sequence:**
1. **Bank** - Token counts based on player count (2p=4, 3p=5, 4p=7) + 5 gold
2. **Decks** - Load from CSV, organize by tier, shuffle
3. **Market** - Draw 4 cards per tier (3 tiers)
4. **Nobles** - Select N+1 nobles (N = player count)
5. **Players** - Initialize empty player states

**Token Distribution:**
| Players | Regular Tokens | Gold Tokens |
|---------|---------------|-------------|
| 2       | 4 each color  | 5           |
| 3       | 5 each color  | 5           |
| 4       | 7 each color  | 5           |

**Code Reference:** `src/game_initializer.cpp:8-61`

---

### 6. State Serializer (`state_serializer.h`, `state_serializer.cpp`)

**Location:** `include/state_serializer.h`, `src/state_serializer.cpp`

Converts C++ game state to Python dictionaries.

**Serialized State Structure:**
```python
{
    'current_player': int,
    'bank': [int, int, int, int, int, int],  # 6 colors
    'market': [
        [Card, Card, Card, Card],  # Tier 1
        [Card, Card, Card, Card],  # Tier 2
        [Card, Card, Card, Card]   # Tier 3
    ],
    'nobles': [Noble, Noble, ...],
    'players': [
        {
            'tokens': [int, int, int, int, int, int],
            'prestige_points': int,
            'reserved': [Card, ...],
            'played': [Card, ...],
            'nobles': [Noble, ...]
        },
        ...
    ]
}
```

**Card Dictionary:**
```python
{
    'prestige_points': int,
    'bonus_color': int,  # 0=WHITE, 1=BLUE, 2=GREEN, 3=RED, 4=BLACK
    'cost': [int, int, int, int, int],  # W, B, G, R, BK
    'tier': int,  # 1, 2, or 3
    'id': int,
    'valid': bool
}
```

---

### 7. CSV Parser (`csv_parser.h`, `csv_parser.cpp`)

**Location:** `include/csv_parser.h`, `src/csv_parser.cpp`

Parses card data from CSV file.

**CSV Format:** `data/cards.csv`
```
Tier,Color,Prestige,White,Blue,Green,Red,Black
1,black,,1,1,1,1,
```

**Parsing Logic:**
- Skips header row
- Assigns unique card IDs
- Handles empty fields (prestige, costs)
- Maps color strings to enum values
- Error handling with line skipping

**Code Reference:** `src/csv_parser.cpp:7-52`

---

## Game Rules & Mechanics

### Win Condition
- First player to reach **15 prestige points** triggers end game
- All players get equal turns
- Highest prestige wins (tiebreakers: fewest cards)

### Prestige Points Sources
1. **Development Cards** - Some cards have prestige points
2. **Nobles** - Worth 3-4 points each (TODO: noble acquisition not implemented)

### Actions (Choose One Per Turn)

#### 1. Take Tokens
**Option A: Take 3 Different Colors**
- Must be 3 different colors
- Each color must be available in bank
- Cannot exceed 10 token hand limit

**Option B: Take 2 Same Color**
- Must have 4+ tokens of that color in bank
- Cannot exceed 10 token hand limit

**Gold Tokens:**
- Not available as regular action
- Only gained when reserving a card (if available)
- Act as wildcards when purchasing

#### 2. Reserve a Card
- Take a card from market into your hand
- Gain 1 gold token (if available and under hand limit)
- Card is replaced from deck
- **Limit: 3 reserved cards per player**

#### 3. Purchase a Card (Market or Reserve)
- Pay token cost (can use bonuses + gold as wildcard)
- Gain the card (permanent discount + prestige)
- Card replaced if from market

### Bonuses & Discounts
- Each purchased card provides **1 permanent discount** of its color
- Discounts reduce token cost when buying cards
- Example: Card costs 3 blue, you have 2 blue bonuses → pay only 1 blue token

### Token Hand Limit
- Maximum **10 tokens** per player
- Must be enforced on token-taking actions
- Gold counts toward this limit

### Market & Decks
- **3 tiers** of cards (1=cheap, 2=medium, 3=expensive)
- **4 slots** per tier visible in market
- When a card is purchased/reserved, immediately draw replacement
- When deck empty, no replacement drawn (market slot becomes empty)

### Nobles (TODO: Not Fully Implemented)
- Nobles visit players who meet their requirements
- Requirements are based on **bonuses** (played cards), not tokens
- Worth prestige points
- **N+1 nobles** for N players

**Current Status:** Noble checking is commented out in `move_executor.cpp:53`

---

## Data Structures

### Constants (`constants.h`)

**Location:** `include/constants.h`

```cpp
constexpr int DEV_COLORS = 5;        // White, Blue, Green, Red, Black
constexpr int COLOR_COUNT = 6;       // DEV_COLORS + Gold
constexpr int NUM_NOBLES = 5;        // Nobles in play
constexpr int TIERS = 3;             // Card tiers
constexpr int SLOTS_PER_TIER = 4;    // Market slots per tier
constexpr int RESERVED_LIMIT = 3;    // Max reserved cards
constexpr int TOKEN_HAND_LIMIT = 10; // Max tokens in hand
constexpr int PRESTIGE_POINTS_TO_WIN = 15;

enum Color : uint8_t {
    WHITE=0, BLUE=1, GREEN=2, RED=3, BLACK=4, GOLD=5
};

enum ACTIONS : uint8_t {
    RESERVE_CARD=0,
    BUY_CARD_FROM_MARKET=1,
    BUY_CARD_FROM_RESERVE=2,
    GET_3_FROM_BANK=3,
    GET_2_FROM_BANK=4
};
```

### Card (`types.h`)

**Location:** `include/types.h:9-23`

```cpp
struct Card {
    uint8_t prestige_points;           // Victory points
    Color bonus_color;                 // Permanent discount color
    std::array<uint8_t, DEV_COLORS> cost;  // Token cost [W,B,G,R,BK]
    uint8_t tier;                      // 1, 2, or 3
    uint8_t id;                        // Unique identifier
    bool valid;                        // Is card present (or empty slot)

    bool operator==(const Card& other) const;
};
```

**Card Stats by Tier:**
- **Tier 1:** 40 cards, costs 0-4 tokens, 0-1 prestige
- **Tier 2:** 30 cards, costs 2-7 tokens, 1-3 prestige
- **Tier 3:** 20 cards, costs 7-10 tokens, 3-5 prestige

### Noble (`types.h`)

**Location:** `include/types.h:25-31`

```cpp
struct Noble {
    std::array<uint8_t, DEV_COLORS> req;  // Bonus requirements [W,B,G,R,BK]
    uint8_t prestige_points;               // Usually 3 or 4
};
```

**Noble Requirements:** Typically require 3-4 bonuses of specific colors

---

## API Reference

### Python API (via pybind11)

**Module:** `splendor_game`

#### Constructor
```python
game = splendor_game.SplendorGame(num_players: int, seed: int = 0)
```
- `num_players`: 2-4 players
- `seed`: Random seed for deterministic games

#### Methods

**`num_players() -> int`**
- Returns number of players in game

**`current_player() -> int`**
- Returns index of current player (0-indexed)

**`state_summary() -> dict`**
- Returns complete game state as nested dictionary
- See [State Serializer](#6-state-serializer-state_serializerh-state_serializercpp) for structure

**`legal_moves() -> list[tuple]`**
- Returns list of legal moves for current player
- Each move is a tuple: `(action_type: int, payload: dict)`

**Move Format Examples:**
```python
# Reserve card from market
(0, {'tier': 1, 'slot': 2})

# Buy card from market
(1, {'tier': 2, 'slot': 0})

# Buy card from reserve
(2, {'index': 1, 'card': {...}})

# Take 3 different tokens (W, B, G)
(3, [1, 1, 1, 0, 0])

# Take 2 same tokens (Red)
(4, [0, 0, 0, 2, 0])
```

**`perform_move(move: tuple) -> None`**
- Executes a move and advances to next player
- Automatically calls `end_turn()`
- Move must be in the format returned by `legal_moves()`

**`is_terminal() -> bool`**
- Returns True if game is over (player reached 15 points)

**`end_turn() -> None`**
- Advances to next player
- Called automatically by `perform_move()`

---

## File Structure

```
SplendorBot/
├── include/               # Header files
│   ├── constants.h        # Game constants and enums
│   ├── types.h            # Card and Noble structures
│   ├── player_state.h     # Player state structure
│   ├── game_state.h       # Main game class
│   ├── game_initializer.h # Initialization logic
│   ├── move_generator.h   # Legal move generation
│   ├── move_executor.h    # Move execution
│   ├── state_serializer.h # State to Python conversion
│   └── csv_parser.h       # CSV parsing utilities
│
├── src/                   # Implementation files
│   ├── types.cpp
│   ├── player_state.cpp
│   ├── game_state.cpp
│   ├── game_initializer.cpp
│   ├── move_generator.cpp
│   ├── move_executor.cpp
│   ├── state_serializer.cpp
│   ├── csv_parser.cpp
│   └── bindings.cpp       # pybind11 bindings
│
├── data/                  # Game data
│   ├── cards.csv          # 90 development cards
│   └── nobles.csv         # 10 nobles
│
├── python/                # Python utilities
│   ├── script.py          # Example usage and testing
│   └── splendor_game.*.so # Compiled Python module
│
├── RL/                    # RL agent directory (empty)
│   └── main.py
│
├── build/                 # CMake build directory
├── CMakeLists.txt         # CMake configuration
├── Makefile               # Build automation
└── requirements.txt       # Python dependencies
```

---

## Build System

### CMake Configuration (`CMakeLists.txt`)

**Location:** `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.12)
project(SplendorGame)

set(CMAKE_CXX_STANDARD 20)
find_package(pybind11 REQUIRED)

include_directories(include)

pybind11_add_module(splendor_game
    src/player_state.cpp
    src/game_state.cpp
    src/game_initializer.cpp
    src/move_executor.cpp
    src/move_generator.cpp
    src/state_serializer.cpp
    src/types.cpp
    src/csv_parser.cpp
    src/bindings.cpp
)

target_compile_options(splendor_game PRIVATE -Wall -Wextra)
```

### Building

**Standard Build:**
```bash
mkdir -p build
cd build
cmake ..
make
```

**Output:** `build/splendor_game.*.so` (platform-specific extension)

**Using from Python:**
```python
import sys
sys.path.append('build')  # or 'python' if copied there
import splendor_game
```

### Dependencies
- **C++20** compiler
- **pybind11** - Python binding library
- **CMake** 3.12+

**Python Dependencies** (`requirements.txt`):
```
pybind11
termcolor
```

---

## Python Bindings

### Binding Code (`bindings.cpp`)

**Location:** `src/bindings.cpp`

```cpp
PYBIND11_MODULE(splendor_game, m) {
    m.doc() = "Splendor game implementation.";

    py::class_<SplendorGame>(m, "SplendorGame")
        .def(py::init<int, uint64_t>(),
             py::arg("num_players"),
             py::arg("seed") = 0)
        .def("num_players", &SplendorGame::num_players)
        .def("current_player", &SplendorGame::current_player)
        .def("end_turn", &SplendorGame::end_turn)
        .def("state_summary", &SplendorGame::state_summary)
        .def("legal_moves", &SplendorGame::legal_moves)
        .def("perform_move", &SplendorGame::perform_move)
        .def("is_terminal", &SplendorGame::is_terminal);
}
```

### Example Usage (`python/script.py`)

**Location:** `python/script.py`

```python
import splendor_game as s

# Create game
game = s.SplendorGame(num_players=3, seed=42)

# Get state
state = game.state_summary()

# Get legal moves
moves = game.legal_moves()

# Perform move
game.perform_move(moves[0])

# Check if game over
if game.is_terminal():
    print("Game Over!")
```

**Utility Functions in script.py:**
- `print_summary_nicely(summary)` - Pretty print game state
- `print_legal_moves_nicely(moves)` - Pretty print moves by type

---

## RL Integration Points

### For Training RL Agents

The game engine is designed to support RL training with the following features:

#### 1. Fast State Access
```python
state = game.state_summary()
# Access specific parts:
bank = state['bank']
current_player_state = state['players'][state['current_player']]
market = state['market']
```

#### 2. Action Space
```python
# Get all legal actions
legal_actions = game.legal_moves()

# Actions are tuples: (action_type, payload)
# 5 action types: RESERVE_CARD, BUY_MARKET, BUY_RESERVE, GET_3, GET_2
```

#### 3. State Representation Options

**Option A: Dictionary (Current)**
- Full state as nested dictionary
- Easy to inspect and debug
- May need conversion for neural networks

**Option B: Feature Vector (TODO)**
Could add method to convert state to fixed-size vector:
- Player tokens (6 × num_players)
- Player prestige (num_players)
- Player bonuses (5 × num_players)
- Market cards (12 cards × features)
- Bank state (6)
- Reserved cards state

**Option C: Tensor (TODO)**
Could add method for direct tensor output for deep learning

#### 4. Reward Signal
```python
# Track prestige points before/after move
prev_points = state['players'][player_idx]['prestige_points']
game.perform_move(action)
new_state = game.state_summary()
new_points = new_state['players'][player_idx]['prestige_points']
reward = new_points - prev_points

# Terminal reward
if game.is_terminal():
    # Determine winner and assign final reward
    points = [p['prestige_points'] for p in new_state['players']]
    winner = points.index(max(points))
    reward = 1.0 if winner == player_idx else -1.0
```

#### 5. Episode Management
```python
# Deterministic episodes with seed
game = s.SplendorGame(num_players=3, seed=episode_number)

# Run episode until terminal
while not game.is_terminal():
    state = game.state_summary()
    legal_moves = game.legal_moves()

    # Agent selects action
    action = agent.select_action(state, legal_moves)

    # Execute action
    game.perform_move(action)

    # Store transition for training
    next_state = game.state_summary()
    reward = calculate_reward(state, next_state)
    store_transition(state, action, reward, next_state)
```

#### 6. Multi-Agent Support
The game naturally supports multi-agent RL:
- 2-4 agents per game
- Turn-based interaction
- Shared environment state

#### 7. Performance Considerations
- **C++ Backend:** Fast move generation and execution
- **Minimal Python Overhead:** Only state serialization crosses Python/C++ boundary
- **Batch Processing:** Can run multiple game instances in parallel

### Recommended RL Algorithms

Given game characteristics:
- **Turn-based:** MCTS, AlphaZero-style approaches
- **Discrete actions:** DQN, PPO, A2C
- **Multi-agent:** Self-play, league training
- **Partial observability:** None (perfect information game)
- **Long episodes:** 50-200 turns typical

### Potential Improvements for RL

1. **Add state copying/cloning** for MCTS tree search
2. **Implement noble acquisition** for complete rules
3. **Add reward shaping** options (card acquisition, token efficiency)
4. **Vectorized state representation** for neural networks
5. **Batch action execution** for parallel environments
6. **Action masking utilities** for invalid action handling
7. **Game statistics** (turn count, card distribution, etc.)

---

## Key Implementation Notes

### Current Limitations & TODOs

1. **Noble Acquisition NOT Implemented**
   - Location: `src/move_executor.cpp:53`
   - Line: `// CheckNobles(game); WE SKIP THIS FOR NOW`
   - Impact: Players don't receive nobles even when qualified

2. **Reserve from Deck Top**
   - Location: `src/move_generator.cpp:94`
   - Line: `// TODO: Add reserve from deck moves`
   - Impact: Can only reserve visible cards, not from deck top

3. **Token Return on Overflow**
   - Not implemented: Returning tokens when exceeding hand limit
   - Current: Move generation prevents overflow moves

4. **Deep Copy / Clone**
   - Location: `SplendorGame.cpp:192-194`
   - Empty implementation, needed for MCTS

5. **No State History**
   - No tracking of previous states or move history
   - Would be useful for training and debugging

### Memory Layout

- **Efficient:** Uses `uint8_t` for small values
- **Fixed Arrays:** Market, bank use fixed-size arrays
- **Dynamic Vectors:** Players, cards use vectors
- **No Pointers:** Value semantics throughout

### Random Number Generation

- **RNG:** Mersenne Twister 64-bit (`std::mt19937_64`)
- **Seeded:** Deterministic with seed parameter
- **Usage:** Card shuffling, noble selection

### Error Handling

- **Validation:** Constructor validates player count (2-4)
- **Exceptions:** Throws `std::invalid_argument`, `std::runtime_error`
- **CSV Parsing:** Skips invalid lines with error messages

---

## Data Files

### Cards CSV (`data/cards.csv`)

**Format:**
```
Tier,Color,Prestige,White,Blue,Green,Red,Black
1,white,1,,,4,,
```

**Statistics:**
- **90 total cards**
- **Tier 1:** 40 cards
- **Tier 2:** 30 cards
- **Tier 3:** 20 cards

**Color Distribution:** Each tier has 8 cards per color

### Nobles CSV (`data/nobles.csv`)

**Format:**
```
Name, Prestige, White, Blue, Green, Red, Black
Mary Stuart, 3, 0, 0, 4, 4, 0
```

**Statistics:**
- **10 total nobles**
- **5 nobles worth 3 points** (3 cards of 2 colors)
- **5 nobles worth 4 points** (3 cards of 3 colors)

**Note:** Nobles CSV is loaded but currently not used during gameplay

---

## Testing & Debugging

### Example Test Script

**Location:** `python/script.py:61-69`

```python
# Quick playthrough
game = s.SplendorGame(num_players=3, seed=42)

for _ in range(9):
    game.perform_move(game.legal_moves()[0])

# Inspect final state
summary = game.state_summary()
print_summary_nicely(summary)
```

### Debugging Tips

1. **State Inspection:**
   ```python
   state = game.state_summary()
   import json
   print(json.dumps(state, indent=2))
   ```

2. **Move Validation:**
   ```python
   moves = game.legal_moves()
   print(f"Player {game.current_player()} has {len(moves)} legal moves")
   ```

3. **Token Tracking:**
   ```python
   state = game.state_summary()
   bank = state['bank']
   player_tokens = state['players'][0]['tokens']
   print(f"Bank: {bank}")
   print(f"Player 0: {player_tokens}")
   ```

4. **Color Enum Mapping:**
   ```python
   colors = ['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK', 'GOLD']
   ```

---

## Glossary

- **Bonus:** Permanent discount provided by a purchased development card
- **Bank:** Central pool of tokens available to all players
- **Development Card:** Card that can be purchased for prestige and bonuses
- **Gold Token:** Wildcard token, only obtained when reserving cards
- **Market:** The 3×4 grid of visible development cards
- **Noble:** Special card worth prestige points, awarded when requirements met
- **Prestige Points:** Victory points, first to 15 triggers end game
- **Reserve:** Action to take a card into hand (max 3)
- **Tier:** Card level (1=cheap, 2=medium, 3=expensive)
- **Token:** Resource used to purchase cards (5 colors + gold)

---

## Version History

- **Current State:** Core game loop functional, nobles not implemented
- **Last Commit:** "finished mostly coding up game rules" (567997d)
- **Previous:** "major improvements on generating game rules" (b351dc2)

---

## Quick Reference

### File Quick Links

| Component | Header | Implementation |
|-----------|--------|----------------|
| Game State | `include/game_state.h:13-47` | `src/game_state.cpp:9-38` |
| Player State | `include/player_state.h:7-18` | `src/player_state.cpp` |
| Move Generator | `include/move_generator.h:7-16` | `src/move_generator.cpp:7-137` |
| Move Executor | `include/move_executor.h:7-22` | `src/move_executor.cpp:10-146` |
| Game Init | `include/game_initializer.h:6-16` | `src/game_initializer.cpp:8-61` |
| Serializer | `include/state_serializer.h:6-18` | `src/state_serializer.cpp` |
| Constants | `include/constants.h:1-24` | - |
| Types | `include/types.h:9-31` | `src/types.cpp` |

### Color Index Reference
```
0: WHITE
1: BLUE
2: GREEN
3: RED
4: BLACK
5: GOLD
```

### Action Type Reference
```
0: RESERVE_CARD
1: BUY_CARD_FROM_MARKET
2: BUY_CARD_FROM_RESERVE
3: GET_3_FROM_BANK
4: GET_2_FROM_BANK
```

---

*End of Documentation*
