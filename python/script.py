from enum import Enum
import splendor_game as s
from termcolor import colored

class Action(Enum):
    RESERVE_CARD=0
    BUY_CARD_FROM_MARKET=1
    BUY_CARD_FROM_RESERVE=2
    GET_3_FROM_BANK=3
    GET_2_FROM_BANK=4 
class Color(Enum):
    WHITE=0
    BLUE=1 
    GREEN=2
    RED=3
    BLACK=4 
    GOLD=5

game = s.SplendorGame(num_players = 3, seed=42)

def print_summary_nicely(summary):
    for k,v in summary.items():
        print(colored(k, 'blue'))
        if k == 'players':
            for i, p in enumerate(v):
                print(colored(f'Player {i + 1}', 'green'))
                for k, v in p.items():
                    print(f'{colored(k,"grey")}: {v}')
        elif k == 'nobles':
            for p in v:
                print(p)
        elif k == 'market':
            for t in v:
                for c in t:
                    print(c)
        else:
            print(v)

def print_legal_moves_nicely(moves):
    reserve_card_moves = [m for m in moves if m[0] == Action.RESERVE_CARD.value]
    buy_card_market = [m for m in moves if m[0] == Action.BUY_CARD_FROM_MARKET.value]
    buy_card_reserve = [m for m in moves if m[0] == Action.BUY_CARD_FROM_RESERVE.value]
    get_3_bank = [m for m in moves if m[0] == Action.GET_3_FROM_BANK.value]
    get_2_bank = [m for m in moves if m[0] == Action.GET_2_FROM_BANK.value]
    print(colored('LEGAL MOVES', 'red'))
    print(colored('reserve card', 'red'))
    [print(m[1]) for m in reserve_card_moves]
    print(colored('buy card from market', 'red'))
    [print(m[1]) for m in buy_card_market]
    print(colored('buy card from reserve', 'red'))
    [print(m[1]) for m in buy_card_reserve]
    print(colored('get 3 different tokens from bank', 'red'))
    [print(m[1]) for m in get_3_bank]
    print(colored('get 2 of same token from bank', 'red'))
    [print(m[1]) for m in get_2_bank]

summary = game.state_summary()
legal_moves = game.legal_moves()


game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])
game.perform_move(game.legal_moves()[0])


# summary = game.state_summary()
# print_summary_nicely(summary)

# legal_moves = game.legal_moves()
# print_legal_moves_nicely(legal_moves)
