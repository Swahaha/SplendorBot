import splendor_game

g = splendor_game.SplendorGame(num_players = 3, seed=7023)
print(g.state_summary()['market'])

