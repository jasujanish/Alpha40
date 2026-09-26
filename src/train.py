'''
Plan:
main function
- Take in num_training_games, eval_every, eval_games, device, update_every, num_simulations, buffer size as command line arguments
- Init random network on device, save it as the "best model"
- call train

train
- init AdamW optimizer
- init replay buffer
- for i in range(num_training_games):
    - node = empty board node, toplay = 1
    - while node is not terminal
        - play the game
        - every update_every moves, train the model
    - add all game moves to the buffer (buffer size is num moves in model)
    - every eval_every games, benchmark model performance 

eval
- runs eval_games games between the old best and new best
- update best model if needed (update best model if avg result is greater than 0.05)
- since connect 4 is solved, we can calculate the percentage of moves that were optimal at this checkpoint
- log the percent optimal moves - scatter plot, percent as y and total training moves as x
- log best checkpoint over time - scatter plot, num moves used to train the best checkpoint as y, total training moves as x
'''