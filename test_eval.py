from evaluation_env import RemoteEvaluationEnv
import time
import numpy as np
from PIL import Image

im = Image.open("./train_data/walkable_mask.png")

walkable_mask = np.asarray(im, dtype=bool) == 0

def main():
    # Initialize the evaluation environment
    # Replace these with your actual team_id and transmitter_id
    team_id = "augheyscale"
    transmitter_id = "tx0"
    
    env = RemoteEvaluationEnv(
        team_id=team_id,
        transmitter_id=transmitter_id,
        base_url="https://tx-hunt.distspec.com"
    )
    
    # Start a new walk
    print("Starting new walk...")
    initial_state = env.reset()
    print(f"Initial state: {initial_state}")
    
    # Define some test actions (0=N, 1=S, 2=E, 3=W)
    ALL_ACTIONS = [0, 1, 2, 3]
    DELTA_XY = [(0,1), (0,-1), (1,0), (-1,0)]
    
    # Perform steps with some localization guesses
    for _ in range(100):
        valid_actions = []
        for action in ALL_ACTIONS:
            delta_xy = DELTA_XY[action]
            next_ij = np.array(initial_state["ij"]) + np.array(delta_xy)
            if not walkable_mask[next_ij[1], next_ij[0]]:
                valid_actions.append(action)
        print(f"Valid actions: {valid_actions}")
        
        guess = None
        action = np.random.choice(valid_actions).item()
        
        # Perform the step
        result = env.step(action, circle=guess)
        print(f"Result: {result}")
        
        # Small delay to avoid overwhelming the server
        #time.sleep(0.5)

if __name__ == "__main__":
    main()
